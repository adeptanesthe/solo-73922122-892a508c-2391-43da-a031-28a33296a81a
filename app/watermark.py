"""密钥水印：确定性绿色集续写生成 与 粘贴文本逐位置核查。

绿色集构造（每步、确定性）：
  对 95 个可打印 ASCII 字符 c，计算
      score(c) = HMAC-SHA256(key, prefix ‖ 0x1F ‖ c)
  按 score 降序取前 47 个为绿色集（HMAC 输出碰撞实际不可能；
  万一并列，稳定排序保留码点较小者，结果仍确定）。
  全程只依赖密钥与完整已确认前缀，不依赖任何进程内随机状态，
  因此相同输入在服务重启后仍可逐位复现。

生成（固定目标模型）：
  每步对完整前缀做一次无缓存整序列前向（use_cache=False，不触碰 KV），
  在 绿色字符 ∪ {EOS} 中取 logit 最高者（平局取词表序号小者）；
  选中 EOS 即停止。整个生成包在 target.preserved_state() 内，
  不改变原解码流程的权重、KV 缓存与前向计数。

核查（粘贴文本）：
  逐位置 i 用 粘贴提示词 + 粘贴续写的前 i 个字符 重建绿色集，
  统计续写字符落入绿色集的次数 hits；在"无水印"零假设下每位置
  命中概率为 47/95，给出二项尾概率 P(X ≥ hits)。
  P ≤ 0.01 仅标记"水印线索"，否则"证据不足"——不宣称确定来源。
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass, field

import torch

from . import tokenizer
from .model import CausalTransformer

# 95 个可打印 ASCII 字符中每步固定 47 个绿色字符
PRINTABLE = tuple(chr(code) for code in range(0x20, 0x7F))  # 95 个
GREEN_SIZE = 47
GREEN_RATE = GREEN_SIZE / len(PRINTABLE)  # 47/95，零假设命中率
# 前缀与候选字符之间的分隔字节：可打印 ASCII 不含 0x1F，避免拼接歧义
_SEP = b"\x1f"
# 判定阈值：二项尾概率不超过该值时标记"水印线索"
CLUE_THRESHOLD = 0.01

VERDICT_CLUE = "水印线索"
VERDICT_INSUFFICIENT = "证据不足"


def green_set(key: str, prefix_text: str) -> frozenset[str]:
    """由密钥与完整已确认前缀确定性地导出 47 个绿色字符。"""
    key_bytes = key.encode("utf-8")
    prefix_bytes = prefix_text.encode("utf-8")
    scored: list[tuple[bytes, str]] = [
        (hmac.new(key_bytes, prefix_bytes + _SEP + ch.encode("ascii"),
                  hashlib.sha256).digest(), ch)
        for ch in PRINTABLE
    ]
    # 按摘要降序；list 按 PRINTABLE 码点升序构建 + 稳定排序，
    # 摘要并列时码点较小者排前，取前 47 个完全确定。
    scored.sort(key=lambda item: item[0], reverse=True)
    return frozenset(ch for _, ch in scored[:GREEN_SIZE])


def binomial_tail(n: int, k: int, p: float) -> float:
    """P(X ≥ k)，X ~ Binomial(n, p)。n ≤ 24，直接精确求和。"""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    q = 1.0 - p
    return float(sum(math.comb(n, i) * p ** i * q ** (n - i)
                     for i in range(k, n + 1)))


@dataclass
class GenerateResult:
    prompt: str
    text: str                       # 续写文本（不含 EOS）
    tokens: list[int]               # 续写字符的 token id（不含 EOS）
    stopped_by_eos: bool
    steps: int                      # 实际执行的解码步数（含选中 EOS 的那一步）


@dataclass
class VerifyResult:
    hits: int
    total: int
    p_value: float
    verdict: str
    positions: list[dict] = field(default_factory=list)  # index/char/in_green


def generate(target: CausalTransformer, prompt_text: str, key: str,
             max_new: int) -> GenerateResult:
    """用固定目标模型生成带密钥水印的续写。

    每步：绿色集 ← (key, 完整已确认前缀)；在 绿色字符 ∪ {EOS} 中取
    logit 最高者（平局取词表序号小者）；选中 EOS 停止。
    全程无缓存整序列前向且包在 preserved_state 内：不改权重、
    不改 KV 缓存、不留前向计数痕迹。
    """
    seq = tokenizer.encode(prompt_text)  # 入参已校验为可打印 ASCII，无 <unk>
    prefix_text = prompt_text
    out_ids: list[int] = []
    stopped = False
    steps = 0
    with target.preserved_state():
        for _ in range(max_new):
            steps += 1
            green = green_set(key, prefix_text)
            cand = sorted(tokenizer.char_id(ch) for ch in green)
            cand.append(tokenizer.EOS_ID)  # EOS 恒为候选
            logits = target.forward(seq, use_cache=False)
            last = logits[-1]
            # 候选内 argmax；cand 升序，torch.argmax 平局取首个 = 序号小者
            nxt = cand[int(torch.argmax(last[cand]).item())]
            if nxt == tokenizer.EOS_ID:
                stopped = True
                break
            seq.append(nxt)
            ch = tokenizer.render_token(nxt)
            prefix_text += ch
            out_ids.append(nxt)
    return GenerateResult(
        prompt=prompt_text,
        text=tokenizer.decode(out_ids),
        tokens=out_ids,
        stopped_by_eos=stopped,
        steps=steps,
    )


def verify(prompt_text: str, key: str, continuation: str) -> VerifyResult:
    """逐位置按粘贴文本的实际前缀重建绿色集并统计命中。

    位置 i 的绿色集由 (key, prompt + continuation[:i]) 决定——完全以
    粘贴回来的文本为准，与当初生成的中间状态无关。
    """
    positions: list[dict] = []
    hits = 0
    for i, ch in enumerate(continuation):
        in_green = ch in green_set(key, prompt_text + continuation[:i])
        hits += 1 if in_green else 0
        positions.append({"index": i, "char": ch, "in_green": in_green})
    total = len(continuation)
    p_value = binomial_tail(total, hits, GREEN_RATE)
    verdict = VERDICT_CLUE if p_value <= CLUE_THRESHOLD else VERDICT_INSUFFICIENT
    return VerifyResult(
        hits=hits,
        total=total,
        p_value=p_value,
        verdict=verdict,
        positions=positions,
    )
