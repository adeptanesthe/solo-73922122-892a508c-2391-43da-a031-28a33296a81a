"""密钥水印关键功能测试：

  1. 绿色集确定性：同密钥同前缀结果相同、恒为 47 个、跨进程可复现；
  2. 生成可复现：同一进程重复生成、以及重建模型（模拟服务重启）后逐位一致；
  3. 状态保全：水印生成不改变目标模型的权重、KV 缓存与前向计数；
  4. 自生成文本核查：全部位置命中，尾概率 ≤ 0.01，判「水印线索」；
  5. 核查按粘贴文本实际前缀重建绿色集（改动前缀后后续位置不再必然命中）；
  6. 二项尾概率数值正确性与判定阈值；
  7. EOS 停止与 API 入参校验（400 不得返回结论字段）。
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import tokenizer, watermark
from app.decoding import speculative_decode
from app.model import _build_pair, get_models

torch.set_num_threads(1)
DRAFT, TARGET = get_models()

PROMPT = "the capital of france is "
KEY = "demo-key-2026"
WRONG_KEY = "demo-key-2027"


def test_green_set_deterministic() -> None:
    g1 = watermark.green_set(KEY, PROMPT)
    g2 = watermark.green_set(KEY, PROMPT)
    assert g1 == g2 and len(g1) == 47
    assert all(0x20 <= ord(c) <= 0x7E for c in g1)
    # 前缀或密钥不同，绿色集应不同（确定性函数，断言安全）
    assert watermark.green_set(KEY, PROMPT + "x") != g1
    assert watermark.green_set(WRONG_KEY, PROMPT) != g1
    # 与随机模块无关：不 seed 任何 RNG，结果仅由 (key, prefix) 决定
    print("ok: 绿色集确定可复现，恒为 95 选 47，随密钥/前缀变化")


def test_generate_reproducible_across_restart() -> None:
    r1 = watermark.generate(TARGET, PROMPT, KEY, 24)
    r2 = watermark.generate(TARGET, PROMPT, KEY, 24)
    assert r1.text == r2.text and r1.tokens == r2.tokens
    # 模拟重启：用固定种子全新构建模型，生成结果必须逐位一致
    _fresh_draft, fresh_target = _build_pair()
    r3 = watermark.generate(fresh_target, PROMPT, KEY, 24)
    assert r1.text == r3.text and r1.tokens == r3.tokens
    assert len(r1.text) <= 24
    print(f"ok: 相同输入重启后逐位复现，续写={r1.text!r}（{len(r1.text)} 字符）")


def test_generate_preserves_model_state() -> None:
    """水印生成不得改变原解码的权重、KV 缓存与前向计数。"""
    # 先跑一轮推测解码，让目标模型留下真实的 KV 缓存与计数
    ids = tokenizer.encode("hello world, this is a demo of ")
    speculative_decode(DRAFT, TARGET, ids, 3, 20)
    saved_count = TARGET.forward_count
    saved_len = TARGET.cache_len
    saved_cache = [(k.clone(), v.clone()) for k, v in TARGET._cache]
    saved_weight_sums = [p.sum().item() for p in TARGET.parameters()]

    watermark.generate(TARGET, PROMPT, KEY, 24)

    assert TARGET.forward_count == saved_count, "前向计数被水印操作改变"
    assert TARGET.cache_len == saved_len, "KV 缓存长度被水印操作改变"
    for (k0, v0), (k1, v1) in zip(saved_cache, TARGET._cache):
        assert torch.equal(k0, k1) and torch.equal(v0, v1), "KV 缓存内容被改变"
    now_sums = [p.sum().item() for p in TARGET.parameters()]
    assert now_sums == saved_weight_sums, "模型权重被水印操作改变"
    print(f"ok: 水印生成后权重/KV/前向计数逐位未变（计数={saved_count}，"
          f"缓存长度={saved_len}）")


def test_self_generated_text_all_hits_and_clue() -> None:
    r = watermark.generate(TARGET, PROMPT, KEY, 24)
    assert len(r.text) > 0
    v = watermark.verify(PROMPT, KEY, r.text)
    assert v.hits == v.total == len(r.text)
    assert all(p["in_green"] for p in v.positions)
    # 24 个位置全命中：P = (47/95)^24 ≈ 4.6e-8，必然 ≤ 0.01
    assert v.p_value <= watermark.CLUE_THRESHOLD
    assert v.verdict == watermark.VERDICT_CLUE
    print(f"ok: 自生成续写 {v.hits}/{v.total} 全命中，"
          f"尾概率={v.p_value:.3e}，判「{v.verdict}」")


def test_wrong_key_insufficient() -> None:
    r = watermark.generate(TARGET, PROMPT, KEY, 24)
    v = watermark.verify(PROMPT, WRONG_KEY, r.text)
    # 错密钥下命中应接近 47/95 的随机水平（确定性用例，断言安全）
    assert v.hits < v.total
    assert v.verdict == watermark.VERDICT_INSUFFICIENT
    print(f"ok: 错密钥命中 {v.hits}/{v.total}，尾概率={v.p_value:.4f}，"
          f"判「{v.verdict}」")


def test_verify_uses_pasted_prefix() -> None:
    """逐位置按粘贴文本的实际前缀重建：改动中间字符后，前面位置仍命中。"""
    r = watermark.generate(TARGET, PROMPT, KEY, 24)
    text = r.text
    assert len(text) >= 8
    cut = 5
    altered = text[:cut] + ("!" if text[cut] != "!" else "?") + text[cut + 1:]
    v = watermark.verify(PROMPT, KEY, altered)
    # 前 cut 个位置的前缀未受改动影响，必须全部命中
    assert all(p["in_green"] for p in v.positions[:cut])
    assert v.total == len(text)
    print(f"ok: 粘贴文本第 {cut} 位被改动，前 {cut} 位仍全命中，"
          f"后续按新前缀重建（总命中 {v.hits}/{v.total}）")


def test_binomial_tail_math() -> None:
    p = watermark.GREEN_RATE
    assert watermark.binomial_tail(0, 0, p) == 1.0
    assert watermark.binomial_tail(10, 0, p) == 1.0
    assert watermark.binomial_tail(10, 11, p) == 0.0
    # 与独立递推实现交叉验证
    def ref(n: int, k: int) -> float:
        total = 0.0
        for i in range(k, n + 1):
            total += math.comb(n, i) * p ** i * (1 - p) ** (n - i)
        return total
    for n in (1, 5, 12, 24):
        for k in range(n + 2):
            assert math.isclose(watermark.binomial_tail(n, k, p), ref(n, k),
                                rel_tol=1e-12, abs_tol=1e-15)
    # 阈值行为：全命中时 n=6 仍 > 0.01（证据不足），n=7 起 ≤ 0.01（水印线索）
    assert watermark.binomial_tail(6, 6, p) > watermark.CLUE_THRESHOLD
    assert watermark.binomial_tail(7, 7, p) <= watermark.CLUE_THRESHOLD
    assert watermark.binomial_tail(24, 24, p) <= watermark.CLUE_THRESHOLD
    # 命中数越多尾概率越小（单调性）
    seq = [watermark.binomial_tail(24, k, p) for k in range(25)]
    assert all(seq[i] >= seq[i + 1] for i in range(24))
    print("ok: 二项尾概率与递推参考一致，阈值 0.01 的判定边界正确")


def test_eos_stop() -> None:
    """强制 EOS 为最大 logit：首步即停、续写为空，且状态仍被保全。"""
    saved_count = TARGET.forward_count

    def eos_hook(module, inp, out):
        out2 = out.clone()
        out2[..., tokenizer.EOS_ID] = out2.max().item() + 100.0
        return out2

    handle = TARGET.lm_head.register_forward_hook(eos_hook)
    try:
        r = watermark.generate(TARGET, PROMPT, KEY, 24)
    finally:
        handle.remove()
    assert r.stopped_by_eos and r.text == "" and r.tokens == [] and r.steps == 1
    assert TARGET.forward_count == saved_count, "EOS 提前停止也要恢复前向计数"
    # 空续写核查：0 个位置，尾概率 1.0，证据不足
    v = watermark.verify(PROMPT, KEY, "")
    assert v.hits == v.total == 0 and v.p_value == 1.0
    assert v.verdict == watermark.VERDICT_INSUFFICIENT
    print("ok: 首步选中 EOS 即停止，空续写核查判「证据不足」，计数已恢复")


def test_api_endpoints() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    # 生成 → 核查闭环
    g = client.post("/api/watermark/generate",
                    json={"prompt": PROMPT, "key": KEY, "max_new": 24})
    assert g.status_code == 200, g.text
    text = g.json()["text"]
    v = client.post("/api/watermark/verify",
                    json={"prompt": PROMPT, "key": KEY, "continuation": text})
    assert v.status_code == 200, v.text
    body = v.json()
    assert body["hits"] == body["total"] == len(text)
    assert body["verdict"] == "水印线索" and body["p_value"] <= 0.01

    # 入参校验：全部返回 400 且不含任何结论字段
    bad_cases = [
        ("/api/watermark/generate", {"prompt": "", "key": KEY, "max_new": 8}),
        ("/api/watermark/generate", {"prompt": "x" * 33, "key": KEY, "max_new": 8}),
        ("/api/watermark/generate", {"prompt": PROMPT, "key": "short", "max_new": 8}),
        ("/api/watermark/generate", {"prompt": PROMPT, "key": KEY, "max_new": 0}),
        ("/api/watermark/generate", {"prompt": PROMPT, "key": KEY, "max_new": 25}),
        ("/api/watermark/generate", {"prompt": "汉字提示", "key": KEY, "max_new": 8}),
        ("/api/watermark/verify", {"prompt": PROMPT, "key": KEY,
                                   "continuation": "y" * 25}),
        ("/api/watermark/verify", {"prompt": PROMPT, "key": KEY,
                                   "continuation": "a\tb"}),
    ]
    for url, payload in bad_cases:
        r = client.post(url, json=payload)
        assert r.status_code == 400, (url, payload, r.status_code)
        data = r.json()
        assert "error" in data and "verdict" not in data and "text" not in data
    # 页面可达
    assert client.get("/watermark").status_code == 200
    print(f"ok: API 闭环与 {len(bad_cases)} 组非法入参校验通过（400 无结论字段）")


if __name__ == "__main__":
    test_green_set_deterministic()
    test_generate_reproducible_across_restart()
    test_generate_preserves_model_state()
    test_self_generated_text_all_hits_and_clue()
    test_wrong_key_insufficient()
    test_verify_uses_pasted_prefix()
    test_binomial_tail_math()
    test_eos_stop()
    test_api_endpoints()
    print("\n水印测试全部通过 ✔")
