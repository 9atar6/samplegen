"""The trainer -> ComfyUI LoRA conversion (pure numpy part; torch only loads real checkpoints)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "convert_lora", Path(__file__).resolve().parents[1] / "trainer" / "convert_lora.py")
convert_lora = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert_lora)

HEADER = {
    "model.model.transformer.layers.0.self_attn.to_qkv.weight": {"shape": [3072, 1024], "dtype": "F32"},
    "model.model.transformer.layers.0.ff.ff.2.weight": {"shape": [1024, 4096], "dtype": "F32"},
}
MARK = ".parametrizations.weight.0."


def lora_state(rank=4):
    rng = np.random.default_rng(0)
    return {
        f"model.transformer.layers.0.self_attn.to_qkv{MARK}lora_A": rng.standard_normal((rank, 1024)),
        f"model.transformer.layers.0.self_attn.to_qkv{MARK}lora_B": rng.standard_normal((3072, rank)),
        f"model.transformer.layers.0.ff.ff.2{MARK}lora_A": rng.standard_normal((rank, 4096)),
        f"model.transformer.layers.0.ff.ff.2{MARK}lora_B": rng.standard_normal((1024, rank)),
        f"conditioners.prompt.model.encoder.q{MARK}lora_A": rng.standard_normal((rank, 8)),
    }


def test_converts_keys_shapes_and_alpha():
    out, skipped = convert_lora.convert(lora_state(), {"adapter_type": "lora", "rank": 4, "alpha": 8}, HEADER)
    qkv = "diffusion_model.transformer.layers.0.self_attn.to_qkv"
    assert out[f"{qkv}.lora_A"].shape == (4, 1024) and out[f"{qkv}.lora_A"].dtype == np.float16
    assert out[f"{qkv}.lora_B"].shape == (3072, 4)
    assert float(out[f"{qkv}.alpha"]) == 8.0
    assert len(out) == 6  # two layers x (A, B, alpha)
    assert skipped == [f"conditioners.prompt.model.encoder.q{MARK}lora_A"]


def test_delta_matches_trainer_maths():
    state = lora_state()
    out, _ = convert_lora.convert(state, {"adapter_type": "lora", "rank": 4, "alpha": 4}, HEADER)
    qkv = "diffusion_model.transformer.layers.0.self_attn.to_qkv"
    trainer_delta = state[f"model.transformer.layers.0.self_attn.to_qkv{MARK}lora_B"] @ \
        state[f"model.transformer.layers.0.self_attn.to_qkv{MARK}lora_A"]
    comfy_delta = out[f"{qkv}.lora_B"].astype(np.float32) @ out[f"{qkv}.lora_A"].astype(np.float32)
    np.testing.assert_allclose(comfy_delta, trainer_delta, rtol=2e-2, atol=5e-2)


def test_transposed_adapters_are_refused_not_guessed():
    state = {
        f"model.transformer.layers.0.ff.ff.2{MARK}lora_A": np.ones((4096, 4)),  # embedding-style layout
        f"model.transformer.layers.0.ff.ff.2{MARK}lora_B": np.ones((4, 1024)),
    }
    with pytest.raises(SystemExit, match="don't fit"):
        convert_lora.convert(state, {"adapter_type": "lora", "rank": 4}, HEADER)


@pytest.mark.parametrize("state, config, message", [
    (lora_state(), {"adapter_type": "dora-rows"}, "Only 'lora'"),
    ({f"model.transformer.layers.9.x{MARK}lora_A": np.ones((4, 4)), f"model.transformer.layers.9.x{MARK}lora_B": np.ones((4, 4))},
     {"adapter_type": "lora"}, "no such layer"),
    ({f"model.transformer.layers.0.ff.ff.2{MARK}lora_A": np.ones((4, 7)), f"model.transformer.layers.0.ff.ff.2{MARK}lora_B": np.ones((1024, 4))},
     {"adapter_type": "lora"}, "don't fit"),
    ({f"model.transformer.layers.0.ff.ff.2{MARK}lora_A": np.ones((4, 4096))}, {"adapter_type": "lora"}, "expected lora_A and lora_B"),
    ({}, {"adapter_type": "lora"}, "No diffusion-model LoRA"),
])
def test_refuses_what_it_cannot_convert_exactly(state, config, message):
    with pytest.raises(SystemExit, match=message):
        convert_lora.convert(state, config, HEADER)


def test_reads_safetensors_headers(tmp_path):
    import json
    import struct
    header = json.dumps({"__metadata__": {}, "a.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}).encode()
    path = tmp_path / "base.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(24))
    assert convert_lora.read_header(path) == {"a.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
