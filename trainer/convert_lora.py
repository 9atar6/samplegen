"""Convert a Stable Audio 3 trainer LoRA checkpoint into a ComfyUI LoRA file.

Runs inside the trainer's Python environment (it needs torch to read .ckpt files):
    trainer\\stable-audio-3\\.venv\\Scripts\\python.exe trainer\\convert_lora.py \\
        --ckpt <path to .ckpt or .safetensors> --base <ComfyUI base checkpoint .safetensors> --out <file.safetensors>

Only plain `lora` adapters on the diffusion transformer convert exactly
(W' = W + alpha/rank * B @ A, same maths in both tools). Trainer keys look like
    model.transformer.layers.0.self_attn.to_qkv.parametrizations.weight.0.lora_A
ComfyUI's generic format wants
    diffusion_model.transformer.layers.0.self_attn.to_qkv.lora_A   (+ .lora_B, .alpha)
Every target layer and every shape is checked against the base checkpoint before
anything is written, so a mismatch fails loudly instead of producing a silent no-op.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

PARAM_MARK = ".parametrizations.weight.0."
TRAINER_PREFIX = "model."          # keys are relative to the DiT wrapper
CHECKPOINT_PREFIX = "model.model."  # how the base .safetensors names the same layers
COMFY_PREFIX = "diffusion_model."


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def load_trainer_checkpoint(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Needs torch (trainer environment). Returns float32 numpy arrays + the embedded lora_config."""
    import torch
    from safetensors import safe_open

    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as f:
            state = {k: f.get_tensor(k) for k in f.keys()}
            config = json.loads((f.metadata() or {}).get("lora_config", "{}"))
    else:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        state, config = ckpt["state_dict"], ckpt.get("lora_config", {})
    return {k: v.detach().float().cpu().numpy() for k, v in state.items()}, config


def convert(state: dict[str, np.ndarray], config: dict, base_header: dict) -> tuple[dict, list[str]]:
    """Pure numpy: trainer LoRA tensors -> ComfyUI LoRA tensors (checked against the base model)."""
    adapter = config.get("adapter_type", "lora")
    if adapter != "lora":
        raise SystemExit(f"Only 'lora' adapters convert exactly; this checkpoint is '{adapter}'.")
    alpha = float(config.get("alpha", config.get("rank", 1)))

    layers: dict[str, dict] = {}
    skipped = []
    for key, tensor in state.items():
        if PARAM_MARK not in key:
            continue
        module, kind = key.split(PARAM_MARK)
        if not module.startswith(TRAINER_PREFIX):
            skipped.append(key)  # e.g. conditioner layers; ComfyUI loads the text encoder separately
            continue
        layers.setdefault(module[len(TRAINER_PREFIX):], {})[kind] = tensor

    out = {}
    for path, parts in sorted(layers.items()):
        if set(parts) != {"lora_A", "lora_B"}:
            raise SystemExit(f"{path}: expected lora_A and lora_B, found {sorted(parts)}")
        base_key = f"{CHECKPOINT_PREFIX}{path}.weight"
        if base_key not in base_header:
            raise SystemExit(f"{path}: no such layer in the base model ({base_key})")
        fan_out, fan_in = base_header[base_key]["shape"][:2]
        down = np.asarray(parts["lora_A"], dtype=np.float32)  # A: (rank, in)
        up = np.asarray(parts["lora_B"], dtype=np.float32)    # B: (out, rank)
        if down.ndim != 2 or up.ndim != 2:
            raise SystemExit(f"{path}: expected 2-D LoRA matrices")
        # The trainer only adds LoRA to nn.Linear layers (A: rank x in, B: out x rank). Anything
        # else (e.g. transposed embedding adapters) would need different maths: refuse, don't guess.
        if down.shape[1] != fan_in or up.shape[0] != fan_out or down.shape[0] != up.shape[1]:
            raise SystemExit(f"{path}: LoRA shapes {down.shape}/{up.shape} don't fit weight ({fan_out}, {fan_in})")
        target = f"{COMFY_PREFIX}{path}"
        out[f"{target}.lora_A"] = down.astype(np.float16)
        out[f"{target}.lora_B"] = up.astype(np.float16)
        out[f"{target}.alpha"] = np.array(alpha, dtype=np.float32)
    if not out:
        raise SystemExit("No diffusion-model LoRA layers found in this checkpoint.")
    return out, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from safetensors.numpy import save_file

    state, config = load_trainer_checkpoint(args.ckpt)
    tensors, skipped = convert(state, config, read_header(args.base))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".part")
    save_file(tensors, str(tmp), metadata={"source": str(args.ckpt.name), "lora_config": json.dumps(config)})
    tmp.replace(args.out)
    layers = len(tensors) // 3
    print(json.dumps({"ok": True, "layers": layers, "skipped": len(skipped), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
