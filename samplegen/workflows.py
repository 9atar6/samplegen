"""Builds ComfyUI API-format graphs for the models in the catalog."""

import math

from .catalog import ModelSpec

OUTPUT_NODE = "SamplegenSaveWav"
LOAD_NODE = "SamplegenLoadWav"
INPAINT_NODE = "SamplegenInpaintConditioning"


def _base(spec: ModelSpec, prompt: str, negative_prompt: str, conditioning_seconds: int) -> dict:
    """Model + text conditioning (with duration) shared by every graph."""
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": spec.checkpoint}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": spec.text_encoder, "type": "stable_audio", "device": "default",
        }},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["clip", 0]}},
        "timing": {"class_type": "ConditioningStableAudio", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0],
            "seconds_start": 0.0, "seconds_total": float(conditioning_seconds),
        }},
    }


def _sample_and_save(spec: ModelSpec, positive: list, negative: list, latent: list, seed: int,
                     denoise: float, job_id: str) -> dict:
    return {
        "sampler": {"class_type": "KSampler", "inputs": {
            "model": ["ckpt", 0], "positive": positive, "negative": negative,
            "latent_image": latent, "seed": seed, "denoise": denoise,
            "steps": spec.sampling.steps, "cfg": spec.sampling.cfg,
            "sampler_name": spec.sampling.sampler_name, "scheduler": spec.sampling.scheduler,
        }},
        "decode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": OUTPUT_NODE, "inputs": {"audio": ["decode", 0], "job_id": job_id}},
    }


def _encoded_source(filename: str) -> dict:
    return {
        "load": {"class_type": LOAD_NODE, "inputs": {"filename": filename}},
        "encode": {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["load", 0], "vae": ["ckpt", 2]}},
    }


def build_text_to_audio(spec: ModelSpec, prompt: str, negative_prompt: str, plan, seed: int,
                        batch_size: int, job_id: str, lora: tuple[str, float] | None = None) -> dict:
    """`lora` = (file name in models/loras, strength) applies a trained style to the model."""
    graph = _base(spec, prompt, negative_prompt, plan.conditioning_seconds)
    graph["latent"] = {"class_type": "EmptyLatentAudio", "inputs": {
        "seconds": plan.latent_seconds, "batch_size": batch_size,
    }}
    graph.update(_sample_and_save(spec, ["timing", 0], ["timing", 1], ["latent", 0], seed, 1.0, job_id))
    if lora:
        lora_name, strength = lora
        graph["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["ckpt", 0], "lora_name": lora_name, "strength_model": float(strength),
        }}
        graph["sampler"]["inputs"]["model"] = ["lora", 0]
    return graph


def build_audio_to_audio(spec: ModelSpec, prompt: str, negative_prompt: str, source_filename: str,
                         conditioning_seconds: int, strength: float, seed: int, batch_size: int,
                         job_id: str) -> dict:
    """Re-noise the encoded source by `strength` (0..1) and denoise toward the prompt."""
    graph = _base(spec, prompt, negative_prompt, conditioning_seconds)
    graph.update(_encoded_source(source_filename))
    graph["repeat"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["encode", 0], "amount": batch_size}}
    graph.update(_sample_and_save(spec, ["timing", 0], ["timing", 1], ["repeat", 0], seed, strength, job_id))
    return graph


def build_inpaint(spec: ModelSpec, prompt: str, negative_prompt: str, source_filename: str,
                  source_seconds: float, regions: list[tuple[float, float]], seed: int, batch_size: int,
                  job_id: str) -> dict:
    """Stable Audio 3 inpainting: regenerate `regions` (seconds), keep the rest as context."""
    graph = _base(spec, prompt, negative_prompt, max(1, math.ceil(source_seconds)))
    graph.update(_encoded_source(source_filename))
    graph["inpaint"] = {"class_type": INPAINT_NODE, "inputs": {
        "positive": ["timing", 0], "negative": ["timing", 1], "samples": ["encode", 0],
        "source_seconds": source_seconds,
        "regions": ",".join(f"{start:.4f}-{end:.4f}" for start, end in regions),
    }}
    graph["repeat"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["inpaint", 2], "amount": batch_size}}
    graph.update(_sample_and_save(spec, ["inpaint", 0], ["inpaint", 1], ["repeat", 0], seed, 1.0, job_id))
    return graph
