from samplegen.catalog import MODELS
from samplegen.timing import plan_generation
from samplegen.workflows import build_audio_to_audio, build_inpaint, build_text_to_audio


def nodes_of(graph, class_type):
    return {k: v for k, v in graph.items() if v["class_type"] == class_type}


def test_sa3_graph_uses_model_files_and_sampling():
    spec = MODELS["sa3-sfx"]
    plan = plan_generation(4.0, 44100)
    graph = build_text_to_audio(spec, "door slam", "", plan, seed=7, batch_size=3, job_id="abc")

    (ckpt,) = nodes_of(graph, "CheckpointLoaderSimple").values()
    assert ckpt["inputs"]["ckpt_name"] == spec.checkpoint
    (clip,) = nodes_of(graph, "CLIPLoader").values()
    assert clip["inputs"] == {"clip_name": spec.text_encoder, "type": "stable_audio", "device": "default"}

    (sampler,) = nodes_of(graph, "KSampler").values()
    assert sampler["inputs"]["seed"] == 7
    assert sampler["inputs"]["steps"] == spec.sampling.steps
    assert sampler["inputs"]["cfg"] == spec.sampling.cfg

    (latent,) = nodes_of(graph, "EmptyLatentAudio").values()
    assert latent["inputs"] == {"seconds": plan.latent_seconds, "batch_size": 3}

    (cond,) = nodes_of(graph, "ConditioningStableAudio").values()
    assert cond["inputs"]["seconds_total"] == 4.0

    (save,) = nodes_of(graph, "SamplegenSaveWav").values()
    assert save["inputs"]["job_id"] == "abc"


def test_graph_links_reference_existing_nodes():
    graph = build_text_to_audio(MODELS["f1-samples"], "Pad, Warm", "noise", plan_generation(8.0, 44100), 1, 1, "j")
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


def test_audio_to_audio_graph_encodes_source_and_uses_strength_as_denoise():
    graph = build_audio_to_audio(MODELS["sa3-medium"], "metal", "", "job.wav", 3, 0.4, 5, 3, "j")
    assert graph["load"] == {"class_type": "SamplegenLoadWav", "inputs": {"filename": "job.wav"}}
    assert graph["encode"]["inputs"] == {"audio": ["load", 0], "vae": ["ckpt", 2]}
    assert graph["repeat"]["inputs"] == {"samples": ["encode", 0], "amount": 3}
    assert graph["sampler"]["inputs"]["latent_image"] == ["repeat", 0]
    assert graph["sampler"]["inputs"]["denoise"] == 0.4
    assert "latent" not in graph


def test_inpaint_graph_routes_conditioning_through_inpaint_node():
    graph = build_inpaint(MODELS["sa3-sfx"], "crash", "", "job.wav", 2.5, [(0.5, 1.0), (1.5, 2.0)], 1, 2, "j")
    inpaint = graph["inpaint"]["inputs"]
    assert inpaint["regions"] == "0.5000-1.0000,1.5000-2.0000"
    assert inpaint["samples"] == ["encode", 0] and inpaint["source_seconds"] == 2.5
    sampler = graph["sampler"]["inputs"]
    assert (sampler["positive"], sampler["negative"]) == (["inpaint", 0], ["inpaint", 1])
    assert graph["repeat"]["inputs"]["samples"] == ["inpaint", 2]
    assert sampler["denoise"] == 1.0
    assert graph["timing"]["inputs"]["seconds_total"] == 3.0
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


def test_negative_prompt_is_encoded():
    graph = build_text_to_audio(MODELS["sa3-medium"], "pos", "neg words", plan_generation(5.0, 44100), 1, 1, "j")
    texts = sorted(n["inputs"]["text"] for n in nodes_of(graph, "CLIPTextEncode").values())
    assert texts == ["neg words", "pos"]
