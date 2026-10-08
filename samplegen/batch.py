"""Make lots of sounds at once: shot lists (overnight batches) and variations of a sample.

Shot list: one sound per line, options after a "|" (or a trailing "x8"):

    door creak x8
    heavy footsteps on gravel | 12x | 3s
    distant thunder rumble | 20s | medium | loop
    # comments and blank lines are ignored
"""

import re
from dataclasses import dataclass, replace

from .audio import ExportFormat
from .catalog import MODELS, build_loop_prompt
from .generation_request import MAX_BATCH_AUDIO_SECONDS, MAX_VARIATIONS, GenerationRequest
from .library import SampleRecord, clean_tags
from .prompting import NUDGES, nudge_prompt

MAX_LINES = 200
MAX_SOUNDS = 1000
DEFAULT_COUNT = 4
COUNT = re.compile(r"^(?:[x×*]\s*(\d+)|(\d+)\s*[x×])$", re.IGNORECASE)
SECONDS = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds|m|min)$", re.IGNORECASE)
TRAILING_COUNT = re.compile(r"\s+[x×*]\s*(\d+)\s*$", re.IGNORECASE)
MODEL_WORDS = {"fast": "sa3-sfx", "small": "sa3-sfx", "sfx": "sa3-sfx", "medium": "sa3-medium", "best": "sa3-medium"}
VARIATION_MODELS = {"sa3-sfx", "sa3-medium", "f1-samples"}  # the ones that can transform


@dataclass(frozen=True)
class Shot:
    prompt: str
    count: int
    seconds: float | None = None
    model: str | None = None
    seamless: bool = False


def parse_shot_list(text: str) -> list[Shot]:
    shots, problems = [], []
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if len(lines) > MAX_LINES:
        raise ValueError(f"Up to {MAX_LINES} lines per batch.")
    for number, line in enumerate(lines, start=1):
        prompt, *options = [part.strip() for part in line.split("|")]
        count, seconds, model, seamless = None, None, None, False
        trailing = TRAILING_COUNT.search(prompt)
        if trailing:
            count, prompt = int(trailing.group(1)), prompt[:trailing.start()].strip()
        for option in filter(None, options):
            if match := COUNT.match(option):
                count = int(match.group(1) or match.group(2))
            elif match := SECONDS.match(option):
                value = float(match.group(1))
                seconds = value * 60 if match.group(2).lower().startswith("m") else value
            elif option.lower() in MODEL_WORDS:
                model = MODEL_WORDS[option.lower()]
            elif option.lower() in ("loop", "seamless"):
                seamless = True
            else:
                problems.append(f"line {number}: don't know “{option}”")
        if not prompt:
            problems.append(f"line {number}: no description")
            continue
        shots.append(Shot(prompt[:500], max(1, min(count or DEFAULT_COUNT, 64)), seconds, model, seamless))
    if problems:
        raise ValueError("Couldn't read the shot list: " + "; ".join(problems[:5]))
    if not shots:
        raise ValueError("The shot list is empty: one sound per line.")
    if sum(s.count for s in shots) > MAX_SOUNDS:
        raise ValueError(f"Up to {MAX_SOUNDS} sounds per batch.")
    return shots


def shot_requests(shots: list[Shot], tag: str, default_model: str = "sa3-sfx", default_seconds: float = 4.0,
                  export: ExportFormat | None = None) -> list[GenerationRequest]:
    """Each shot as one or more jobs: at most 8 takes per job, within the GPU's batch budget."""
    tags = clean_tags([tag]) if tag else ()
    requests = []
    for shot in shots:
        model = shot.model or default_model
        spec = MODELS[model]
        seconds = min(shot.seconds or (max(default_seconds, 8.0) if shot.seamless else default_seconds), spec.max_seconds)
        mode = "free" if seconds > MODELS["sa3-sfx"].max_seconds and model == "sa3-medium" else "sfx"
        per_job = max(1, min(MAX_VARIATIONS, int(MAX_BATCH_AUDIO_SECONDS // (seconds + 3.0))))
        left = shot.count
        while left > 0:
            n = min(per_job, left)
            requests.append(GenerationRequest(
                mode=mode, model=model, prompt=shot.prompt, duration=seconds, variations=n,
                seamless=shot.seamless, title=shot.prompt, tags=tags, export=export or ExportFormat(),
            ))
            left -= n
    return requests


def variation_request(record: SampleRecord, source_id: str, count: int, strength: float,
                      nudge: str | None = None) -> GenerationRequest:
    """More takes like `record`: a gentle Transform of it, keeping its name (so the takes
    export as round robins of it) and its tags. With a `nudge` ("darker"...), the same take
    pushed one way, named after it ("door slam darker")."""
    model = record.model if record.model in VARIATION_MODELS else ("sa3-medium" if record.duration > 30 else "sa3-sfx")
    prompt = (record.params.get("full_prompt") if model == "f1-samples" else None) or record.prompt or record.name
    title = record.name
    if nudge:
        prompt = nudged_prompt(record, model, prompt, nudge)
        base = record.name
        for earlier in NUDGES.values():  # "door darker" nudged again is "door bigger", not "door darker bigger"
            suffix = f" {earlier.label.lower()}"
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        title = f"{base} {NUDGES[nudge].label.lower()}"[:80]
    looping = record.mode == "loop" or bool(record.params.get("loop"))
    request = GenerationRequest(
        mode="transform", model=model, prompt=prompt, source_id=source_id, strength=strength,
        variations=count, source_is_loop=looping, title=title, tags=record.tags,
        normalize_db=-1.0, fade_out_ms=0.0 if looping else 5.0,
    )
    return replace(request, variations=min(count, max(1, int(MAX_BATCH_AUDIO_SECONDS // max(record.duration, 0.5)))))


def nudged_prompt(record: SampleRecord, model: str, prompt: str, nudge: str) -> str:
    """Foundation-1 wants its tags before bars / BPM / key, so a loop's tags are nudged and
    the musical part put back after them."""
    params = record.params
    key, _, scale = str(params.get("key") or "").partition(" ")
    if model == "f1-samples" and params.get("bars") and params.get("bpm") and key and scale:
        tags = nudge_prompt(record.prompt, nudge, tags=True)
        return build_loop_prompt(tags, int(params["bars"]), int(params["bpm"]), key, scale)
    return nudge_prompt(prompt, nudge, tags=model == "f1-samples")
