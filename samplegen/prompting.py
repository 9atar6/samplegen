"""Make prompting quick: Explore (a few words -> contrasting full prompts to choose from by
ear) and nudges (the same sound, darker / bigger / drier...). Word lists: prompt_ideas.py."""

import random
import re

from .catalog import LOOP_TAGS
from .prompt_ideas import (
    FREE_ANGLES, FREE_SURPRISES, LOOP_BASS_STRUCTURES, LOOP_FX, LOOP_OPPOSITES, LOOP_STRUCTURES,
    LOOP_TECHNICAL_TIMBRES, NUDGES, SFX_ANGLES, SFX_PACKS, Angle,
)

EXPLORE_MODES = ("sfx", "free", "loop")
EXPLORE_COUNT = 4
MAX_SKETCH_CHARS = 500
MAX_PACK_WORDS = 2  # longer sketches already say what they want: dress them, don't replace them
WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")

__all__ = ["EXPLORE_MODES", "NUDGES", "explore", "nudge_prompt"]


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def explore(sketch: str, mode: str, count: int = EXPLORE_COUNT, rng: random.Random | None = None) -> list[str]:
    """`count` different full prompts from a few words (or from nothing: surprise me)."""
    if mode not in EXPLORE_MODES:
        raise ValueError("Explore works in SFX, Free and Loop modes.")
    if len(sketch) > MAX_SKETCH_CHARS:
        raise ValueError(f"Keep the idea under {MAX_SKETCH_CHARS} characters.")
    rng = rng or random.Random()
    sketch = " ".join(sketch.split())
    if mode == "loop":
        return _loop_ideas(sketch, count, rng)
    if not sketch:
        pool = FREE_SURPRISES if mode == "free" else [p for pack in SFX_PACKS.values() for p in pack]
        return rng.sample(list(pool), count)
    if mode == "sfx" and (pack := _pack_for(sketch)):
        return rng.sample(list(pack), min(count, len(pack)))
    return _dressed(sketch, SFX_ANGLES if mode == "sfx" else FREE_ANGLES, count, rng)


def _pack_for(sketch: str) -> tuple[str, ...] | None:
    words = WORD.findall(sketch.lower())
    if len(words) > MAX_PACK_WORDS:
        return None
    for triggers, pack in SFX_PACKS.items():
        if any(w in triggers for w in words):
            return pack
    return None


def _dressed(sketch: str, angles: tuple[Angle, ...], count: int, rng: random.Random) -> list[str]:
    said = _words(sketch)
    fitting = [a for a in angles if not (a.avoid & said)]
    if len(fitting) < count:  # a sketch that rules out nearly everything: contrast beats consistency
        fitting += [a for a in angles if a not in fitting]
    return [a.template.format(x=sketch) for a in rng.sample(fitting, count)]


# ---------- Loop mode: Foundation-1 tags ----------

_GROUP_OF = {tag.lower(): (group, tag) for group, tags in LOOP_TAGS.items() for tag in tags}
_CHARACTER_TIMBRES = tuple(t for t in LOOP_TAGS["Timbre"] if t not in LOOP_TECHNICAL_TIMBRES)


def _read_tags(sketch: str) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    """The sketch as (exact tags by group, tag families by group, words Foundation won't know).
    "dark bass" -> Dark (Timbre), every "... Bass" (Instrument family)."""
    exact: dict[str, list[str]] = {}
    families: dict[str, list[str]] = {}
    unknown: list[str] = []
    for part in filter(None, (p.strip() for p in sketch.split(","))):
        if part.lower() in _GROUP_OF:
            group, tag = _GROUP_OF[part.lower()]
            exact.setdefault(group, []).append(tag)
            continue
        for word in part.split():
            low = word.lower()
            if low in _GROUP_OF:
                group, tag = _GROUP_OF[low]
                exact.setdefault(group, []).append(tag)
                continue
            found = False
            for group, tags in LOOP_TAGS.items():
                family = [t for t in tags if low in t.lower().split()]
                if family:
                    families.setdefault(group, []).extend(family)
                    found = True
            if not found:
                unknown.append(word)
    return exact, families, unknown


def _clashes(tag: str, chosen: list[str]) -> bool:
    return any(tag in pair and (pair - {tag}) & set(chosen) for pair in LOOP_OPPOSITES)


def _pick(pool, chosen: list[str], rng: random.Random) -> str | None:
    options = [t for t in pool if t not in chosen and not _clashes(t, chosen)]
    return rng.choice(options) if options else None


def _loop_idea(exact, families, unknown, instrument: str | None, fx: str, rng: random.Random) -> list[str]:
    tags: list[str] = []
    for group in LOOP_TAGS:  # what the user said, in Foundation's order
        tags += [t for t in exact.get(group, []) if t not in tags]
    if instrument:
        tags.insert(0, instrument)
    for group in ("Timbre", "FX", "Structure"):
        if families.get(group) and not exact.get(group):
            tags.append(rng.choice(families[group]))
    for _ in range(2 - len(exact.get("Timbre", []))):
        if tag := _pick(_CHARACTER_TIMBRES, tags, rng):
            tags.append(tag)
    if not exact.get("FX") and not families.get("FX"):
        tags.append(fx)
    if not exact.get("Structure") and not families.get("Structure"):
        bass = any(t.endswith("Bass") for t in tags)
        tags.append(_pick(LOOP_BASS_STRUCTURES if bass else LOOP_STRUCTURES, tags, rng) or "Simple")
    return tags + [w for w in unknown if w.lower() not in {t.lower() for t in tags}]


def _loop_ideas(sketch: str, count: int, rng: random.Random) -> list[str]:
    exact, families, unknown = _read_tags(sketch)
    # A different instrument for each idea when the sketch leaves it open (or names a family).
    pool = [] if exact.get("Instrument") else list(dict.fromkeys(families.get("Instrument") or LOOP_TAGS["Instrument"]))
    instruments = rng.sample(pool, min(count, len(pool))) if pool else []
    effects = rng.sample(LOOP_FX, min(count, len(LOOP_FX)))
    ideas: list[str] = []
    for attempt in range(count * 20):  # distinct ideas; a narrow sketch may allow only a few
        instrument = instruments[attempt % len(instruments)] if instruments else None
        idea = ", ".join(_loop_idea(exact, families, unknown, instrument, effects[attempt % len(effects)], rng))
        if idea not in ideas:
            ideas.append(idea)
        if len(ideas) == count:
            break
    return ideas


# ---------- nudges ----------

def nudge_prompt(prompt: str, nudge: str, tags: bool = False) -> str:
    """The prompt pushed one way: words that say the opposite go, the nudge's words come in.
    In a description the first part is the sound itself, so only the clashing words leave it."""
    if nudge not in NUDGES:
        raise ValueError(f"Unknown nudge “{nudge}”.")
    spec = NUDGES[nudge]
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    kept: list[str] = []
    for index, part in enumerate(parts):
        clash = _words(part) & spec.against
        if not clash:
            kept.append(part)
        elif tags and part.lower() in {t.lower() for t in LOOP_TAGS["Instrument"]}:
            kept.append(part)  # never drop the instrument
        elif index == 0 and not tags:
            trimmed = " ".join(w for w in part.split() if w.lower().strip(".!?") not in clash)
            if trimmed:
                kept.append(trimmed)
    have = {p.lower() for p in kept}
    added = [w.strip() for w in (spec.tags if tags else spec.words).split(",")]
    return ", ".join(kept + [w for w in added if w.lower() not in have])
