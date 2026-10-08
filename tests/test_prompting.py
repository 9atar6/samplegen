import random

import pytest

from samplegen.catalog import LOOP_TAGS
from samplegen.prompting import EXPLORE_MODES, NUDGES, explore, nudge_prompt

LOOP_VOCAB = {t.lower() for tags in LOOP_TAGS.values() for t in tags}


def ideas(sketch, mode="sfx", seed=1):
    return explore(sketch, mode, rng=random.Random(seed))


def test_explore_a_bare_noun_gives_contrasting_takes_on_it():
    out = ideas("door")
    assert len(out) == 4 and len(set(out)) == 4
    assert all("door" in p.lower() for p in out)


def test_explore_keeps_a_detailed_sketch_word_for_word():
    sketch = "door creak in an old house"
    out = ideas(sketch)
    assert len(set(out)) == 4
    assert all(sketch in p for p in out)


def test_explore_never_contradicts_the_sketch():
    for seed in range(20):
        for prompt in ideas("glass shatter with long reverb", seed=seed):
            assert "dry" not in prompt and "close-miked" not in prompt


def test_explore_varies_between_presses():
    assert {tuple(ideas("whoosh", seed=s)) for s in range(5)} != {tuple(ideas("whoosh", seed=0))}


def test_explore_with_nothing_typed_suggests_sounds():
    out = ideas("   ")
    assert len(set(out)) == 4 and all(len(p) > 10 for p in out)


def test_explore_free_mode_leans_to_textures():
    out = ideas("ocean", mode="free")
    assert len(set(out)) == 4 and all("ocean" in p for p in out)


def test_explore_loop_speaks_foundation_tags():
    for seed in range(10):
        out = ideas("dark bass", mode="loop", seed=seed)
        assert len(set(out)) == 4
        for prompt in out:
            tags = [t.strip() for t in prompt.split(",")]
            assert "Dark" in tags
            assert any(t.endswith("Bass") for t in tags)
            assert all(t.lower() in LOOP_VOCAB for t in tags)
            assert len(tags) == len({t.lower() for t in tags})  # no duplicates


def test_explore_loop_tries_a_different_instrument_each_time():
    for seed in range(10):
        for sketch in ("", "warm, dreamy", "piano"):
            firsts = [p.split(",")[0] for p in ideas(sketch, mode="loop", seed=seed)]
            assert len(set(firsts)) == 4


def test_explore_loop_tries_different_effects():
    for seed in range(10):
        effects = [p.split(", ")[-2] for p in ideas("dark bass", mode="loop", seed=seed)]
        assert len(set(effects)) == 4


def test_explore_loop_never_adds_a_contradiction():
    for seed in range(30):
        for prompt in ideas("dark pad", mode="loop", seed=seed):
            tags = {t.strip() for t in prompt.split(",")}
            assert not tags & {"Bright", "Sparkly", "Shiny", "Glassy"}


def test_explore_doesnt_repeat_what_the_sketch_says():
    for seed in range(20):
        for prompt in ideas("glass shatter with long reverb", seed=seed):
            assert prompt.count("reverb") == 1


def test_explore_loop_keeps_words_it_doesnt_know():
    assert all("lofi" in p for p in ideas("lofi piano", mode="loop"))


def test_explore_only_where_a_prompt_describes_the_sound():
    assert set(EXPLORE_MODES) == {"sfx", "free", "loop"}
    with pytest.raises(ValueError):
        ideas("piano", mode="instrument")
    with pytest.raises(ValueError):
        ideas("x" * 501)


def test_nudge_replaces_what_it_contradicts():
    out = nudge_prompt("bright metallic door slam, long reverb tail", "darker")
    assert out.startswith("metallic door slam, long reverb tail")
    assert "bright" not in out and "dark" in out


def test_nudge_drier_drops_the_space():
    out = nudge_prompt("heavy metal door slam in a concrete hallway, long reverb tail", "drier")
    assert "reverb" not in out
    assert out.startswith("heavy metal door slam in a concrete hallway") and "dry" in out


def test_nudge_tags_for_loops():
    out = nudge_prompt("Synth Lead, Bright, Medium Reverb, Melody", "darker", tags=True)
    assert out == "Synth Lead, Medium Reverb, Melody, Dark, Warm"
    assert nudge_prompt(out, "drier", tags=True) == "Synth Lead, Melody, Dark, Warm, Dry"


def test_nudge_twice_doesnt_repeat_itself():
    once = nudge_prompt("rain on a roof", "bigger")
    assert nudge_prompt(once, "bigger") == once


def test_every_nudge_has_an_opposite_free_result():
    for name in NUDGES:
        assert nudge_prompt("a sound", name) != "a sound"
        assert nudge_prompt("Pad", name, tags=True) != "Pad"


def test_unknown_nudge():
    with pytest.raises(ValueError):
        nudge_prompt("door", "louder-ish")
