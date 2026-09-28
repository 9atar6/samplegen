import pytest

from samplegen.timing import GENERATION_PAD_SECONDS, MIN_LATENT_SECONDS, loop_seconds, plan_generation


def test_loop_seconds_four_four():
    assert loop_seconds(bpm=120, bars=4) == pytest.approx(8.0)
    assert loop_seconds(bpm=140, bars=8) == pytest.approx(13.7142857)


def test_loop_seconds_rejects_bad_input():
    with pytest.raises(ValueError):
        loop_seconds(bpm=0, bars=4)
    with pytest.raises(ValueError):
        loop_seconds(bpm=120, bars=0)


def test_plan_generation_pads_latent_and_counts_exact_samples():
    plan = plan_generation(13.7142857142857, sample_rate=44100)
    assert plan.target_samples == 604800
    assert plan.latent_seconds == pytest.approx(13.7142857 + GENERATION_PAD_SECONDS)
    assert plan.conditioning_seconds == 14


def test_plan_generation_whole_seconds_condition_exactly():
    plan = plan_generation(8.0, sample_rate=44100)
    assert plan.target_samples == 352800
    assert plan.conditioning_seconds == 8


def test_plan_generation_enforces_engine_minimum_length():
    plan = plan_generation(0.3, sample_rate=44100)
    assert plan.latent_seconds == MIN_LATENT_SECONDS
    assert plan.target_samples == 13230
    assert plan.conditioning_seconds == 1


def test_plan_generation_rejects_non_positive():
    with pytest.raises(ValueError):
        plan_generation(0, sample_rate=44100)
