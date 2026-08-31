"""Regression coverage for the promoted `models.iterative.initializers`
module (Phase 2.5): the moved control-initializer behavior is exercised
indirectly by every unfolded/open_loop test that constructs one, so this
file focuses on what's new here -- `get_signature()`, added so a solve's
provenance record can capture which initialization strategy produced a
given run."""

import torch

from mbl.models.iterative.initializers import (
    ConstantInitializer,
    ControlInitMethod,
    SamplerInitializer,
    WarmStartInitializer,
    build_control_initializer,
)
from mbl.models.samplers import RandomSampler, gaussian_sampler

DTYPE = torch.float64
CPU = torch.device("cpu")


def test_constant_initializer_always_returns_the_same_value() -> None:
    init = ConstantInitializer(
        constant=1.5, control_dim=2, dtype=DTYPE, device=torch.device("cpu")
    )
    y = torch.zeros(3, 4, dtype=DTYPE)

    u = init(0, y, None)

    assert u.shape == (3, 2)
    assert torch.all(u == 1.5)


def test_constant_initializer_get_signature_reports_type_and_config() -> None:
    init = ConstantInitializer(
        constant=0.0, control_dim=2, dtype=DTYPE, device=torch.device("cpu")
    )
    signature = init.get_signature()

    assert signature == {
        "type": "ConstantInitializer",
        "constant": 0.0,
        "control_dim": 2,
        "dtype": str(DTYPE),
    }


def test_warm_start_initializer_delegates_to_fallback_at_t_zero() -> None:
    fallback = ConstantInitializer(
        constant=2.0, control_dim=1, dtype=DTYPE, device=torch.device("cpu")
    )
    init = WarmStartInitializer(fallback)
    y = torch.zeros(2, 3, dtype=DTYPE)

    u0 = init(0, y, None)
    assert torch.all(u0 == 2.0)

    prev_u = torch.full((2, 1), 9.0, dtype=DTYPE)
    u1 = init(1, y, prev_u)
    assert torch.equal(u1, prev_u)


def test_warm_start_initializer_get_signature_nests_fallback_signature() -> None:
    fallback = ConstantInitializer(
        constant=0.0, control_dim=1, dtype=DTYPE, device=torch.device("cpu")
    )
    init = WarmStartInitializer(fallback)

    signature = init.get_signature()

    assert signature["type"] == "WarmStartInitializer"
    assert signature["fallback"] == fallback.get_signature()


def test_sampler_initializer_draws_from_the_wrapped_sampler() -> None:
    sampler = RandomSampler("u0", gaussian_sampler)
    sampler.unify_sampler(device=torch.device("cpu"), dtype=DTYPE)
    init = SamplerInitializer(
        sampler, control_dim=3, dtype=DTYPE, device=torch.device("cpu")
    )
    y = torch.zeros(4, 5, dtype=DTYPE)

    u = init(0, y, None)

    assert u.shape == (4, 3)
    assert u.dtype == DTYPE


def test_sampler_initializer_get_signature_nests_sampler_signature() -> None:
    sampler = RandomSampler("u0", gaussian_sampler)
    init = SamplerInitializer(
        sampler, control_dim=3, dtype=DTYPE, device=torch.device("cpu")
    )

    signature = init.get_signature()

    assert signature["type"] == "SamplerInitializer"
    assert signature["sampler"] == sampler.get_signature()
    assert signature["control_dim"] == 3


# --- The ControlInitMethod vocabulary + build_control_initializer factory ----


def test_build_cold_initializer_produces_zeros() -> None:
    init = build_control_initializer(
        ControlInitMethod.COLD, control_dim=2, dtype=DTYPE, device=CPU
    )
    u = init(0, torch.zeros(3, 4, dtype=DTYPE), None)

    assert isinstance(init, ConstantInitializer)
    assert u.shape == (3, 2)
    assert torch.all(u == 0.0)


def test_build_warm_initializer_reuses_the_previous_control() -> None:
    init = build_control_initializer(
        ControlInitMethod.WARM, control_dim=2, dtype=DTYPE, device=CPU
    )
    y = torch.zeros(3, 4, dtype=DTYPE)
    prev = torch.arange(6, dtype=DTYPE).reshape(3, 2)

    assert isinstance(init, WarmStartInitializer)
    # t == 0 (no previous control) falls back to the cold zero start ...
    assert torch.all(init(0, y, None) == 0.0)
    # ... t > 0 reuses the previous step's refined control verbatim.
    assert torch.equal(init(1, y, prev), prev)


def test_build_randomized_initializer_is_seeded_and_reproducible() -> None:
    y = torch.zeros(5, 4, dtype=DTYPE)

    def draw(seed: int) -> torch.Tensor:
        init = build_control_initializer(
            ControlInitMethod.RANDOMIZED,
            control_dim=2,
            dtype=DTYPE,
            device=CPU,
            random_std=1.0,
            seed=seed,
        )
        assert isinstance(init, SamplerInitializer)
        return init(0, y, None)

    first = draw(seed=7)
    assert first.shape == (5, 2)
    assert first.dtype == DTYPE
    assert not torch.all(first == 0.0)  # genuinely randomized, not a cold start
    assert torch.equal(first, draw(seed=7))  # same seed -> identical draw
    assert not torch.equal(first, draw(seed=8))  # different seed -> different draw
