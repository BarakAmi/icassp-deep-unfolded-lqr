import numpy as np
import torch

from mbl.core.utils.signing import (
    Signable,
    compute_run_signature,
    compute_signature_digest,
    flatten_signature,
    hash_array,
)


class _Stub:
    """Minimal `Signable`: signs whatever dict it was built with."""

    def __init__(self, signature: dict) -> None:
        self._signature = signature

    def get_signature(self) -> dict:
        return self._signature


def test_hash_array_is_invariant_to_transpose_view() -> None:
    base = np.arange(12.0).reshape(3, 4)
    transposed_twice = base.T.T  # same data, went through a non-contiguous transpose
    assert hash_array(base) == hash_array(transposed_twice)


def test_hash_array_is_invariant_to_slice_view() -> None:
    base = np.arange(20.0)
    sliced = base[::2]  # a non-contiguous strided view
    direct = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
    assert hash_array(sliced) == hash_array(direct)


def test_hash_array_differs_for_genuinely_different_content() -> None:
    assert hash_array(np.eye(3)) != hash_array(np.eye(3) * 2)


def test_hash_array_differs_for_different_shape() -> None:
    assert hash_array(np.zeros((2, 3))) != hash_array(np.zeros((3, 2)))


def test_hash_array_differs_for_different_dtype() -> None:
    assert hash_array(np.eye(2).astype(np.float32)) != hash_array(
        np.eye(2).astype(np.float64)
    )


def test_hash_array_handles_noncontiguous_torch_tensor() -> None:
    t = torch.arange(12.0).reshape(3, 4).T  # non-contiguous torch view
    assert hash_array(t) == hash_array(t.contiguous())


def test_hash_array_detaches_grad_requiring_tensor() -> None:
    t = torch.eye(2, requires_grad=True)
    assert hash_array(t) == hash_array(torch.eye(2))  # must not raise, must match


def test_hash_array_format_is_prefixed_with_algorithm() -> None:
    assert hash_array(np.eye(2)).startswith("sha256:")


def test_flatten_signature_never_produces_duplicate_keys_across_components() -> None:
    """Every component signs its own 'type'/'dim'-shaped fields independently;
    this proves the flattened result never collapses two of them together."""
    tree = {
        "problem": {
            "system": {"type": "LinearSystem", "state_dim": 4},
            "cost": {"type": "QuadraticCost", "state_dim": 4},
        },
        "sampling": {"process_noise": {"type": "Gaussian", "dim": 4}},
    }
    flat = flatten_signature(tree)
    assert flat["problem.system.type"] == "LinearSystem"
    assert flat["problem.cost.type"] == "QuadraticCost"
    assert flat["problem.system.state_dim"] == 4
    assert flat["problem.cost.state_dim"] == 4
    assert flat["sampling.process_noise.type"] == "Gaussian"
    assert len(flat) == 6  # every leaf survives; nothing silently overwritten


def test_flatten_signature_handles_lists_with_index_segments() -> None:
    tree = {
        "constraints": [{"type": "Box", "u_max": 1.0}, {"type": "Box", "u_max": 2.0}]
    }
    flat = flatten_signature(tree)
    assert flat == {
        "constraints.0.type": "Box",
        "constraints.0.u_max": 1.0,
        "constraints.1.type": "Box",
        "constraints.1.u_max": 2.0,
    }


def test_flatten_signature_handles_empty_list() -> None:
    assert flatten_signature({"constraints": []}) == {"constraints": []}


def test_flatten_signature_passes_through_scalar_leaves_unchanged() -> None:
    tree = {"a": 1, "b": "text", "c": None, "d": True}
    assert flatten_signature(tree) == tree


def test_compute_signature_digest_is_order_independent() -> None:
    a = {"x": 1, "y": {"z": 2}}
    b = {"y": {"z": 2}, "x": 1}
    assert compute_signature_digest(a) == compute_signature_digest(b)


def test_compute_signature_digest_differs_for_different_content() -> None:
    assert compute_signature_digest({"x": 1}) != compute_signature_digest({"x": 2})


def test_compute_signature_digest_is_stable_across_repeated_calls() -> None:
    tree = {"a": {"b": [1, 2, 3]}}
    assert compute_signature_digest(tree) == compute_signature_digest(tree)


def test_compute_run_signature_matches_manual_tree_assembly() -> None:
    """`compute_run_signature` must build the exact same {"problem":,
    "controller":, "sampling":} tree `ProblemSignatureCallback` persists --
    regression guard against the two code paths drifting apart."""
    problem = _Stub({"type": "OptimalControlProblem"})
    controller = _Stub({"type": "RiccatiController", "horizon": 10})
    distributions = {"initial_state": _Stub({"type": "Gaussian", "seed": 1})}

    expected = compute_signature_digest(
        {
            "problem": problem.get_signature(),
            "controller": controller.get_signature(),
            "sampling": {
                "initial_state": distributions["initial_state"].get_signature()
            },
        }
    )
    assert compute_run_signature(problem, controller, distributions) == expected


def test_compute_run_signature_defaults_to_empty_sampling() -> None:
    problem = _Stub({"type": "OptimalControlProblem"})
    controller = _Stub({"type": "RiccatiController"})

    expected = compute_signature_digest(
        {
            "problem": problem.get_signature(),
            "controller": controller.get_signature(),
            "sampling": {},
        }
    )
    assert compute_run_signature(problem, controller) == expected


def test_compute_run_signature_differs_when_any_component_differs() -> None:
    problem = _Stub({"type": "OptimalControlProblem"})
    controller = _Stub({"type": "RiccatiController"})
    other_controller = _Stub({"type": "RiccatiController", "horizon": 99})

    assert compute_run_signature(problem, controller) != compute_run_signature(
        problem, other_controller
    )


def test_signable_protocol_is_runtime_checkable_and_structural() -> None:
    class _HasSignature:
        def get_signature(self) -> dict:
            return {"type": "Dummy"}

    class _NoSignature:
        pass

    assert isinstance(_HasSignature(), Signable)
    assert not isinstance(_NoSignature(), Signable)
