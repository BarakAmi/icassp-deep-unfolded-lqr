import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from mbl.core.constraint.box_constraint import BoxConstraint
from mbl.viz.plots.convergence import (
    CocpOverlay,
    plot_control_convergence_contour_2d,
    plot_high_dimensional_convergence,
)
from mbl.workbench.analysis import (
    OneStepCostModel,
    compute_gradient_descent_path,
    compute_stage_cost_grid,
)


def test_compute_stage_cost_grid_matches_direct_evaluation() -> None:
    A = np.array([[0.9, 0.0], [0.0, 0.9]])
    B = np.eye(2)
    R = np.eye(2)
    P_next = np.diag([2.0, 3.0])
    x = np.array([1.0, -0.5])
    u1_range = np.linspace(-1, 1, 9)
    u2_range = np.linspace(-2, 2, 11)

    U1, U2, Z = compute_stage_cost_grid(
        u1_range, u2_range, x, model=OneStepCostModel(A=A, B=B, R=R, P_next=P_next)
    )

    assert Z.shape == (11, 9)
    for i in (0, 5, 10):
        for j in (0, 4, 8):
            u = np.array([U1[i, j], U2[i, j]])
            expected = u @ R @ u + (A @ x + B @ u) @ P_next @ (A @ x + B @ u)
            assert Z[i, j] == pytest.approx(expected)


def test_compute_gradient_descent_path_matches_hand_derived_reference() -> None:
    """Cross-check against the same independent reference used to validate
    RiccatiRefinement's own math (tests/models/unfolded/test_iterative_refinement.py),
    proving this dashboard-facing replay function computes identical iterates
    to the real controller."""
    A = np.array([[0.9, 0.0], [0.0, 0.9]])
    B = np.array([[1.0], [0.0]])
    R = np.array([[1.0]])
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    x = np.array([1.0, 0.0])
    num_iterations = 3
    alpha = 0.1
    step_sizes = np.full(num_iterations, alpha)

    path = compute_gradient_descent_path(
        x, model=OneStepCostModel(A=A, B=B, R=R, P_next=P), step_sizes=step_sizes
    )

    assert path.shape == (num_iterations + 1, 1)
    BtP = B.T @ P
    M2 = 2 * (R + BtP @ B)
    C2 = 2 * (BtP @ A)
    x_Ct = x @ C2.T
    u_ref = np.zeros(1)
    for _ in range(num_iterations):
        grad = u_ref @ M2 + x_Ct
        u_ref = u_ref - alpha * grad
    assert np.allclose(path[-1], u_ref)
    assert np.isclose(path[-1, 0], -0.5616, atol=1e-4)


def test_compute_gradient_descent_path_respects_initial_guess() -> None:
    A, B, R, P = np.eye(2), np.eye(2), np.eye(2), np.eye(2)
    x = np.zeros(2)
    u0 = np.array([3.0, -3.0])
    path = compute_gradient_descent_path(
        x,
        model=OneStepCostModel(A=A, B=B, R=R, P_next=P),
        step_sizes=np.zeros(4),
        u0=u0,
    )
    # Zero step sizes -> no movement at all, every iterate equals u0.
    assert np.allclose(path, np.tile(u0, (5, 1)))


def test_compute_gradient_descent_path_constraint_none_is_unaffected() -> None:
    """NB04 plan Sec 3.4/7: `constraint=None` (the default, every existing
    unconstrained call site) must replay byte-for-byte identically to
    before this parameter existed."""
    A = np.array([[0.9, 0.0], [0.0, 0.9]])
    B = np.array([[1.0], [0.0]])
    R = np.array([[1.0]])
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    x = np.array([1.0, 0.0])
    step_sizes = np.full(3, 0.1)
    model = OneStepCostModel(A=A, B=B, R=R, P_next=P)

    without_kwarg = compute_gradient_descent_path(x, model=model, step_sizes=step_sizes)
    with_none = compute_gradient_descent_path(
        x, model=model, step_sizes=step_sizes, constraint=None
    )
    assert np.array_equal(without_kwarg, with_none)


def test_compute_gradient_descent_path_projects_every_intermediate_iterate() -> None:
    """The replayed path must match what the REAL controller's inner loop
    does: project after EVERY step, not just clip the final iterate. A
    large step size that would overshoot far past the box on the first
    step must show the SECOND iterate already saturated at the boundary,
    not at some larger unconstrained value."""
    A = np.eye(2)
    B = np.eye(2)
    R = np.eye(2)
    P = np.eye(2)
    x = np.array([5.0, -5.0])  # large state -> large unconstrained gradient
    u_max = 0.3
    step_sizes = np.full(4, 0.5)

    path = compute_gradient_descent_path(
        x,
        model=OneStepCostModel(A=A, B=B, R=R, P_next=P),
        step_sizes=step_sizes,
        constraint=BoxConstraint(u_max=u_max),
    )

    assert np.all(np.abs(path) <= u_max + 1e-12)
    # The unconstrained update from u=0 for this model is a large jump
    # (|grad| ~ |2*C*x| = |2*B^T*P*A*x| = 10 here) -- confirm it WOULD have
    # overshot without projection, so this test is not vacuously true.
    unconstrained_first_step = 0.0 - step_sizes[0] * (2 * B.T @ P @ A @ x)
    assert np.any(np.abs(unconstrained_first_step) > u_max)
    assert np.allclose(path[1], np.clip(unconstrained_first_step, -u_max, u_max))


def _make_synthetic_contour():
    A, B, R, P = np.eye(2), np.eye(2), np.eye(2), np.diag([2.0, 1.0])
    x = np.array([1.0, 1.0])
    u1_range = np.linspace(-2, 2, 20)
    u2_range = np.linspace(-2, 2, 20)
    U1, U2, Z = compute_stage_cost_grid(
        u1_range, u2_range, x, model=OneStepCostModel(A=A, B=B, R=R, P_next=P)
    )
    path = compute_gradient_descent_path(
        x,
        model=OneStepCostModel(A=A, B=B, R=R, P_next=P),
        step_sizes=np.full(8, 0.05),
    )
    return U1, U2, Z, path


def test_plot_control_convergence_contour_2d_returns_figure_unfolded_only() -> None:
    U1, U2, Z, path = _make_synthetic_contour()

    fig = plot_control_convergence_contour_2d(U1, U2, Z, path)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_control_convergence_contour_2d_with_cocp_overlay() -> None:
    U1, U2, Z, path = _make_synthetic_contour()
    cocp_grid = Z * 1.1  # a distinguishable "different cost-to-go" landscape
    cocp_point = path[-1] + 0.2

    fig = plot_control_convergence_contour_2d(
        U1, U2, Z, path, cocp=CocpOverlay(cost_grid=cocp_grid, point=cocp_point)
    )

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "COCP solution (optimum)" in legend_labels


def test_plot_control_convergence_contour_2d_rejects_mismatched_grid_shapes() -> None:
    U1, U2, Z, path = _make_synthetic_contour()
    with pytest.raises(ValueError, match="same shape"):
        plot_control_convergence_contour_2d(U1, U2[:, :-1], Z, path)


def test_plot_control_convergence_contour_2d_accepts_torch_trajectory() -> None:
    U1, U2, Z, path = _make_synthetic_contour()
    fig = plot_control_convergence_contour_2d(U1, U2, Z, torch.as_tensor(path))
    assert isinstance(fig, Figure)


def test_plot_control_convergence_contour_2d_draws_box_constraint_when_u_max_given() -> (
    None
):
    from matplotlib.patches import Rectangle

    U1, U2, Z, path = _make_synthetic_contour()

    fig = plot_control_convergence_contour_2d(
        U1, U2, Z, path, cocp=CocpOverlay(u_max=1.5)
    )

    ax = fig.axes[0]
    rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == 1
    assert rectangles[0].get_width() == pytest.approx(3.0)
    assert rectangles[0].get_xy() == pytest.approx((-1.5, -1.5))


def test_plot_control_convergence_contour_2d_omits_box_constraint_by_default() -> None:
    from matplotlib.patches import Rectangle

    U1, U2, Z, path = _make_synthetic_contour()
    fig = plot_control_convergence_contour_2d(U1, U2, Z, path)
    ax = fig.axes[0]
    assert not [p for p in ax.patches if isinstance(p, Rectangle)]


def test_plot_control_convergence_contour_2d_cocp_marker_uses_the_optimum_color() -> (
    None
):
    from matplotlib.colors import to_rgba

    from mbl.viz.style.theme import CATEGORICAL_PALETTE

    U1, U2, Z, path = _make_synthetic_contour()
    fig = plot_control_convergence_contour_2d(
        U1, U2, Z, path, cocp=CocpOverlay(point=path[-1] + 0.1)
    )

    ax = fig.axes[0]
    rgb_colors = {
        tuple(np.asarray(color)[:3])
        for coll in ax.collections
        for color in coll.get_facecolor()
    }
    assert to_rgba(CATEGORICAL_PALETTE[5])[:3] in rgb_colors


def test_plot_high_dimensional_convergence_returns_two_panel_figure() -> None:
    A, B, R, P = np.eye(5), np.eye(5), np.eye(5), np.eye(5) * 2
    x = np.ones(5)
    path = compute_gradient_descent_path(
        x,
        model=OneStepCostModel(A=A, B=B, R=R, P_next=P),
        step_sizes=np.full(10, 0.02),
    )

    fig = plot_high_dimensional_convergence(path)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2


def test_plot_high_dimensional_convergence_uses_custom_dim_labels() -> None:
    path = np.random.default_rng(0).normal(size=(6, 3))
    fig = plot_high_dimensional_convergence(path, dim_labels=["thrust", "yaw", "pitch"])
    heatmap_ax = fig.axes[0]
    assert [t.get_text() for t in heatmap_ax.get_yticklabels()] == [
        "thrust",
        "yaw",
        "pitch",
    ]


def test_plot_high_dimensional_convergence_rejects_wrong_length_dim_labels() -> None:
    path = np.random.default_rng(0).normal(size=(6, 3))
    with pytest.raises(ValueError, match="dim_labels"):
        plot_high_dimensional_convergence(path, dim_labels=["only_one"])


def test_plot_high_dimensional_convergence_final_distance_is_zero() -> None:
    """The last iterate is by definition the reference point, so its distance
    to itself must be exactly zero -- a basic sanity check on the norm curve."""
    path = np.random.default_rng(1).normal(size=(7, 4))
    fig = plot_high_dimensional_convergence(path)
    # axes[0] is the heatmap, axes[1] is the norm panel (axes[2:], if any,
    # are the heatmap's colorbar, appended after both main axes).
    norm_ax = fig.axes[1]
    line = norm_ax.get_lines()[0]
    assert np.asarray(line.get_ydata())[-1] == pytest.approx(0.0, abs=1e-9)
