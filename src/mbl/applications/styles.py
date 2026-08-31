"""The applications' series-style registration (REFACTOR_PLAN v3, T4.c --
the M7 inversion): the model-family -> color/linestyle mapping that used to
be hardcoded inside the generic style layer now lives with the layer that
actually owns the family names, and is pushed into `viz.style`'s neutral
registry at import time. The generic fallback (palette cycling) stays
generic: an unregistered label never raises.
"""

from ..viz.style.semantic import register_series_styles
from ..viz.style.theme import CATEGORICAL_PALETTE

#: Canonical model-family name -> fixed color, covering every controller
#: this project's applications produce -- assigned by identity (never
#: re-cycled per plot) so the same controller reads as the same color across
#: every figure in the dashboard.
MODEL_FAMILY_COLORS = {
    "analytic": CATEGORICAL_PALETTE[0],
    "truncated_riccati": CATEGORICAL_PALETTE[0],
    "neural": CATEGORICAL_PALETTE[4],
    "unfolded_fixed": CATEGORICAL_PALETTE[2],
    "unfolded_learned_step_size": CATEGORICAL_PALETTE[1],
    "unfolded_learned_step_size_and_matrix": CATEGORICAL_PALETTE[5],
    "cocp": CATEGORICAL_PALETTE[7],
    "cocp_lower_bound": CATEGORICAL_PALETTE[3],
}

#: Iterative/gradient-trained families, rendered dashed so the closed-form/
#: iterative distinction survives grayscale printing too (color already
#: uniquely identifies each individual family above).
ITERATIVE_MODEL_FAMILIES = frozenset(
    {
        "neural",
        "unfolded_fixed",
        "unfolded_learned_step_size",
        "unfolded_learned_step_size_and_matrix",
    }
)

register_series_styles(MODEL_FAMILY_COLORS, dashed=ITERATIVE_MODEL_FAMILIES)

#: NB06's six contenders are labeled by INSTANCE name (e.g.
#: "unfolded_alpha"/"unfolded_alpha_p"), not by the RECIPE FAMILY name
#: `MODEL_FAMILY_COLORS` above is keyed by -- so, unregistered, they fell
#: through `viz.style.semantic.series_color`'s positional-palette-cycling
#: fallback (NB06_OOD_GENERALIZATION_PLAN.md Sec 5.0/11 finding): NB06's
#: flagship, "unfolded_alpha_p", landed on `CATEGORICAL_PALETTE[2]` --
#: `unfolded_fixed`'s (the UNTRAINED baseline's) own color everywhere else
#: -- purely because of where it happened to sit in a bands mapping's
#: iteration order, so removing/reordering any contender silently
#: reassigned every other one's color too. Registering these instance
#: labels explicitly, by reference into the SAME family colors above (never
#: a fresh palette index), fixes both: an exact-label match short-circuits
#: the positional fallback entirely, and the color is identical to the
#: recipe family's own color everywhere else in the dashboard (NB04
#: included), as the plan always intended.
NB06_CONTENDER_COLORS = {
    "truncated_riccati": MODEL_FAMILY_COLORS["truncated_riccati"],
    "unfolded_alpha": MODEL_FAMILY_COLORS["unfolded_learned_step_size"],
    "unfolded_alpha_p": MODEL_FAMILY_COLORS["unfolded_learned_step_size_and_matrix"],
    "cocp": MODEL_FAMILY_COLORS["cocp"],
    "neural": MODEL_FAMILY_COLORS["neural"],
    "cocp_lower_bound": MODEL_FAMILY_COLORS["cocp_lower_bound"],
}

#: Same dashed/solid law as `ITERATIVE_MODEL_FAMILIES`, restated for NB06's
#: own instance labels (`unfolded_alpha_p"` and `"unfolded_alpha"` are both
#: gradient-trained; `"truncated_riccati"`/`"cocp"`/`"cocp_lower_bound"` are
#: closed-form/convex-program, not gradient-trained in the same sense).
NB06_DASHED_CONTENDERS = frozenset({"unfolded_alpha", "unfolded_alpha_p", "neural"})

register_series_styles(NB06_CONTENDER_COLORS, dashed=NB06_DASHED_CONTENDERS)
