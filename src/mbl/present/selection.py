"""Which rows of an analysis a figure actually draws (Annex 03 §B.1.2 clause 2).

One function, in its own module, because it has to run in two places and must
be one implementation in both.

§B.1.2 clause 2 requires that "every mark a reader can see in the figure has a
row in the table, and every row of the table is drawn in the figure. A figure
that draws a subset of the analysis selects that subset **before** the frame
is frozen, so that 'the exact values plotted' is true of the bytes and not
only of the intention."

It was not. The renderer applied `config["series"]` internally while the
runner froze the table it had been handed, so a figure drawing one contender
of two froze **six rows behind three drawn marks** — and `<id>.data.parquet`,
which §B.1 documents as "the exact values plotted", was not. The selection
therefore happens in the runner, before the frame is frozen; the renderer
still calls it so that a renderer invoked directly honours the same
declaration, and calling it twice is idempotent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..spec.errors import SpecificationError


def select_series(table: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """The rows `config["series"]` names, or all of them.

    Args:
        table: The analysis table.
        config: The figure's declaration.

    Returns:
        The selected rows, or `table` unchanged when nothing is selected.

    Raises:
        SpecificationError: If a named series is absent. Refused rather than
            skipped: a figure quietly missing the contender its caption is
            about is the plausible and wrong output this architecture exists
            to prevent.
    """
    selection = config.get("series")
    if selection is None:
        return table
    wanted = [str(name) for name in selection]
    available = set(table["contender"])
    unknown = sorted(set(wanted) - available)
    if unknown:
        raise SpecificationError(
            f"figure selects series {', '.join(unknown)}, which the analysis "
            f"table does not carry; available: {', '.join(sorted(available))}"
        )
    return pd.DataFrame(table[table["contender"].isin(wanted)])
