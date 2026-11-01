"""Pre-pass kernels: one component of a vector array, and the cell of each point link."""

import tack


@tack.kernel
def component(values, k, out, n):
    for i in range(n):
        out[i] = values[i][k]


@tack.kernel
def entry_cells(cells, entries, out, n, start, end):
    """The cell of each link entry in [start, end): one cell group's corners in the
    topology's corner layout, NUM_POINTS per cell."""
    for i in range(n):
        e = entries[i]
        if e >= start and e < end:
            out[i] = cells.entity_id((e - start) // cells.NUM_POINTS)

