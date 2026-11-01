"""Pre-pass kernels: one component of a vector array, and the cell of each point link."""

import tack


@tack.kernel
def component(values, k, out, n):
    for i in range(n):
        out[i] = values[i][k]


@tack.kernel
def entry_cells(cells, entries, out, n):
    for i in range(n):
        out[i] = cells.entity_id(entries[i] // cells.NUM_POINTS)

