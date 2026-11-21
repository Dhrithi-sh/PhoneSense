"""Edit-cost heatmap: a picture of which spelling slips are cheap.

The learned edit distance (``editdist``) stores a cost per character edit. This
renders those substitution costs as a character × character grid — cheap edits
(the common Hinglish confusions: `au↔o`, `v↔w`, `s↔sh`) dark, expensive ones
light. A figure for the report and video.

``cost_matrix`` is pure data (no plotting), so it is unit-tested directly;
``render`` imports matplotlib lazily and writes a PNG.

    python -m dhvani.query.heatmap dhvani/query/edit_costs.json documentation/figures/edit_cost_heatmap.png
"""

import argparse

from dhvani.query.editdist import load_costs

# The Roman letters our romanisation and matchers actually produce.
DEFAULT_ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def cost_matrix(table, default, alphabet=None):
    """Return ``(alphabet, matrix)`` of substitution costs.

    ``matrix[i][j]`` is the cost of turning ``alphabet[i]`` into ``alphabet[j]``
    (0 on the diagonal, the learned cost where known, else ``default``).
    """
    alphabet = alphabet or DEFAULT_ALPHABET
    matrix = []
    for a in alphabet:
        row = []
        for b in alphabet:
            if a == b:
                row.append(0.0)
            else:
                row.append(table.get((a, b), default))
        matrix.append(row)
    return alphabet, matrix


def render(table, default, out_path, alphabet=None):
    """Render the substitution-cost heatmap to ``out_path`` (PNG)."""
    import matplotlib
    matplotlib.use("Agg")  # headless; no display needed
    import matplotlib.pyplot as plt

    alphabet, matrix = cost_matrix(table, default, alphabet)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="equal")
    ax.set_xticks(range(len(alphabet)), labels=alphabet)
    ax.set_yticks(range(len(alphabet)), labels=alphabet)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    ax.set_title("Learned edit-cost heatmap (darker = cheaper = more common slip)")
    fig.colorbar(im, ax=ax, label="−log P(edit)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the edit-cost heatmap.")
    ap.add_argument("costs", help="dhvani/query/edit_costs.json")
    ap.add_argument("out", help="output PNG path")
    args = ap.parse_args(argv)
    table, default = load_costs(args.costs)
    path = render(table, default, args.out)
    print(f"wrote heatmap -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
