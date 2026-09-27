"""Visualize document chunk embeddings stored in scatterbrain.db.

Reads every embedding vector out of the `vec_chunks` table (paired with
its text/metadata from `document_chunks`), reduces the 768-dim vectors to
2D with PCA, and produces:

  1. embeddings_plot.png   - static scatter plot, colored by document
  2. embeddings_plot.html  - interactive Plotly scatter, hover shows the
                              source filename + a snippet of the chunk text

Usage (run from the backend/ directory, with the venv active):

    python visualize_embeddings.py [--db scatterbrain.db] [--method pca|tsne]

Requires: numpy, scikit-learn, matplotlib, plotly (dev-only, not in
requirements.txt since they're not needed to run the API).
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import sqlite_vec


def load_embeddings(db_path: str) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Load all embeddings + metadata from the sqlite-vec database.

    Returns:
        (vectors, filenames, document_ids, snippets)
    """
    conn = sqlite3.connect(db_path)

    # The vec_chunks table is a virtual table backed by the vec0 module,
    # which lives in the sqlite-vec extension. Load it the same way
    # services/vector_store.py does before querying that table.
    vec_path = str(Path(sqlite_vec.loadable_path())).replace("\\", "/")
    conn.enable_load_extension(True)
    conn.execute(f"SELECT load_extension('{vec_path}')")
    conn.enable_load_extension(False)

    cursor = conn.execute(
        """
        SELECT vc.embedding, dc.filename, dc.document_id, dc.content
        FROM vec_chunks vc
        JOIN document_chunks dc ON dc.chunk_id = vc.chunk_id
        ORDER BY dc.filename, dc.chunk_index
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"No embeddings found in {db_path}. Upload some documents first.")
        sys.exit(1)

    vectors = []
    filenames = []
    document_ids = []
    snippets = []
    for embedding_blob, filename, document_id, content in rows:
        n_floats = len(embedding_blob) // 4
        vectors.append(struct.unpack(f"{n_floats}f", embedding_blob))
        filenames.append(filename)
        document_ids.append(document_id)
        snippets.append((content[:120] + "...") if len(content) > 120 else content)

    return np.array(vectors, dtype=np.float32), filenames, document_ids, snippets


def reduce_to_2d(vectors: np.ndarray, method: str) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, len(vectors) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)

    return reducer.fit_transform(vectors)


def plot_static(coords: np.ndarray, filenames: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_files = sorted(set(filenames))
    cmap = plt.get_cmap("tab10" if len(unique_files) <= 10 else "tab20")

    for i, fname in enumerate(unique_files):
        mask = [f == fname for f in filenames]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=fname, color=cmap(i % cmap.N), s=40, alpha=0.75)

    ax.set_title("Document chunk embeddings (2D projection)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize="small", markerscale=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved static plot to {out_path}")


def plot_interactive(
    coords: np.ndarray,
    filenames: list[str],
    snippets: list[str],
    out_path: Path,
) -> None:
    import plotly.express as px

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=filenames,
        hover_name=filenames,
        hover_data={"snippet": snippets},
        title="Document chunk embeddings (2D projection)",
        labels={"x": "Component 1", "y": "Component 2", "color": "Document"},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.75))
    fig.write_html(out_path)
    print(f"Saved interactive plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sqlite-vec embeddings.")
    parser.add_argument("--db", default="scatterbrain.db", help="Path to the SQLite DB")
    parser.add_argument(
        "--method",
        choices=["pca", "tsne"],
        default="pca",
        help="Dimensionality reduction method (pca is fast, tsne separates clusters better)",
    )
    parser.add_argument("--out-dir", default=".", help="Directory to write output files")
    args = parser.parse_args()

    vectors, filenames, document_ids, snippets = load_embeddings(args.db)
    print(f"Loaded {len(vectors)} embeddings ({vectors.shape[1]} dims) from {args.db}")

    coords = reduce_to_2d(vectors, args.method)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_static(coords, filenames, out_dir / "embeddings_plot.png")
    plot_interactive(coords, filenames, snippets, out_dir / "embeddings_plot.html")


if __name__ == "__main__":
    main()
