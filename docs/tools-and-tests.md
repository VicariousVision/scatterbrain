# Tools and Tests

## `visualize_embeddings.py`

Dev-only script that plots stored chunk embeddings in 2D. Reads `vec_chunks` joined to `document_chunks` directly with `sqlite3` + the sqlite-vec extension (same loading approach as `VectorStore`), so it doesn't need the API running.

```
python visualize_embeddings.py [--db scatterbrain.db] [--method pca|tsne] [--out-dir .]
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--db` | `scatterbrain.db` | DB to read |
| `--method` | `pca` | `pca` is fast; `tsne` separates clusters better |
| `--out-dir` | `.` | Where outputs go |

Outputs `embeddings_plot.png` (matplotlib, coloured by file) and `embeddings_plot.html` (Plotly, hover shows a 120-char snippet). Needs `numpy`, `scikit-learn`, `matplotlib`, `plotly`, which aren't in `requirements.txt`.

Functions: `load_embeddings(db_path)`, `reduce_to_2d(vectors, method)`, `plot_static(coords, filenames, out_path)`, `plot_interactive(coords, filenames, snippets, out_path)`, `main()`.

## `tests/`

Run from `backend/` with `pytest`.

| File | Covers | External deps |
| --- | --- | --- |
| `test_text_chunker.py` | `chunk_text`: empty input, small/large text, word boundaries (patches chunk settings) | none |
| `test_ragas_adapters.py` | Telemetry opt-out, `parse_structured_output`, judge schema + `think=False`, judge retries, batch embeddings, golden dataset validity, threshold checks | none (mocks) |
| `test_ragas_evaluation.py` | Full pipeline must meet `RAGAS_MIN_*` thresholds | Ollama + pulled models; marked `ragas`, skipped unless `RUN_RAGAS_EVAL=1` |

Not currently covered by tests: routers, `DocumentService`, `DocumentDB`, `VectorStore`, `document_parser`, `text_cleaner`, `OllamaClient`, `embedding_provider`.
