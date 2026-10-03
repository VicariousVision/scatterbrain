# Missing embeddings investigation

## Summary answer

The embeddings are present. Direct read-only inspection of `backend/scatterbrain.db` shows **1,381 rows in `vec_chunks`**, including **1,270 vectors linked to the uploaded Currency and Exchanges Manual**. The vectors are stored by sqlite-vec as a virtual table, with the embedding column represented as **3,072-byte BLOBs** (`768 × 4` bytes, float32), not as human-readable values in an ordinary SQLite table browser.

There are two likely reasons they appeared to be missing:

1. A normal SQLite connection/viewer that has not loaded the sqlite-vec `vec0` extension cannot query the `vec_chunks` virtual table; it may only show sqlite-vec shadow tables or report `no such module: vec0`.
2. The UUID in the investigation request is `dfa8e4cc-f66b-423f-9f1b-f83dfde7b41b`, but the completed document row actually uses `dfa8e4cc-f66b-423c-9f1b-f83dfde7b41b` (`423c`, not `423f`). Querying the supplied UUID therefore returns zero chunks/vectors. This is an identifier mismatch, not evidence that ingestion failed.

No persistence bug was found in the inspected database. The separate `backend/scatterbrain_huggingface.db` is an older/different database: it contains 846 vectors using Hugging Face/`litillabs/litil-embed-0.6b` at 1024 dimensions and does not contain this completed manual ingestion.

## Evidence

### Configuration and database selection

- `.env` sets `SQLITE_DB_PATH=scatterbrain.db` and `EMBEDDING_PROVIDER=ollama`.
- `backend/config.py`, `Settings.sqlite_db_path` (line 45), defaults to the relative path `scatterbrain.db`.
- `backend/services/vector_store.py`, `VectorStore.__init__` (around lines 140–153), uses that configured path unless an explicit path is supplied.
- `backend/main.py`, `lifespan` (around lines 75–93), constructs `VectorStore` without an explicit path, so it uses `settings.sqlite_db_path`.
- Because the configured value is relative, starting Uvicorn from `backend` resolves it to `backend/scatterbrain.db`. Starting from the repository root would resolve the same relative name in the root and can create/use a different database. The root-level `scatterbrain.db` was not present during this inspection.

### Direct database results

All database reads below were performed with SQLite opened using `mode=ro`; no application server was restarted and no database was modified.

`backend/scatterbrain.db`:

- File exists; size was 17,596,416 bytes.
- `documents`: 1 row.
- `embedding_metadata`: `{provider: ollama, model: nomic-embed-text, dimension: 768}`.
- `document_chunks`: 1,381 rows.
- `document_parents`: 512 rows.
- `vec_chunks`: **1,381 rows** after loading the sqlite-vec extension.
- `vec_chunks_rowids`: 1,381 internal row mappings.
- `vec_chunks` embedding values: `typeof(embedding) = 'blob'`; count 1,381; minimum and maximum length both 3,072 bytes.

The document metadata row is:

```text
document_id:   dfa8e4cc-f66b-423c-9f1b-f83dfde7b41b
filename:      Currency and Exchanges Manual for Authorised Dealers.pdf
status:        completed
chunk_count:   1270
needs_reingestion: 0
```

The exact ownership query using the database's UUID returned:

```text
count(document_chunks) = 1270
count(vec_chunks joined by chunk_id) = 1270
```

The remaining 111 chunks/vectors belong to `MSGraphRAG.pdf`. Grouped `document_chunks` counts were:

```text
84840065-95a4-496a-b038-535b6dab2352  MSGraphRAG.pdf                                  111
 dfa8e4cc-f66b-423c-9f1b-f83dfde7b41b  Currency and Exchanges Manual for Authorised Dealers.pdf  1270
```

Using the UUID exactly as supplied in the investigation request (`...423f...`) returned zero `documents`, zero `document_chunks`, and zero vectors because no such UUID exists in this database.

`backend/scatterbrain_huggingface.db`:

- File exists; size was 6,791,168 bytes.
- `embedding_metadata`: `{provider: huggingface, model: litillabs/litil-embed-0.6b, dimension: 1024}`.
- `vec_chunks`: 846 vectors, each 4,096 bytes (`1024 × 4`).
- Its manual records are failed uploads with zero chunks; its completed 846-chunk document is `From vectors to knowledge graphs.pdf`.
- The target completed manual UUID is not present in this database.

### Why ordinary browsing looks empty

`backend/services/vector_store.py`, `VectorStore.initialize` (around lines 231–237), creates:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[768]
)
```

`backend/services/vector_store.py`, `VectorStore.add_document` (around lines 399–403), packs each embedding with `struct.pack("768f", ...)` and inserts it into `vec_chunks( chunk_id, embedding )`.

The direct `sqlite_master` schema also exposes sqlite-vec implementation tables:

```text
vec_chunks                  virtual table, vec0, FLOAT[768]
vec_chunks_chunks           shadow table
vec_chunks_info             shadow table
vec_chunks_rowids           shadow table
vec_chunks_vector_chunks00  shadow table containing packed vector pages
```

Without loading sqlite-vec, querying `vec_chunks` fails with `no such module: vec0`. This explains why a viewer may show a virtual table but no browsable rows/columns, or may appear to show no ordinary embedding table. With the extension loaded, `SELECT count(*) FROM vec_chunks` returns 1,381 and `SELECT typeof(embedding), length(embedding)` shows the BLOB representation above. The shadow vector-page table contains two 3,145,728-byte pages; those are internal packed storage and are not the recommended inspection interface.

### Relevant retrieval code

`backend/services/vector_store.py`, `search_candidates` (around lines 584–607), joins `document_chunks` to `vec_chunks` by `chunk_id` and computes `vec_distance_cosine(vc.embedding, ?)`. This is the same path the application uses to retrieve the stored vectors, so the database layout is intentional rather than an accidental unreferenced blob table.

`backend/services/document_db.py`, `DocumentDB` schema and `update_status` methods (around lines 53–109 and 133–175), persist the completed status and `chunk_count` separately from vector rows. The manual's `completed`, `chunk_count=1270`, and `needs_reingestion=0` metadata is consistent with successful ingestion.

`docs/legal-document-chunking.md`, “Stable metadata and SQLite migration” and “Operation and verification”, documents that only searchable child records are embedded and that legal/manual ingestion stores parents separately. Therefore the 1,270 count is the expected searchable-child/vector count, not a requirement that all 512 parent records also have vectors.

## Exact read-only commands for a SQLite viewer or SQLite CLI

Use the actual application database:

```text
C:\Users\ozzey\Documents\git-projects\scatterbrain\backend\scatterbrain.db
```

First load sqlite-vec in a Python session using the backend virtual environment (the extension is a native DLL, so a generic viewer must also be configured to load the matching sqlite-vec extension):

```powershell
cd C:\Users\ozzey\Documents\git-projects\scatterbrain
@'
import sqlite3, sqlite_vec
p = r'C:\Users\ozzey\Documents\git-projects\scatterbrain\backend\scatterbrain.db'
c = sqlite3.connect('file:' + p.replace('\\\\','/') + '?mode=ro', uri=True)
c.enable_load_extension(True)
c.load_extension(sqlite_vec.loadable_path())
c.enable_load_extension(False)
print(c.execute("SELECT count(*) FROM vec_chunks").fetchone()[0])
print(c.execute("SELECT provider, model, dimension FROM embedding_metadata").fetchall())
print(c.execute("SELECT typeof(embedding), count(*), min(length(embedding)), max(length(embedding)) FROM vec_chunks").fetchall())
print(c.execute("SELECT count(*) FROM vec_chunks WHERE chunk_id IN (SELECT chunk_id FROM document_chunks WHERE document_id = 'dfa8e4cc-f66b-423c-9f1b-f83dfde7b41b')").fetchone()[0])
c.close()
'@ | .\backend\.venv\Scripts\python.exe -
```

Expected output is equivalent to:

```text
1381
[('ollama', 'nomic-embed-text', 768)]
[('blob', 1381, 3072, 3072)]
1270
```

Once the extension is loaded, these SQL statements can be pasted into a compatible SQLite viewer:

```sql
SELECT count(*) AS total_vectors FROM vec_chunks;

SELECT provider, model, dimension
FROM embedding_metadata;

SELECT typeof(embedding) AS value_type,
       count(*) AS vectors,
       min(length(embedding)) AS min_bytes,
       max(length(embedding)) AS max_bytes
FROM vec_chunks;

SELECT d.document_id, d.filename, d.status, d.chunk_count,
       count(c.chunk_id) AS chunk_rows,
       count(v.chunk_id) AS vector_rows
FROM documents d
LEFT JOIN document_chunks c ON c.document_id = d.document_id
LEFT JOIN vec_chunks v ON v.chunk_id = c.chunk_id
GROUP BY d.document_id, d.filename, d.status, d.chunk_count;

SELECT count(*) AS manual_vectors
FROM vec_chunks
WHERE chunk_id IN (
    SELECT chunk_id
    FROM document_chunks
    WHERE document_id = 'dfa8e4cc-f66b-423c-9f1b-f83dfde7b41b'
);
```

If the viewer cannot load sqlite-vec, it can still confirm the metadata/chunk side with:

```sql
SELECT document_id, filename, status, chunk_count, needs_reingestion
FROM documents;

SELECT document_id, filename, count(*) AS chunk_rows
FROM document_chunks
GROUP BY document_id, filename;

SELECT name, type, sql
FROM sqlite_master
WHERE name IN ('vec_chunks', 'vec_chunks_rowids', 'vec_chunks_vector_chunks00');
```

Those queries do not expose decoded vector values, but they establish that the virtual table and its internal storage exist. A viewer must load the same sqlite-vec extension to query `vec_chunks` itself.

## Conclusions and recommendations

- **Conclusion:** the manual was successfully ingested into `backend/scatterbrain.db`; its 1,270 searchable children have 1,270 persisted sqlite-vec embeddings. The user is likely looking at the wrong database, using a viewer without sqlite-vec loaded, or filtering with the mistyped `423f` UUID.
- Use the absolute backend database path above and the database's exact UUID containing `423c`.
- Configure the SQLite viewer's loadable extension to the backend environment's sqlite-vec DLL, or use the read-only Python command above.
- Avoid opening `backend/scatterbrain_huggingface.db` when checking this upload; it is a different provider/model/dimension database.
- To prevent future ambiguity, set `SQLITE_DB_PATH` to an explicit absolute path (or ensure Uvicorn is always started from `backend`) and have the UI/logging display the resolved database path and the stored document ID.
- The code's relative-path behavior is a maintainability risk, but changing it is outside this read-only investigation. No code or database changes were made.
