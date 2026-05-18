# Chunker

Reads the Shulchan Arukh RAG JSON and produces a `chunks.json` file for the embedder.

---

## Input

JSON file with this structure (`data/processed/shulchan_aruch_rag_with_breadcrumb.json`):
```json
{
  "title": "...",
  "source": "...",
  "simanim": [
    {
      "siman": 1,
      "hilchot_group": "...",
      "siman_sign": "...",
      "seifim": [
        {
          "seif": 1,
          "text": "...",
          "hagah": "...",
          "text_raw": "...",
          "modern_summary": "...",
          "questions": ["...", "...", "..."]
        }
      ]
    }
  ]
}
```

- `hilchot_group`, `siman_sign` — siman-level fields, configurable via `siman_fields`
- `text`, `hagah`, `text_raw`, `modern_summary`, `questions` — seif-level fields, configurable via `chunk_fields`
- list-valued fields (e.g. `questions`) are joined with spaces; `hagah` may be `null`; null/empty fields are silently skipped

---

## Output

Path is set by `paths.chunks_json` in `config/config.yaml`.

**Without `text_variants`** — a flat JSON array, one object per chunk:
```json
[
  { "id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..." },
  { "id": 1, "siman": 1, "seif": null, "siman_seif": "סימן 1", "text": "..." }
]
```

**With `text_variants`** — a JSON array of table objects, one per variant:
```json
[
  {
    "metadata": { "type_text": "text+hagah" },
    "data": [
      { "id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..." }
    ]
  }
]
```

`seif` is `null` for siman-level and sliding-window chunks.

---

## Options (config/config.yaml)

```yaml
chunker:
  mode: seif            # seif | siman | sliding_window
  chunk_size: 200       # words per chunk (sliding_window only)
  overlap: 50           # overlapping words between chunks (sliding_window only)
  chunk_fields:         # seif-level fields joined into the chunk text
    - text
    # - hagah           # uncomment to append Rema commentary
    # - siman_title     # uncomment to prepend the siman heading
  siman_fields:         # siman-level fields prepended to every chunk (all modes)
    # - hilchot_group   # uncomment to prepend the halachic category
    # - siman_sign      # uncomment to prepend the siman sign/marker
  text_variants:        # optional; if present, overrides single-mode output
    # 28 variants live in config/config.yaml (chunker.text_variants); these are
    # representative examples — see the config for the full list.
    - type_text: text+hagah          # label for this table
      chunk_fields: [text, hagah]
      siman_fields: []
      # mode: seif                   # optional — overrides top-level mode for this variant
    - type_text: text_only
      chunk_fields: [text]
      siman_fields: []
    - type_text: text+modern_summary
      chunk_fields: [text, modern_summary]
      siman_fields: []
    - type_text: questions_only
      chunk_fields: [questions]
      siman_fields: []
```

When `text_variants` is present, `build_tables` is called and the output contains one table per variant. Each variant can supply its own `chunk_fields`, `siman_fields`, and `mode`. If `chunk_fields` or `mode` is omitted from a variant, it falls back to the top-level config value. If `siman_fields` is omitted, it defaults to `[]` (no siman-level prefix).

| Mode | Description |
|---|---|
| `seif` | One chunk per seif (default) |
| `siman` | One chunk per siman (all seifim merged) |
| `sliding_window` | Fixed word-count windows across the full corpus |

---

## Run (CLI)

```bash
python3 -m chunker.main
```

Manual entry point. Output goes to `paths.chunks_json` from `config/config.yaml`. Input is taken from `paths.schema_json` if set; otherwise it falls back to `data/processed/shulchan_aruch_rag.json` (the non-breadcrumb file). Full-pipeline runs do **not** invoke this CLI — `experiments/exp_main.py` calls `chunker.chunker.run(data_file, chunks_json, variants)` directly with `paths.data_file_with_breadcrumb` as the input.

### Preview

Print sample chunks from the saved output:

```bash
python3 -m chunker.preview [--file PATH] [--n N]
```

Defaults: `--file` → `paths.chunks_json`, `--n` → 2 chunks per table.

---

## Use as API

**Recommended — load + build + save in one call (matches what `experiments/exp_main.py` does):**

```python
from chunker.chunker import run

# Reads variants from config.yaml automatically.
run(
    data_file="data/processed/shulchan_aruch_rag_with_breadcrumb.json",
    chunks_json="data/chunks.json",
)
```

Or use the existing CLI wrapper:

```python
import chunker.main as m
m.main()   # reads input path, output path, and variants from config.yaml
```

**Lower-level — build tables in memory without saving:**

```python
from chunker import build_tables, load_schema

schema = load_schema("data/processed/shulchan_aruch_rag_with_breadcrumb.json")

# Uses chunker.text_variants from config.yaml by default.
tables = build_tables(schema)
```

**For experimentation only — override variants without touching config:**

```python
from chunker import build_tables, build_dataframe, load_schema

schema = load_schema("data/processed/shulchan_aruch_rag_with_breadcrumb.json")
tables = build_tables(schema, variants=[
    {"type_text": "text+hagah",  "chunk_fields": ["text", "hagah"], "siman_fields": []},
    {"type_text": "with_breadcrumb", "chunk_fields": ["text"], "siman_fields": ["hilchot_group"]},
])

# Or get a single flat DataFrame instead of the multi-table structure:
df = build_dataframe(schema, chunk_fields=["text", "hagah"], siman_fields=[], mode="seif")
```
