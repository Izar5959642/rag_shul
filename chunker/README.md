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
        { "seif": 1, "text": "...", "hagah": "...", "text_raw": "..." }
      ]
    }
  ]
}
```

- `hilchot_group`, `siman_sign` — siman-level fields, configurable via `siman_fields`
- `text`, `hagah`, `text_raw` — seif-level fields, configurable via `chunk_fields`
- `hagah` may be `null`; null fields are silently skipped

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
    - type_text: text+hagah          # label for this table
      chunk_fields: [text, hagah]
      siman_fields: []
      # mode: seif                   # optional — overrides top-level mode for this variant
    - type_text: text_only
      chunk_fields: [text]
      siman_fields: []
    - type_text: text+hilchot_group
      chunk_fields: [text]
      siman_fields: [hilchot_group]
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

All paths are read from `config/config.yaml` (`paths.chunks_json` for output; input defaults to `data/processed/shulchan_aruch_rag.json` unless overridden by a `schema_json` key under the active `run_mode` paths).

---

## Use as API

**Recommended — read everything from `config.yaml` (no hardcoded paths or overrides):**

```python
from chunker import build_tables, load_schema
from chunker.chunker import load_config
from pathlib import Path

cfg         = load_config()
schema_path = Path(__file__).parent.parent / cfg["paths"]["schema_json"]
schema      = load_schema(schema_path)

# builds all variants defined in config.yaml → list of {metadata, data} tables
tables = build_tables(schema)
```

Or run the full pipeline (load + build + save) in one call:

```python
import chunker.main as m
m.main()   # reads input path, output path, and variants from config.yaml
```

**For experimentation only — override variants without touching config:**

```python
from chunker import build_tables, load_schema

schema = load_schema("data/processed/shulchan_aruch_rag_with_breadcrumb.json")
tables = build_tables(schema, variants=[
    {"type_text": "text+hagah",  "chunk_fields": ["text", "hagah"], "siman_fields": []},
    {"type_text": "with_breadcrumb", "chunk_fields": ["text"], "siman_fields": ["hilchot_group"]},
])
```
