# Architecture

```mermaid
flowchart LR
  subgraph Sources
    AN[Arbeitnow public API]
    WDC[Web Data Commons host list] --> CR[Polite crawler: robots.txt, TDMRep, sitemaps, JSON-LD]
    ESCOAPI[ESCO web service API]
    GN[GeoNames DE]
  end
  AN --> ST[(stores/postings.sqlite<br/>raw postings + snapshot presence)]
  CR --> ST
  ST --> BC[build-corpus: clean, redact, language, sections, location, MinHash dedupe, aggregator exclusion]
  GN --> BC
  BC --> PQ[(data/processed/postings.parquet)]
  ESCOAPI --> ES[(data/interim/esco_skills.parquet)]
  PQ --> MINE[mining of unlisted terms]
  ES --> GAZ[gazetteer: surface and lemma n-grams]
  MINE --> GAZ
  GOLD[(annotation/ gold set, 100 dev + 100 test)] --> TUNE[tuning on dev: mining thresholds, blocked keys]
  GAZ --> TUNE --> CFG[(stores/skills_config.json)]
  TUNE --> NER[spaCy NER trained on weak labels]
  CFG --> EXT[SkillExtractor + necessity rules]
  PQ --> CLS[classifier: TF-IDF LR, e5 LR, German BERT; grouped splits]
  PQ --> MKT[market: DuckDB shares + Wilson CIs, PMI, NMF profiles]
  EXT --> MKT
  EXT --> GAP[CV gap: match, related via e5, score]
  GAP --> LLM[local LLM via Ollama + guards]
  CLS --> API[FastAPI /v1/*]
  EXT --> API
  MKT --> API
  GAP --> API
  API --> UI[Streamlit: Market Trends, CV Analyser]
```

## Components (as implemented)

| module | responsibility |
|---|---|
| `harvest/client.py` | polite HTTP client: per-host spacing, backoff on 429 and 5xx, SQLite response cache |
| `harvest/arbeitnow.py` | snapshot of the Arbeitnow API (all pages, cached per snapshot date) |
| `harvest/crawl.py`, `policy.py`, `jsonld.py` | career-site crawler: robots.txt, TDMRep, job-board exclusion, sitemap discovery, JSON-LD JobPosting extraction, 16 hosts in parallel |
| `harvest/store.py` | raw posting store with first/last seen and per-snapshot presence |
| `text/` | HTML cleaning, redaction, language detection (lingua), section splitting, GeoNames location matching, MinHash deduplication, corpus build |
| `skills/` | ESCO fetch, gazetteer, mining, necessity rules, weak-label NER, gold evaluation, tuned runtime extractor |
| `classify/` | label table and grouped splits, baselines, BERT fine-tuning, metrics with bootstrap intervals |
| `market/` | DuckDB aggregates with Wilson intervals and suppression, co-occurrence PMI, NMF role profiles |
| `gap/` | CV parsing, gap matching, score, e5 embeddings, LLM with structured output, guards, evaluation |
| `manifest.py`, `colab.py`, `report.py` | reproducibility manifests, Colab packaging and import, result rendering |
| `api/`, `app/` | FastAPI service and Streamlit UI (HTTP client only) |

Storage: SQLite for the raw store and HTTP cache, Parquet for the corpus and ESCO table, DuckDB for aggregate queries. Nothing in `data/`, `stores/`, `models/`, `annotation/` (except the guidelines in `docs/`) or `logs/` is committed.
