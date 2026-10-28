# German job market NLP analyser and CV gap tool

Skill extraction, occupational-field classification and skill-demand analysis on real German job postings, plus a local CV gap analyser with a guarded local LLM. Everything runs on a CPU laptop; no paid API or cloud service is used.

## Summary

<!-- results:start -->
- Corpus: 5118 German postings after deduplication (published 2014-08-14 to 2026-10-17).
- Skill extraction, held-out gold half (union): partial-match F1 0.568, exact-span F1 0.458 (gold set annotated by a model, see limitations).
- Occupational-field classifier: best tfidf_lr:title+description, macro-F1 0.596 (95% CI 0.531 to 0.652).
- CV matching sanity: same-field versus different-field AUC 0.600.
<!-- results:end -->

Full tables: [docs/RESULTS.md](docs/RESULTS.md). Every number there is generated from a file in `results/`.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the diagram and module list. In short: two permitted sources (the Arbeitnow public API and employer career sites with schema.org JobPosting markup) feed a raw store; a corpus build cleans, redacts, locates and deduplicates; an ESCO-based extractor tuned on a gold set feeds the market analysis and the CV tool; a FastAPI service and a Streamlit UI sit on top.

## Quick start

Everything runs in a Linux development container with the project folder mounted, so results land in the same folders you see in Explorer. (On this Windows laptop, Smart App Control blocks some compiled Python packages in a native virtual environment; the container avoids that without touching security settings.)

```
docker build -f docker/Dockerfile.dev -t jobnlp-dev .      # once, about 10 minutes
.\scripts\run_pipeline.ps1                                  # every step in order
.\scripts\dev.ps1 python -m jobnlp.cli market               # or any single step
.\scripts\dev.ps1 python -m pytest -q                       # tests
docker compose -f docker/docker-compose.yml up --build      # API on :8000, app on :8501
```

Pipeline steps (`python -m jobnlp.cli <step>`): `harvest`, `crawl`, `esco`, `build-corpus`, `train-ner` (tunes the extractor on the gold development half, needs `annotation/`), `evaluate`, `market`, `train-classifier`, `evaluate-gap`, `report`.

CV analysis on the command line (output to the terminal only): `.\scripts\dev.ps1 python -m jobnlp.cli gap job.txt data/private/my_cv.pdf`. The local LLM needs Ollama on the host with `qwen2.5:3b-instruct` (`ollama pull qwen2.5:3b-instruct`); without it the tool still returns the gap and the score.

Weekly snapshots: run `scripts\schedule_harvest.ps1` once in PowerShell to register a Sunday task in Windows Task Scheduler (it runs `scripts\harvest.ps1`: harvest, crawl and rebuild in the container; Docker Desktop must be running).


## Data

- **Sources.** The Bundesagentur für Arbeit Jobbörse was the intended source, but its terms forbid robots and the use of its interfaces for data collection, so it is not used ([docs/DATA_TERMS.md](docs/DATA_TERMS.md)). Instead: the [Arbeitnow](https://www.arbeitnow.com) public job board API (jobs listed by Arbeitnow; thank you for the free API), and employer career sites that publish schema.org JobPosting markup, found through the Web Data Commons host list and crawled with robots.txt and text-and-data-mining reservations honoured, at most one request per 5 seconds per site. Job boards and agency feeds are excluded.
- **Coverage and bias.** Arbeitnow shows about one week of postings and leans towards technology, startups and English-language roles; the crawled career sites lean towards larger employers, care providers and staffing agencies that publish structured markup. Neither is representative of the whole German labour market. Only postings active at harvest time are visible, so long-running postings are over-represented; real trends need the weekly snapshots.
- **Privacy.** Emails, phone numbers, URLs and contact names are removed at ingestion; posting text stays in gitignored folders; only aggregates are published.
- **ESCO.** Skills from the ESCO classification, © European Union, https://esco.ec.europa.eu, via the ESCO web service API. Locations from GeoNames (CC BY 4.0).

## Method

- **Cleaning:** HTML to text with headings kept, redaction, lingua language detection, rule-based German and English section splitting, MinHash near-duplicate clusters.
- **Skill extraction:** ESCO labels (German and English, surface and lemma forms) plus terms mined from requirement sections after cues such as "Kenntnisse in" or "experience with" and from short list items; thresholds and blocked keys tuned on the gold development half; a spaCy NER model trained on the dictionary's weak labels; must-have versus nice-to-have from necessity cues.
- **Gold set:** 200 real postings (120 German, 80 English), requirement sections annotated at span level following [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md). **The gold set was annotated by a language model in a single pass, not by independent human experts**; no agreement figure exists. `jobnlp annotate-export` writes it to an editable review sheet; corrections rerun without code changes.
- **Classifier:** platform department tags as labels; TF-IDF and e5 embedding baselines with logistic regression and a fine-tuned German DistilBERT, on splits grouped by employer and duplicate cluster, with bootstrap intervals and ablations (title only, description only, both).
- **Market analysis:** shares of postings mentioning each skill, with 95% Wilson intervals, by state, region (Rhine-Main and Rhine-Neckar), category, language, experience level, remote flag and week; groups under 30 postings suppressed; PMI co-occurrence; NMF role profiles with stability across seeds; a data and ML role view.
- **CV tool:** deterministic gap (covered, related, missing), a transparent weighted-coverage score ([docs/MODEL_CARD.md](docs/MODEL_CARD.md)) and guarded LLM suggestions.

## API and app

`GET /v1/health`, `GET /v1/ready`, `POST /v1/extract`, `POST /v1/classify`, `POST /v1/gap` (JSON) and `POST /v1/gap/upload` (txt, pdf or docx), `GET /v1/market/skills`, `GET /v1/market/profile`, `GET /metrics`. Inputs are size-limited, uploads are type-checked and parsed with a timeout, errors follow one schema without stack traces, logs never contain CV or posting text, an API key can be set with `JOBNLP_API_KEY`, and CORS is off unless `JOBNLP_CORS_ORIGINS` is set. The Streamlit app has two tabs, Market Trends and CV Analyser, and talks to the API over HTTP only.

## Docker

```
docker compose -f docker/docker-compose.yml up --build
```

The API (port 8000) and the app (port 8501) run as non-root containers; data, models, stores and results are mounted from the project folder; Ollama is reached on the host.

## Limitations

- The data sources are not official labour-market statistics and can change or end; Arbeitnow can revoke API access.
- Only currently active postings are visible: the window favours long-running postings, and trends need repeated snapshots.
- Platform department tags are noisy, overlapping labels.
- Postings are a proxy for employer demand, not for hiring.
- The gold set is model-annotated by one annotator in one pass.
- ESCO misses many tools and German soft-skill nouns; mined skills are less reliable.
- The match score is a heuristic, not a prediction of recruiter decisions or applicant tracking systems.
- Small groups are suppressed; results come from one machine and one harvest period.

## Reproduction

See [HANDOFF.md](HANDOFF.md) for the exact commands, timings and status, and [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) for a step-by-step path with checks.

## References

- ESCO classification, European Commission, https://esco.ec.europa.eu
- Arbeitnow job board API, https://www.arbeitnow.com/blog/job-board-api
- Web Data Commons, Schema.org Table Corpus, https://webdatacommons.org/structureddata/schemaorgtables/
- GeoNames, https://www.geonames.org
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158), 209-212.
- spaCy, https://spacy.io; Hugging Face Transformers, https://huggingface.co/docs/transformers; multilingual-e5, https://huggingface.co/intfloat/multilingual-e5-small
