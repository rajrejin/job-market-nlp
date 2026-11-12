# German job market NLP and CV gap tool

Two things in one repo:

1. An NLP pipeline over real German job postings: pull the skills out of each posting, classify
   the posting into an occupational field, and count which skills are in demand where.
2. A CV gap tool: give it a job posting and your CV and it tells you which required skills you
   cover, which are related to something you have, and which are missing. A small local LLM then
   suggests how to word things, with guards so it cannot invent skills you do not have.

Everything runs on a CPU laptop. No paid API, no cloud.

## Where the data comes from

My first plan was the Bundesagentur für Arbeit Jobbörse. Its terms of use forbid automated
collection, so I dropped it ([docs/DATA_TERMS.md](docs/DATA_TERMS.md)). The corpus instead comes
from:

- the public [Arbeitnow](https://www.arbeitnow.com) job board API, and
- employer career pages that publish schema.org `JobPosting` markup, crawled politely
  (robots.txt respected, one request per 5 seconds per site, everything cached).

That gave 10,284 postings, 5,118 of them German postings after removing near-duplicates. Emails,
phone numbers and contact names are stripped at ingestion. The posting text itself stays on my
machine and is not in this repo, only aggregated results are.

This is not a representative sample of the German labour market. Arbeitnow leans towards tech
and startups, the crawled sites towards large employers and staffing agencies.

## Results

| Task | Best approach | Score |
|---|---|---|
| Skill extraction (held-out half of the gold set) | ESCO dictionary + mined terms + weakly supervised NER | partial-match F1 0.57, exact F1 0.46 |
| Occupational field, 12 classes | TF-IDF + logistic regression on title and description | macro-F1 0.60 (95% CI 0.53 to 0.65) |
| Must-have vs nice-to-have | cue-word rules | accuracy 0.94 |
| LLM suggestions passing the guards | Qwen2.5 3B via Ollama, with retries | 100% (17.5% failed on the first attempt) |

What I learned along the way:

- The ESCO skill dictionary alone finds only 13 to 17% of the annotated skills. Job ads say
  "Kenntnisse in SAP FI", ESCO says something much more formal. Mining candidate terms after cue
  phrases ("Kenntnisse in", "experience with") roughly doubled F1.
- Plain TF-IDF beat a fine-tuned German DistilBERT (macro-F1 0.60 vs 0.47). With about 1,100
  training postings, noisy labels and a CPU-sized training budget, the simple model wins.
- Skill rankings are fragile. Counting one posting per employer changes 4 of the top 10 skills,
  so I report the robustness table next to every ranking.

One important caveat: **the 200-posting gold set for skill extraction was annotated by a language
model in a single pass, not by human annotators.** Treat the extraction scores as indicative.
`jobnlp annotate-export` writes the annotations to a review sheet so they can be corrected.

All tables are in [docs/RESULTS.md](docs/RESULTS.md).

## Running it

I run everything inside a development container, because Windows Smart App Control on my laptop
blocks some compiled Python packages in a normal virtual environment. The project folder is
mounted into the container, so results appear in the normal folders.

```powershell
docker build -f docker/Dockerfile.dev -t jobnlp-dev .      # once, about 10 minutes
.\scripts\run_pipeline.ps1                                  # all steps in order
.\scripts\dev.ps1 python -m jobnlp.cli market               # or a single step
.\scripts\dev.ps1 python -m pytest -q                       # tests
```

On Linux or macOS use `scripts/dev.sh`, or skip the container: `pip install -r requirements.txt
&& pip install -e .`

Pipeline steps, in order: `harvest`, `crawl`, `esco`, `build-corpus`, `train-ner`, `evaluate`,
`market`, `train-classifier`, `evaluate-gap`, `report`.

API and app:

```bash
docker compose -f docker/docker-compose.yml up --build      # API on :8000, Streamlit on :8501
```

CV gap on the command line (nothing is stored, output goes to the terminal):

```powershell
.\scripts\dev.ps1 python -m jobnlp.cli gap job.txt data/private/my_cv.pdf
```

The LLM suggestions need [Ollama](https://ollama.com) with `qwen2.5:3b-instruct` running on the
host. Without it you still get the gap and the score.

## How it works

- **Cleaning** (`src/jobnlp/text/`): HTML to text, redaction, language detection, splitting
  postings into sections (tasks, requirements, benefits), MinHash near-duplicate detection.
- **Skills** (`src/jobnlp/skills/`): ESCO labels in German and English, terms mined from
  requirement sections, a spaCy NER model trained on those weak labels, and cue-word rules for
  must-have versus nice-to-have.
- **Classifier** (`src/jobnlp/classify/`): TF-IDF, e5 embeddings and DistilBERT compared on
  splits grouped by employer, so the same company never appears in both train and test.
- **Market** (`src/jobnlp/market/`): share of postings mentioning each skill with Wilson
  intervals, by state, region, language and experience level. Groups under 30 postings are hidden.
- **CV gap** (`src/jobnlp/gap/`): deterministic matching and a transparent weighted score first,
  the LLM only for wording. Its output is checked against the CV and rejected if it adds skills
  or facts.
- **API** (`api/`): FastAPI with size limits, typed uploads (txt, pdf, docx) and no CV text in logs.

More: [architecture](docs/ARCHITECTURE.md), [model card](docs/MODEL_CARD.md),
[decisions](docs/DECISIONS.md), [known issues](docs/KNOWN_ISSUES.md),
[Colab runbook for the larger BERT run](docs/COLAB_RUNBOOK.md).

## Limitations

- One snapshot of about a week, so the trend analysis is only a sketch. It needs the weekly
  harvest (`scripts/schedule_harvest.ps1`) running for a few months.
- Job postings show what employers ask for, not who gets hired.
- The department tags used as classifier labels are noisy and overlap.
- The match score is a heuristic. It does not predict what a recruiter or an ATS will do.

## Credits

Skills from the ESCO classification, © European Union. Job data from the Arbeitnow API (thank you
for keeping it free). Locations from GeoNames (CC BY 4.0). Host list from Web Data Commons.
