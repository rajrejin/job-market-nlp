# Model card

Numbers for every component are in `docs/RESULTS.md`, generated from `results/`.

## 1. Skill extractor

- **What it does:** finds skill mentions in German and English job texts and maps them to a canonical skill (an ESCO skill URI, or `mined:<term>` for tools and products mined from the corpus), with a must-have or nice-to-have label.
- **How:** dictionary matching on lower-cased surface and lemma n-grams (up to 5 tokens) over ESCO preferred and alternative labels in German and English plus mined terms; German compounds such as "SQL-Kenntnisse" or "Deutschkenntnisse" resolve to their stem; one-letter tools (R, C) only next to another tool. Mining thresholds and a list of blocked keys were tuned on the gold development half. An optional spaCy NER model trained on the dictionary's weak labels can be added (the union), but the API uses the dictionary alone for speed and predictability.
- **Necessity:** rule-based. A nice-to-have cue (wünschenswert, von Vorteil, idealerweise, ein Plus, nice to have, preferred, ideally, and others) in the same sentence or list line, or in the closest heading above, marks a skill as nice-to-have; negated must cues ("nicht zwingend erforderlich") count as nice-to-have; otherwise must-have.
- **Evaluation:** on a gold set of 200 real postings (100 for tuning, 100 held out and used once), annotated by a language model in one pass, not by independent human annotators.
- **Limits:** ESCO misses many tools and German soft-skill nouns; mined terms are less reliable than ESCO labels; span boundaries differ from the annotation (for example "Arbeitsweise" versus "strukturierte Arbeitsweise"), so exact-span scores are lower than partial-match scores.

## 2. Occupational-field classifier

- **Labels:** the platform's own department tags (Arbeitnow), kept as classes when enough postings carry exactly one of them. They are set by employers' applicant tracking systems, overlap (for example "Engineering" and "Software Development") and are noisy; the confusion between related labels is a property of the label scheme.
- **Models:** TF-IDF (word 1-2 grams and character 3-5 grams) with logistic regression; multilingual-e5-small embeddings with logistic regression; fine-tuned `distilbert/distilbert-base-german-cased` (maximum 256 tokens, local reduced budget: see `configs/default.yaml`, one seed).
- **Splits:** 70/10/20, grouped so that each employer and each near-duplicate cluster sits in one split only (`results/split_manifest.json`).
- **Intended use:** routing or summarising postings by department; not for decisions about people.

## 3. CV match score

```
score = sum(w_i * c_i) / sum(w_i)       over the canonical skills i named in the posting
w_i = 1.0 if must-have, 0.4 if nice-to-have
c_i = 1.0 if the CV contains the same canonical skill (covered)
      0.5 if the CV contains a different skill whose e5 embedding similarity to it is at least
          the threshold chosen on the development half (related)
      0.0 otherwise (missing)
```

Missing skills are ranked by `w_i x (0.5 + demand) x position`, where position is 1.0 when the skill is first named in the first third of the posting and 0.8 otherwise, and demand is the skill's share among similar postings when available.

**The score is a heuristic readiness indicator.** It shows how many of the skills a posting names also appear in a CV. It does not predict recruiter decisions, interview chances or how an applicant tracking system ranks a CV, and it ignores experience depth, seniority, education and everything the extractor misses.

## 4. LLM suggestions and guards

- **Model:** `qwen2.5:3b-instruct` through a local Ollama server, temperature 0, JSON output validated against a Pydantic schema. Only the target skill, up to three posting lines that mention it and up to five CV bullets go into the prompt.
- **Guards:**
  1. Only skills from the deterministic gap are discussed.
  2. A rewrite may name only skills found in the original bullet, elsewhere in the CV, or the target skill itself when the model reported evidence in that bullet.
  3. Numbers, dates and organisation names in a rewrite must appear in the original bullet.
  4. A failing output is regenerated up to two times, then replaced by the deterministic result with a visible note.
- **Limits:** the guards only see what the skill extractor detects, so an invented skill the extractor does not know passes. Suggestions to do something (a course, a project) are labelled as suggestions, not claims.

## 5. Privacy

CV text is processed in memory by the local API, never written to disk, never logged (logs hold endpoint, status and timing only) and never sent to a network service. Posting text is redacted at ingestion (emails, phone numbers, URLs, names in contact blocks and after salutations or role titles) and stays in gitignored folders.
