# Decisions

1. Project root: `D:\Work\Code` held unrelated files, so the project lives in the new subfolder `job-market-nlp` (the layout name from the specification). Alternatives: build in the non-empty folder. Reason: avoid mixing with unrelated work.
2. Python 3.13 is installed locally (spec asks 3.11+). Packages are installed in `.venv`; if a dependency lacks 3.13 wheels, it will be recorded in KNOWN_ISSUES.
3. Colab: the user opened Colab in Chrome. Colab steps stay optional; no data is uploaded to Drive without explicit user confirmation in chat.
4. Data source pivot (2026-10-17). The Bundesagentur Jobbörse is excluded by its terms (DATA_TERMS.md), and the user ruled out manual collection and any registration. Sources checked without registration:
   - Arbeitnow free job board API (https://www.arbeitnow.com/api/job-board-api): offered publicly to third parties, no key, robots.txt allows all; terms ask for a link back, allow revocation, and limit site materials to non-commercial use with no public display. Full HTML descriptions (median about 4,800 characters in a probe of 1,664 postings), about 26 percent of them in German locations, with roughly a week of history visible (pages ran to about 35). **Chosen as primary source.** Our use is non-commercial, stored privately, with only aggregates published, which fits these terms.
   - Hugging Face `mischeiwiller/german-job-postings` (CC-BY-4.0, 70,584 rows): titles and metadata only, with no description text, and itself harvested from the Bundesagentur interface. Rejected: no text, and it would route around the Bundesagentur terms.
   - SmartRecruiters public posting feeds (no key): intended for companies' own career sites, and coverage dominated by a few employers (Bosch 774 German postings). Rejected.
   - Greenhouse and Personio public boards: per company, few German postings, no way to discover companies at scale (the Common Crawl index timed out). Rejected for now.
   - Web Data Commons JobPosting tables (about 3.06 million rows, 2020 extraction, published "for research purposes", no explicit licence): possible historical supplement, not used for now.
   - ESCO web service API works without a key (verified with a live query).
   Consequences: corpus size grows through weekly snapshots instead of a single 20,000-posting harvest; Arbeitnow tags replace Bundesagentur occupational fields as classifier labels (noisier); the Rhine-Main and Rhine-Neckar view depends on how many postings fall there.
