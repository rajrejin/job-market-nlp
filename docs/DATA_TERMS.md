# Data terms

Accessed 2026-10-17. This is a record of what the sources say and how the project complies; it is not legal advice.

## 1. Bundesagentur für Arbeit Jobbörse: not used

- Nutzungsbedingungen, https://www.arbeitsagentur.de/nutzungsbedingungen ("Stand 08.07.2026"), section 2a paragraph 3: robots, web spiders or similar technologies, and use of interfaces against their purpose (for example for data collection and evaluation), are not permitted. Section 3 and the Impressum (https://www.arbeitsagentur.de/impressum): copyright of the portal lies with the BA; reproduction needs written permission.
- bundesAPI documentation (https://github.com/bundesAPI/jobsuche-api) is unofficial and states no permission for bulk use.
- Conclusion: the BA interface is forbidden for this purpose. Third-party scrapers of the BA (for example on Apify) and datasets harvested from it (for example Hugging Face `mischeiwiller/german-job-postings`) were not used, as they would route around these terms.

## 2. Arbeitnow job board API: used

- Endpoint https://www.arbeitnow.com/api/job-board-api, no key. The API response itself states: "This is a free public API for jobs, please do not abuse. I would appreciate linking back to the site. By using the API, you agree to the terms of service present on Arbeitnow.com".
- Terms of service (https://www.arbeitnow.com/terms): permission to use the API can be revoked; users agree to link back; site materials are licensed for personal, non-commercial transitory viewing, without public display, and downloaded materials must be destroyed when the licence ends.
- robots.txt (https://www.arbeitnow.com/robots.txt): no disallowed paths.
- Compliance: one request per second; non-commercial research and portfolio use; raw and processed posting text stays on the local machine (gitignored) and is never displayed publicly; only aggregates (counts, shares, models, metrics) are published; the README links back to Arbeitnow. If the licence ends, delete `stores/postings.sqlite`, `stores/http_cache.sqlite` and `data/`.

## 3. Employer career sites (schema.org JobPosting): used, with exclusions

- Candidate hosts come from the Web Data Commons Schema.org Table Corpus statistics (https://webdatacommons.org/structureddata/schemaorgtables/, JobPosting class, September 2020 extraction): hosts that published JobPosting markup. Only the statistics file (host names and row counts) was downloaded.
- Legal basis assumed: section 44b UrhG (text and data mining of lawfully accessible works is permitted unless the rights holder has reserved it in machine-readable form). Reservations are honoured: robots.txt (a disallow for the project user agent or for all agents stops the crawl of that host), TDMRep (`/.well-known/tdmrep.json`, the `tdm-reservation` meta tag).
- Excluded: job boards, staffing-agency boards and media sites, by host-name rules and by a data rule (a host whose postings name more than three hiring organisations is a board). Rationale: job boards commonly forbid automated collection in their terms, and their listings belong to third parties.
- Politeness: at most one request per 5 seconds per host, job URLs only from the host's own sitemaps, user agent `jobnlp-research/0.1 (non-commercial portfolio project)`, every response cached.
- Storage and reuse: as for Arbeitnow, text stays local, only aggregates are published.

## 4. ESCO: used

- ESCO web service API, https://ec.europa.eu/esco/api, no key; 13,479 of 13,485 skill records retrieved (6 records returned HTTP 500 from the server and were skipped, logged in `logs/esco.log`).
- The ESCO download page sends a download link by email, which counts as registration, so the API was used instead.
- Licence: the Commission's legal notice states that EU-owned content on Commission websites is licensed under CC BY 4.0 unless otherwise indicated. An ESCO-specific licence statement could not be retrieved on the access date (the FAQ pages returned 404); this is listed in KNOWN_ISSUES.md. Attribution used: "ESCO classification, © European Union, https://esco.ec.europa.eu".

## 5. GeoNames: used

- https://download.geonames.org/export/dump/DE.zip and admin1CodesASCII.txt, licence CC BY 4.0 (GeoNames). Used to map location strings to German cities and states.

## 6. Models

- spaCy `de_core_news_lg` and `en_core_web_md` (MIT), `intfloat/multilingual-e5-small` (MIT), `distilbert/distilbert-base-german-cased` (Apache 2.0), `deepset/gbert-base` (MIT, Colab notebook only), local LLM `qwen2.5:3b-instruct` through Ollama (Qwen2.5-3B-Instruct is listed with licence "other", the Qwen research licence; used locally for personal, non-commercial suggestions, outputs not redistributed). Licences checked on the Hugging Face hub on 2026-10-17.
