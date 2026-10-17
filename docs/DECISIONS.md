# Decisions

1. Project root: `D:\Work\Code` held unrelated files, so the project lives in the new subfolder `job-market-nlp` (the layout name from the specification). Alternatives: build in the non-empty folder. Reason: avoid mixing with unrelated work.
2. Python 3.13 is installed locally (spec asks 3.11+). Packages are installed in `.venv`; if a dependency lacks 3.13 wheels, it will be recorded in KNOWN_ISSUES.
3. Colab: the user opened Colab in Chrome. Colab steps stay optional; no data is uploaded to Drive without explicit user confirmation in chat.
