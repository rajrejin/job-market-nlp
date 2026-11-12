# Known issues

1. **Bundesagentur data not used.** Its terms forbid automated collection (DATA_TERMS.md). The original plan (a corpus stratified by Bundesagentur occupational fields) was replaced by the Arbeitnow feed plus crawled career sites; the classifier uses Arbeitnow department tags as labels.
2. **Gold set: one model annotator, one pass.** No independent second pass, so no agreement figure. Export for human review with `jobnlp annotate-export`.
3. **Test half evaluated twice.** After corpus-level inspection showed fragment matches, the extraction was reworked and the test half evaluated again; both evaluations are kept (`results/extraction_eval_v1.json`, `results/extraction_eval.json`, DECISIONS.md item 16).
4. **Short time window.** One Arbeitnow snapshot (about one week) and one crawl; trend tests are hypotheses and partly reflect changing employer mix. The weekly task (scripts/schedule_harvest.ps1) is needed for real trends.
5. **Residual noise in market skills.** Some benefit words ("Weihnachtsgeld", "Entwicklungsmöglichkeiten") and qualifications ("Abgeschlossene Berufsausbildung") still appear among mined skills, where the section splitter labelled an offer or qualification block as requirements.
6. **ESCO licence statement** could not be retrieved on 2026-10-17 (404); the Commission's general CC BY 4.0 notice is assumed, with attribution. Six ESCO records return HTTP 500 and are missing.
7. **Crawler simplifications.** Gzipped sitemaps and microdata are skipped; 150 job pages per host per snapshot; the host list comes from a 2020 Web Data Commons extraction. Staffing agencies posting under their own name remain (robustness check excludes them by an employer-name rule).
8. **Redaction is rule-based.** A bare first name in running text can survive; product names are protected by design.
9. **Experience level and data-role rules** are not checked against labelled data.
10. **Local BERT budget.** Trained on the CPU with one seed and the reduced budget in `configs/default.yaml`; the full run (larger model, three seeds) is prepared for Colab but not executed.
11. **Windows Smart App Control** blocks some compiled Python modules natively on this laptop; everything runs in the development container instead.
12. **Docker image size** is about 5 GB (PyTorch, transformers, spaCy models).
