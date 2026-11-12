# Annotation guidelines: skill spans in job postings

## What is annotated

The text of each gold item is the requirements section of a real posting (the tasks section when no requirements section was found), capped at 1,200 characters. Only that text is annotated, so the gold set measures extraction where skills are densest.

## What counts as a skill

Annotate a span when it names something a candidate is expected to know, use or be:

| type | meaning | examples (as they appear in postings) |
|---|---|---|
| `tool` | software, programming language, framework, platform, product | Python, SQL, SAP S/4HANA, Excel, Kubernetes, Figma |
| `knowledge` | domain knowledge or method | Machine Learning, Buchhaltung, Projektmanagement, Arbeitsrecht, agile Methoden |
| `soft` | personal or interpersonal competence named as a noun or fixed adjective phrase | Teamfähigkeit, Kommunikationsstärke, analytisches Denken, stakeholder management |
| `language` | a natural language requirement | Deutsch, Englisch, Deutschkenntnisse, fluent English (span: English) |
| `certificate` | licence or certificate | Führerschein Klasse B, Staplerschein, AWS Certified Solutions Architect |

## What is not annotated

- Degrees, fields of study and vocational qualifications (Studium der Informatik, abgeschlossene Ausbildung als Kauffrau), years of experience, seniority.
- Generic attitudes without a skill core: motivation, Spaß an der Arbeit, Leidenschaft, hands-on mentality.
- Benefits, tasks phrased as duties without a skill noun ("du betreust unsere Kunden").

## Boundaries

1. Take the minimal phrase that names the skill. Drop proficiency words and frames: "sehr gute Kenntnisse in SQL" gives `SQL`; "Erfahrung mit Docker" gives `Docker`; "verhandlungssichere Englischkenntnisse" gives `Englischkenntnisse`.
2. Coordinated items are separate spans when each is complete ("Python und SQL" gives two spans). A German ellipsis compound stays one span ("Projekt- und Prozessmanagement").
3. A version or variant belongs to the span when written together ("Java 17", "SAP S/4HANA").
4. Parenthesised examples are separate spans ("Cloud-Plattformen (AWS, Azure)" gives `Cloud-Plattformen`, `AWS`, `Azure`).
5. Language levels are not part of the span ("Deutsch (C1)" gives `Deutsch`).

## Necessity

Each span gets `must` or `nice`:

- `nice` when the span is under a heading or in a sentence marked as optional: wünschenswert, von Vorteil, idealerweise, gerne, ein Plus, nice to have, bonus, preferred, ideally, a plus.
- `must` otherwise (the default for requirement lists).

## Procedure and status

The gold set was annotated by a language model in a single pass, reading each text in full before writing spans as exact substrings. It was not annotated by independent human experts, and no second independent pass was possible; inter-annotator agreement is therefore not reported. `jobnlp annotate-export` writes the gold set to JSONL and a CSV review sheet so a person can correct it; evaluation reruns without code changes.
