# Observed source schemas

Recorded from real responses on 2026-10-17. The sources can change without notice; the harvest code only relies on the fields listed here.

## Arbeitnow job board API

`GET https://www.arbeitnow.com/api/job-board-api?page=N` (no key).

Response: `{"data": [...], "links": {"first", "last", "prev", "next"}, "meta": {"current_page", "from", "path", "per_page", "to", "terms", "info"}}`.

Observed behaviour: `per_page` varied between 300 and 50 depending on the page; `links.next` is `null` on the last page; the first snapshot had 38 pages and 6,425 rows (6,360 unique `slug` values, because the list shifts while paging); postings are ordered by `created_at` and the visible window was about one week. `meta.info` says jobs are updated every hour.

Fields per posting (types observed):

| field | type | use |
|---|---|---|
| `slug` | str | posting id |
| `company_name` | str | employer |
| `title` | str | title |
| `description` | str (HTML, sometimes HTML-escaped twice) | text after cleaning and redaction |
| `remote` | bool | remote flag |
| `url` | str | not used beyond provenance |
| `tags` | list of str | platform category labels (classifier target) |
| `job_types` | list of str | employment type (free text) |
| `location` | str | free text; mapped to German places with GeoNames |
| `created_at` | int (Unix seconds) | publication date |

## schema.org JobPosting (JSON-LD on employer career sites)

Read from `<script type="application/ld+json">` blocks, including `@graph` containers. Fields kept, with the share of stored crawled postings where the field was non-empty (2,089 postings): `title` (100 %), `description` (85.8 %), `hiringOrganization.name` (99.7 %), `jobLocation.address.addressLocality` (96.9 %), `datePosted` (99.9 %), `employmentType` (78.1 %), `industry` (16.5 %), `occupationalCategory` (3.0 %). Only postings with `jobLocation.address.addressCountry` equal to DE, DEU, Germany or Deutschland are stored.

## ESCO web service API

`GET https://ec.europa.eu/esco/api/search?type=skill&language=de&limit=100&offset=P&full=true`. Observed: `offset` is a page index, not a row offset (`offset=2` with `limit=100` returns rows 200 to 299); `total` was 13,485; some pages return HTTP 500 (records 453, 2180, 5194, 5209, 5218 and 8080 failed even singly). Each result carries `preferredLabel` and `alternativeLabel` as dictionaries by language, and `_links.hasSkillType`, `hasReuseLevel`, `broaderHierarchyConcept`, `isInScheme`.
