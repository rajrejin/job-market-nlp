# Data terms check (Phase 1)

Accessed 2026-10-17.

## Sources read

1. Nutzungsbedingungen of the Arbeitsagentur portal, https://www.arbeitsagentur.de/nutzungsbedingungen (dated "Stand 08.07.2026"). Applies to the whole portal and apps, with job and applicant offers regulated separately in chapter 6.
2. Impressum, https://www.arbeitsagentur.de/impressum (copyright section).
3. bundesAPI documentation, https://github.com/bundesAPI/jobsuche-api (unofficial; states no terms, licence or rate limits).

## What they say

- Section 2a, paragraph 3: it is not permitted to use "Robots, Webspider oder ähnliche Technologien", nor to use interfaces against their intended purpose, for example for data collection and evaluation (as summarised from the fetched page; wording was shortened by the fetch tool, so re-read the original before relying on it).
- Section 6d, paragraphs 4 and 5: downloaded postings must be deleted after the hiring process ends; building one's own job pool is prohibited for private and temp agencies.
- Section 3 and the Impressum: the BA holds copyright in the portal as a whole; reproduction or onward use needs prior written permission and attribution.
- bundesAPI: the interface is unofficial, with a publicly listed shared client ID; no permission for automated bulk use is stated.

## Conclusion

Automated bulk retrieval of postings for data collection and analysis is forbidden by the current terms. Under the build specification (Section 4.1) the project therefore does **not** harvest, and does not use third-party scraping services as a workaround. Only data-independent parts are built. Options for the owner are in HANDOFF.md.
