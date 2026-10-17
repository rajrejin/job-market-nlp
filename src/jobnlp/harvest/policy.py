"""Crawl policy: robots.txt, text-and-data-mining (TDM) reservations, and excluded hosts.

Legal basis assumed: section 44b UrhG permits text and data mining of lawfully accessible
content unless the rights holder reserved it in machine-readable form. Reservations are read
from robots.txt, the TDMRep protocol (/.well-known/tdmrep.json, `tdm-reservation` header and
meta tag). Job boards, staffing agencies and media sites are excluded on top of that, because
their terms commonly forbid automated collection; only employer career sites are crawled.
"""

from __future__ import annotations

import json
import re
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

BOARD_PATTERNS = re.compile(
    r"(job|stellen|karriere-?(portal|boerse)|career(s)?board|azubi|aubi|praktik|minijob|"
    r"freelanc|talent|recruit|personal(vermittlung|service|dienst)|zeitarbeit|adecco|randstad|"
    r"orizon|gigroup|dekra-arbeit|manpower|hays|academics|connecticum|indeed|stepstone|monster|"
    r"xing|linkedin|kimeta|arbeitsagentur|meinestadt|kalaydo|zeit\.de|sueddeutsche|faz\.net|"
    r"allgemeine|zeitung|online\.de$|shz\.de|mv-online|twago|campus|experts|personal|gesucht|"
    r"hogapage|aerzteblatt|flagma|poolia|best2find|jigub|consulting-?jobs|vermittlung)",
    re.IGNORECASE,
)


def is_excluded_host(host: str) -> bool:
    """True for hosts that look like job boards, agencies, or media rather than employers."""
    return bool(BOARD_PATTERNS.search(host))


def parse_robots(text: str, url: str) -> RobotFileParser:
    rp = RobotFileParser()
    rp.set_url(url)
    rp.parse(text.splitlines())
    return rp


def robots_allows(rp: RobotFileParser, user_agent: str, url: str) -> bool:
    return rp.can_fetch(user_agent, url)


def robots_sitemaps(text: str) -> list[str]:
    return [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.lower().startswith("sitemap:")
    ]


def tdmrep_reserved(text: str) -> bool:
    """True if a tdmrep.json declares a reservation (tdm-reservation = 1) for any location."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    rules = data if isinstance(data, list) else [data]
    return any(isinstance(r, dict) and str(r.get("tdm-reservation")) == "1" for r in rules)


def page_reserved(html: str, headers: dict[str, str] | None = None) -> bool:
    """TDM reservation in the HTTP header or the HTML meta tag of a page."""
    if headers and str(headers.get("tdm-reservation", "")).strip() == "1":
        return True
    soup = BeautifulSoup(html[:20000], "lxml")
    meta = soup.find("meta", attrs={"name": "tdm-reservation"})
    return bool(meta and str(meta.get("content", "")).strip() == "1")
