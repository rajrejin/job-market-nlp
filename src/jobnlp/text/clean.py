"""HTML to plain text. Headings are kept as lines starting with '## ' for the section splitter."""

from __future__ import annotations

import html
import re
import unicodedata

from bs4 import BeautifulSoup, NavigableString, Tag

BLOCK = {
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "tr",
    "section",
    "article",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "header",
    "footer",
}
HEADING = {"h1", "h2", "h3", "h4", "h5", "h6"}
MAX_HEADING_CHARS = 80


def unescape_html(text: str) -> str:
    """Unescape until stable: some feeds escape the HTML twice."""
    for _ in range(3):
        new = html.unescape(text)
        if new == text:
            break
        text = new
    return text


def _is_heading(tag: Tag) -> bool:
    if tag.name in HEADING:
        return True
    # a paragraph or list item whose whole content is bold acts as a heading
    if tag.name in {"p", "div", "li"}:
        kids = [k for k in tag.children if not (isinstance(k, NavigableString) and not k.strip())]
        if len(kids) == 1 and isinstance(kids[0], Tag) and kids[0].name in {"strong", "b"}:
            return len(tag.get_text(strip=True)) <= MAX_HEADING_CHARS
    return False


def html_to_text(raw: str) -> str:
    soup = BeautifulSoup(unescape_html(raw or ""), "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    lines: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        s = " ".join("".join(buf).split())
        if s:
            lines.append(s)
        buf.clear()

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                buf.append(str(child))
            elif isinstance(child, Tag):
                if child.name in BLOCK:
                    flush()
                    if _is_heading(child):
                        txt = " ".join(child.get_text(" ").split())
                        if txt:
                            lines.append("## " + txt)
                        continue
                    first = len(lines)
                    walk(child)
                    flush()
                    # mark list items on their first line (li content is often wrapped in <p>)
                    if child.name == "li" and len(lines) > first:
                        lines[first] = "- " + lines[first].removeprefix("- ")
                else:
                    walk(child)

    walk(soup)
    flush()
    return normalise("\n".join(lines))


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
