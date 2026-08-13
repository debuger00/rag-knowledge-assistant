"""Parse Obsidian wikilinks without treating code examples as links."""
from __future__ import annotations

from pathlib import PurePosixPath
import re

from rag_core.graph.models import WikiLink


_WIKILINK_RE = re.compile(r"(?P<embedded>!)?\[\[(?P<body>[^\]\r\n]{1,300})\]\]")
_INLINE_CODE_RE = re.compile(r"`+[^`\r\n]*`+")
_ATTACHMENT_SUFFIXES = {
    ".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".pdf", ".png",
    ".svg", ".webm", ".webp", ".wav", ".mp3", ".mp4",
}


def _without_code(text: str) -> str:
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = "```" if stripped.startswith("```") else "~~~" if stripped.startswith("~~~") else None
        if marker:
            fence = None if fence == marker else marker if fence is None else fence
            lines.append("\n" if line.endswith("\n") else "")
            continue
        if fence is not None:
            lines.append("\n" if line.endswith("\n") else "")
        else:
            lines.append(_INLINE_CODE_RE.sub("", line))
    return "".join(lines)


def is_attachment_target(target: str) -> bool:
    suffix = PurePosixPath(target.replace("\\", "/")).suffix.lower()
    return suffix in _ATTACHMENT_SUFFIXES


def parse_wikilinks(text: str) -> list[WikiLink]:
    links: list[WikiLink] = []
    for match in _WIKILINK_RE.finditer(_without_code(text)):
        body = match.group("body").strip()
        if not body or body == "..." or "&&" in body or "\x00" in body:
            continue
        target_and_anchor, separator, alias = body.partition("|")
        target, anchor_separator, anchor = target_and_anchor.partition("#")
        target = target.strip().replace("\\", "/")
        anchor = anchor.strip() if anchor_separator else ""
        if not target and not anchor:
            continue
        links.append(WikiLink(
            target=target,
            anchor=anchor,
            alias=alias.strip() if separator else "",
            embedded=bool(match.group("embedded")),
            raw=match.group(0),
        ))
    return links

