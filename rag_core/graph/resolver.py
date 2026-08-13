"""Resolve Obsidian wikilinks against the current vault."""
from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
from urllib.parse import unquote

from rag_core.graph.models import LinkResolution, WikiLink
from rag_core.graph.parser import is_attachment_target


def _without_md(value: str) -> str:
    return value[:-3] if value.lower().endswith(".md") else value


class ObsidianLinkResolver:
    def __init__(
        self,
        sources: list[str] | set[str],
        aliases: dict[str, list[str]] | None = None,
    ):
        self._exact: dict[str, str] = {}
        self._basenames: dict[str, list[str]] = defaultdict(list)
        self._aliases: dict[str, list[str]] = defaultdict(list)
        for original in sorted({source.replace("\\", "/") for source in sources}):
            normalized = _without_md(original).strip("/").casefold()
            self._exact[normalized] = original
            basename = PurePosixPath(normalized).name
            self._basenames[basename].append(original)
        for source, values in (aliases or {}).items():
            for alias in values:
                normalized_alias = alias.strip().casefold()
                if normalized_alias and source not in self._aliases[normalized_alias]:
                    self._aliases[normalized_alias].append(source)

    def resolve(self, link: WikiLink, current_source: str) -> LinkResolution:
        target = unquote(link.target).strip().replace("\\", "/")
        if not target:
            return LinkResolution(status="resolved", source=current_source)
        if link.embedded or is_attachment_target(target):
            return LinkResolution(status="attachment")

        normalized = _without_md(target).strip("/")
        exact = self._exact.get(normalized.casefold())
        if exact:
            return LinkResolution(status="resolved", source=exact)

        current_dir = PurePosixPath(current_source).parent
        relative = _without_md(str(current_dir / normalized)).strip("/")
        exact = self._exact.get(relative.casefold())
        if exact:
            return LinkResolution(status="resolved", source=exact)

        alias_candidates = tuple(self._aliases.get(normalized.casefold(), ()))
        if len(alias_candidates) == 1:
            return LinkResolution(status="resolved", source=alias_candidates[0])
        if alias_candidates:
            return LinkResolution(status="ambiguous", candidates=alias_candidates)

        basename = PurePosixPath(normalized).name.casefold()
        candidates = tuple(self._basenames.get(basename, ()))
        if len(candidates) == 1:
            return LinkResolution(status="resolved", source=candidates[0])
        if candidates:
            return LinkResolution(status="ambiguous", candidates=candidates)
        return LinkResolution(status="unresolved")
