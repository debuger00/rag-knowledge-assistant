from rag_core.graph.parser import is_attachment_target, parse_wikilinks
from rag_core.graph.resolver import ObsidianLinkResolver


def test_parser_ignores_code_and_preserves_obsidian_parts():
    text = """
[[notes/Target#Section|Alias]]
`[[nodiscard]]`
```cpp
[[maybe_unused]]
```
![[image.png]]
"""
    links = parse_wikilinks(text)

    assert len(links) == 2
    assert links[0].target == "notes/Target"
    assert links[0].anchor == "Section"
    assert links[0].alias == "Alias"
    assert links[1].embedded
    assert is_attachment_target(links[1].target)


def test_resolver_supports_relative_and_unique_basename():
    resolver = ObsidianLinkResolver({
        "area/current.md",
        "area/relative.md",
        "other/unique.md",
    })
    relative = parse_wikilinks("[[relative]]")[0]
    unique = parse_wikilinks("[[unique]]")[0]

    assert resolver.resolve(relative, "area/current.md").source == "area/relative.md"
    assert resolver.resolve(unique, "area/current.md").source == "other/unique.md"


def test_resolver_reports_ambiguous_and_unresolved_targets():
    resolver = ObsidianLinkResolver({"a/same.md", "b/same.md"})

    ambiguous = resolver.resolve(parse_wikilinks("[[same]]")[0], "root.md")
    missing = resolver.resolve(parse_wikilinks("[[missing]]")[0], "root.md")

    assert ambiguous.status == "ambiguous"
    assert set(ambiguous.candidates) == {"a/same.md", "b/same.md"}
    assert missing.status == "unresolved"


def test_resolver_supports_frontmatter_aliases():
    resolver = ObsidianLinkResolver(
        {"notes/long-name.md"},
        {"notes/long-name.md": ["Short Name"]},
    )

    result = resolver.resolve(parse_wikilinks("[[Short Name]]")[0], "index.md")

    assert result.status == "resolved"
    assert result.source == "notes/long-name.md"
