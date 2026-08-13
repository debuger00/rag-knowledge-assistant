"""Export the controlled GraphRAG fixture as standalone HTML, GraphML and CSV."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.eval.run_graph_eval import build_fixture_index  # noqa: E402


EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EVAL_ROOT / "results" / "visualization"
GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


def write_graphml(path: Path, snapshot: dict) -> None:
    ET.register_namespace("", GRAPHML_NS)
    root = ET.Element(f"{{{GRAPHML_NS}}}graphml")
    node_keys = (
        ("n_label", "label", "string"),
        ("n_type", "type", "string"),
        ("n_source", "source", "string"),
        ("n_anchor", "anchor", "string"),
    )
    edge_keys = (
        ("e_type", "type", "string"),
        ("e_weight", "weight", "double"),
        ("e_evidence_source", "evidence_source", "string"),
        ("e_evidence_anchor", "evidence_anchor", "string"),
    )
    for key_id, name, value_type in node_keys:
        ET.SubElement(root, f"{{{GRAPHML_NS}}}key", {
            "id": key_id, "for": "node", "attr.name": name, "attr.type": value_type,
        })
    for key_id, name, value_type in edge_keys:
        ET.SubElement(root, f"{{{GRAPHML_NS}}}key", {
            "id": key_id, "for": "edge", "attr.name": name, "attr.type": value_type,
        })
    graph = ET.SubElement(root, f"{{{GRAPHML_NS}}}graph", {
        "id": "GraphRAGEvalFixture", "edgedefault": "directed",
    })
    for node in snapshot["nodes"]:
        element = ET.SubElement(graph, f"{{{GRAPHML_NS}}}node", {"id": node["id"]})
        values = (node["name"], node["type"], node["source"], node["anchor"])
        for (key_id, _, _), value in zip(node_keys, values):
            ET.SubElement(element, f"{{{GRAPHML_NS}}}data", {"key": key_id}).text = str(value)
    for edge in snapshot["edges"]:
        element = ET.SubElement(graph, f"{{{GRAPHML_NS}}}edge", {
            "id": edge["id"], "source": edge["source_id"], "target": edge["target_id"],
        })
        values = (
            edge["type"], edge["weight"], edge["evidence_source"],
            edge["evidence_anchor"],
        )
        for (key_id, _, _), value in zip(edge_keys, values):
            ET.SubElement(element, f"{{{GRAPHML_NS}}}data", {"key": key_id}).text = str(value)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_csv(path: Path, snapshot: dict) -> None:
    with (path / "nodes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Id", "Label", "Type", "Source", "Anchor"))
        for node in snapshot["nodes"]:
            writer.writerow((
                node["id"], node["name"], node["type"], node["source"], node["anchor"],
            ))
    with (path / "edges.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "Source", "Target", "Type", "Label", "Weight",
            "EvidenceSource", "EvidenceAnchor",
        ))
        for edge in snapshot["edges"]:
            writer.writerow((
                edge["source_id"], edge["target_id"], "Directed", edge["type"],
                edge["weight"], edge["evidence_source"], edge["evidence_anchor"],
            ))


def render_html(snapshot: dict, stats: dict) -> str:
    payload = json.dumps(snapshot, ensure_ascii=False).replace("</", "<\\/")
    node_types = sorted({node["type"] for node in snapshot["nodes"]})
    edge_types = sorted({edge["type"] for edge in snapshot["edges"]})
    node_checks = "\n".join(
        f'<label><input class="node-filter" type="checkbox" value="{escape(kind)}" '
        f'{"" if kind == "section" else "checked"}> {escape(kind)}</label>'
        for kind in node_types
    )
    edge_checks = "\n".join(
        f'<label><input class="edge-filter" type="checkbox" value="{escape(kind)}" '
        f'{"" if kind == "HAS_SECTION" else "checked"}> {escape(kind)}</label>'
        for kind in edge_types
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphRAG 可控测试集图谱</title>
<style>
  :root {{ color-scheme: dark; font-family: Inter, "Microsoft YaHei", sans-serif; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #07111f; color: #dce8f5; overflow: hidden; }}
  header {{ height: 64px; padding: 11px 18px; border-bottom: 1px solid #25364b;
    background: #0b1728; display: flex; align-items: center; justify-content: space-between; }}
  h1 {{ margin: 0; font-size: 19px; }}
  .summary {{ color: #91a7bd; font-size: 13px; }}
  main {{ display: grid; grid-template-columns: 260px 1fr 300px; height: calc(100vh - 64px); }}
  aside {{ padding: 16px; background: #0b1728; overflow: auto; }}
  aside.left {{ border-right: 1px solid #25364b; }}
  aside.right {{ border-left: 1px solid #25364b; }}
  h2 {{ font-size: 14px; margin: 4px 0 10px; color: #9fc5e8; }}
  label {{ display: block; padding: 5px 0; font-size: 13px; cursor: pointer; }}
  button {{ border: 1px solid #38506b; background: #15273d; color: #dce8f5;
    padding: 7px 10px; border-radius: 6px; cursor: pointer; margin: 8px 4px 12px 0; }}
  button:hover {{ background: #1d3551; }}
  #canvas {{ width: 100%; height: 100%; background: radial-gradient(circle at center, #10233a, #07111f 72%); }}
  .edge {{ stroke-opacity: .58; fill: none; }}
  .edge.HAS_SECTION {{ stroke: #52677d; }}
  .edge.LINKS_TO {{ stroke: #f6bd60; }}
  .edge.TAGGED_WITH {{ stroke: #63d2a1; }}
  .node circle {{ stroke: #d8e7f5; stroke-width: 1.2; cursor: pointer; }}
  .node text {{ fill: #e7f1fa; font-size: 11px; pointer-events: none;
    paint-order: stroke; stroke: #07111f; stroke-width: 3px; stroke-linejoin: round; }}
  .node.document circle {{ fill: #4285f4; }}
  .node.section circle {{ fill: #8e7dff; }}
  .node.tag circle {{ fill: #28b487; }}
  .node.unresolved circle {{ fill: #ef5b5b; }}
  .legend span {{ display: inline-flex; align-items: center; margin: 4px 8px 4px 0; font-size: 12px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
  pre {{ white-space: pre-wrap; word-break: break-word; color: #bed1e3; font: 12px/1.55 Consolas, monospace; }}
  .hint {{ color: #7f96ab; font-size: 12px; line-height: 1.5; }}
</style>
</head>
<body>
<header>
  <h1>GraphRAG 可控测试集图谱</h1>
  <div class="summary">完整数据：{stats['node_count']} 个节点 · {stats['edge_count']} 条边 · {stats['markdown_files']} 篇 Markdown</div>
</header>
<main>
  <aside class="left">
    <h2>节点类型</h2>
    {node_checks}
    <h2>边类型</h2>
    {edge_checks}
    <button id="reset">重置视图</button><button id="fit">重新布局</button>
    <div class="legend">
      <span><i class="dot" style="background:#4285f4"></i>文档</span>
      <span><i class="dot" style="background:#8e7dff"></i>章节</span>
      <span><i class="dot" style="background:#28b487"></i>标签</span>
      <span><i class="dot" style="background:#ef5b5b"></i>断链</span>
    </div>
    <p class="hint">滚轮缩放；拖动画布平移；点击节点或边查看详情。默认隐藏 section 和 HAS_SECTION，让文档关系更清晰。</p>
  </aside>
  <svg id="canvas" viewBox="0 0 1200 800">
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#91a7bd"/></marker></defs>
    <g id="viewport"><g id="edges"></g><g id="nodes"></g></g>
  </svg>
  <aside class="right"><h2>元素详情</h2><pre id="details">点击一个节点或边。</pre></aside>
</main>
<script>
const DATA = {payload};
const svg = document.getElementById('canvas');
const viewport = document.getElementById('viewport');
const edgeLayer = document.getElementById('edges');
const nodeLayer = document.getElementById('nodes');
const details = document.getElementById('details');
let transform = {{x: 0, y: 0, scale: 1}};
let dragging = null;

function enabled(selector) {{
  return new Set([...document.querySelectorAll(selector + ':checked')].map(x => x.value));
}}

function shortLabel(node) {{
  if (node.type === 'document') return node.source.replace(/\\.md$/i, '').split('/').slice(-2).join('/');
  if (node.type === 'section') return '#' + (node.anchor || node.name);
  if (node.type === 'tag') return '#' + node.name;
  return '?' + node.name;
}}

function visibleGraph() {{
  const nodeTypes = enabled('.node-filter');
  const edgeTypes = enabled('.edge-filter');
  const nodes = DATA.nodes.filter(n => nodeTypes.has(n.type));
  const ids = new Set(nodes.map(n => n.id));
  const allNodes = new Map(DATA.nodes.map(n => [n.id, n]));
  const documentBySource = new Map(
    DATA.nodes.filter(n => n.type === 'document').map(n => [n.source, n.id])
  );
  const edges = DATA.edges.filter(e => edgeTypes.has(e.type)).map(edge => {{
    let sourceId = edge.source_id, targetId = edge.target_id;
    const sourceNode = allNodes.get(sourceId), targetNode = allNodes.get(targetId);
    if (!ids.has(sourceId) && sourceNode?.type === 'section') sourceId = documentBySource.get(sourceNode.source);
    if (!ids.has(targetId) && targetNode?.type === 'section') targetId = documentBySource.get(targetNode.source);
    if (sourceId === edge.source_id && targetId === edge.target_id) return edge;
    return {{...edge, source_id:sourceId, target_id:targetId,
      visual_projection:true, original_source_id:edge.source_id, original_target_id:edge.target_id}};
  }}).filter(e => ids.has(e.source_id) && ids.has(e.target_id) && e.source_id !== e.target_id);
  return {{nodes, edges}};
}}

function layout(nodes, edges) {{
  const byId = new Map(nodes.map(n => [n.id, n]));
  const adj = new Map(nodes.map(n => [n.id, []]));
  edges.forEach(e => {{ adj.get(e.source_id).push(e.target_id); adj.get(e.target_id).push(e.source_id); }});
  const seen = new Set(), components = [];
  [...nodes].sort((a,b) => a.id.localeCompare(b.id)).forEach(start => {{
    if (seen.has(start.id)) return;
    const queue = [start.id], component = []; seen.add(start.id);
    while (queue.length) {{
      const id = queue.shift(); component.push(byId.get(id));
      adj.get(id).sort().forEach(next => {{ if (!seen.has(next)) {{ seen.add(next); queue.push(next); }} }});
    }}
    components.push(component);
  }});
  components.sort((a,b) => b.length - a.length || a[0].id.localeCompare(b[0].id));
  const cols = Math.max(1, Math.ceil(Math.sqrt(components.length * 1.5)));
  const cellW = 220, cellH = 190;
  components.forEach((component, index) => {{
    const cx = 130 + (index % cols) * cellW, cy = 110 + Math.floor(index / cols) * cellH;
    const ordered = [...component].sort((a,b) => a.type.localeCompare(b.type) || a.id.localeCompare(b.id));
    const radius = Math.min(82, Math.max(28, 18 * ordered.length));
    ordered.forEach((node, i) => {{
      const angle = -Math.PI / 2 + (2 * Math.PI * i / Math.max(ordered.length, 1));
      node.x = ordered.length === 1 ? cx : cx + radius * Math.cos(angle);
      node.y = ordered.length === 1 ? cy : cy + radius * Math.sin(angle);
    }});
  }});
}}

function show(value) {{ details.textContent = JSON.stringify(value, null, 2); }}
function el(name, attrs={{}}) {{
  const item = document.createElementNS('http://www.w3.org/2000/svg', name);
  Object.entries(attrs).forEach(([key,value]) => item.setAttribute(key, value));
  return item;
}}

function render() {{
  const {{nodes, edges}} = visibleGraph(); layout(nodes, edges);
  edgeLayer.replaceChildren(); nodeLayer.replaceChildren();
  const byId = new Map(nodes.map(n => [n.id, n]));
  edges.forEach(edge => {{
    const a = byId.get(edge.source_id), b = byId.get(edge.target_id);
    const line = el('line', {{x1:a.x, y1:a.y, x2:b.x, y2:b.y, class:'edge ' + edge.type,
      'stroke-width': Math.max(1, edge.weight * 2), 'marker-end':'url(#arrow)'}});
    line.addEventListener('click', event => {{ event.stopPropagation(); show(edge); }});
    edgeLayer.appendChild(line);
  }});
  nodes.forEach(node => {{
    const group = el('g', {{class:'node ' + node.type, transform:`translate(${{node.x}} ${{node.y}})`}});
    const circle = el('circle', {{r: node.type === 'document' ? 9 : 7}});
    const label = el('text', {{x:12, y:4}}); label.textContent = shortLabel(node);
    group.append(circle, label); group.addEventListener('click', event => {{ event.stopPropagation(); show(node); }});
    nodeLayer.appendChild(group);
  }});
  const rows = Math.ceil(Math.max(componentsCount(nodes, edges), 1) / Math.max(1, Math.ceil(Math.sqrt(Math.max(componentsCount(nodes, edges), 1) * 1.5))));
  svg.setAttribute('viewBox', `0 0 ${{Math.max(900, Math.ceil(Math.sqrt(Math.max(componentsCount(nodes, edges),1)*1.5))*220+40)}} ${{Math.max(650, rows*190+40)}}`);
  details.textContent = `当前显示：${{nodes.length}} 个节点，${{edges.length}} 条边`;
}}

function componentsCount(nodes, edges) {{
  const parent = new Map(nodes.map(n => [n.id, n.id]));
  const find = x => {{ while (parent.get(x) !== x) {{ parent.set(x, parent.get(parent.get(x))); x = parent.get(x); }} return x; }};
  edges.forEach(e => {{ const a=find(e.source_id), b=find(e.target_id); if(a!==b) parent.set(a,b); }});
  return new Set(nodes.map(n => find(n.id))).size;
}}

function applyTransform() {{ viewport.setAttribute('transform', `translate(${{transform.x}} ${{transform.y}}) scale(${{transform.scale}})`); }}
svg.addEventListener('wheel', event => {{ event.preventDefault(); const factor = event.deltaY < 0 ? 1.12 : .89; transform.scale = Math.min(5, Math.max(.2, transform.scale * factor)); applyTransform(); }});
svg.addEventListener('pointerdown', event => {{ dragging = {{x:event.clientX, y:event.clientY, tx:transform.x, ty:transform.y}}; svg.setPointerCapture(event.pointerId); }});
svg.addEventListener('pointermove', event => {{ if (!dragging) return; transform.x=dragging.tx+(event.clientX-dragging.x); transform.y=dragging.ty+(event.clientY-dragging.y); applyTransform(); }});
svg.addEventListener('pointerup', () => dragging = null);
svg.addEventListener('click', () => show({{help:'点击节点或边查看属性'}}));
document.querySelectorAll('input').forEach(input => input.addEventListener('change', render));
document.getElementById('fit').addEventListener('click', render);
document.getElementById('reset').addEventListener('click', () => {{ transform={{x:0,y:0,scale:1}}; applyTransform(); render(); }});
render();
</script>
</body>
</html>
"""


def export(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="graph-visualization-") as temp_dir:
        graph_store, _, stats = build_fixture_index(Path(temp_dir) / "graph.sqlite3")
        try:
            snapshot = graph_store.snapshot()
        finally:
            graph_store.close()
    html_path = output_dir / "graph.html"
    graphml_path = output_dir / "graph.graphml"
    html_path.write_text(render_html(snapshot, stats), encoding="utf-8")
    write_graphml(graphml_path, snapshot)
    write_csv(output_dir, snapshot)
    return {
        "nodes": len(snapshot["nodes"]),
        "edges": len(snapshot["edges"]),
        "files": [html_path, graphml_path, output_dir / "nodes.csv", output_dir / "edges.csv"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize the controlled GraphRAG fixture graph.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    args = parser.parse_args()
    result = export(args.output.resolve())
    print(f"Exported {result['nodes']} nodes and {result['edges']} edges:")
    for path in result["files"]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
