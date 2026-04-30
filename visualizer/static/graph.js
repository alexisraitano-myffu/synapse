/* ── Palette ────────────────────────────────────────────────────────────────── */
const COLORS = [
  "#4a90e2", // blue
  "#7b68ee", // purple
  "#34d399", // emerald
  "#f87171", // coral
  "#fbbf24", // amber
  "#22d3ee", // cyan
  "#c084fc", // violet
  "#fb923c", // orange
];

/* ── State ──────────────────────────────────────────────────────────────────── */
let simulation, svg, gLinks, gNodes, zoom;
let allNodes = [], allEdges = [];
let selectedId = null;

/* ── Init ───────────────────────────────────────────────────────────────────── */
async function init() {
  await loadStats();
  const [nodes, edges] = await Promise.all([
    fetch("/api/nodes").then(r => r.json()),
    fetch("/api/edges").then(r => r.json()),
  ]);
  allNodes = nodes;
  allEdges = edges;

  if (nodes.length === 0) {
    document.getElementById("empty-state").style.display = "flex";
    return;
  }
  document.getElementById("empty-state").style.display = "none";
  buildGraph(nodes, edges);
}

async function loadStats() {
  const s = await fetch("/api/stats").then(r => r.json());
  document.getElementById("stat-notes").textContent = s.notes_vectorized;
  document.getElementById("stat-notes-total").textContent = s.notes_total;
  document.getElementById("stat-edges").textContent = s.connections;
  document.getElementById("stat-inbox").textContent = s.inbox_pending;
}

/* ── Graph ──────────────────────────────────────────────────────────────────── */
function buildGraph(nodes, edges) {
  const wrap = document.getElementById("graph-wrap");
  const W = wrap.offsetWidth, H = wrap.offsetHeight;

  svg = d3.select("#graph").attr("viewBox", [0, 0, W, H]);

  // Zoom
  zoom = d3.zoom().scaleExtent([0.15, 8]).on("zoom", ({ transform }) => {
    gLinks.attr("transform", transform);
    gNodes.attr("transform", transform);
  });
  svg.call(zoom);

  // Size scale: 8–28px radius
  const maxSize = d3.max(nodes, d => d.size) || 1;
  const rScale = d3.scaleSqrt().domain([0, maxSize]).range([8, 28]);

  // Deep-copy nodes/edges so simulation can mutate them
  const simNodes = nodes.map(d => ({ ...d }));
  const nodeById = new Map(simNodes.map(d => [d.id, d]));
  const simEdges = edges
    .filter(e => nodeById.has(e.source) && nodeById.has(e.target))
    .map(e => ({ ...e }));

  // Simulation
  simulation = d3.forceSimulation(simNodes)
    .force("link", d3.forceLink(simEdges).id(d => d.id).distance(130).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-350))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide(d => rScale(d.size) + 12));

  // Draw edges
  gLinks = svg.append("g").attr("class", "links");
  const link = gLinks.selectAll("line")
    .data(simEdges)
    .join("line")
    .attr("class", "link");

  // Draw nodes
  gNodes = svg.append("g").attr("class", "nodes");
  const node = gNodes.selectAll("g")
    .data(simNodes)
    .join("g")
    .attr("class", "node")
    .call(
      d3.drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
    )
    .on("click", (event, d) => {
      event.stopPropagation();
      selectNode(d, node, link, rScale);
    })
    .on("mouseenter", (event, d) => showTooltip(event, d))
    .on("mousemove", moveTooltip)
    .on("mouseleave", hideTooltip);

  node.append("circle")
    .attr("r", d => rScale(d.size))
    .attr("fill", d => COLORS[d.group]);

  node.append("text")
    .attr("y", d => rScale(d.size) + 13)
    .text(d => d.title.length > 22 ? d.title.slice(0, 22) + "…" : d.title);

  // Tick
  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });

  // Click on background deselects
  svg.on("click", () => deselect(node, link));
}

/* ── Selection ──────────────────────────────────────────────────────────────── */
function selectNode(d, node, link, rScale) {
  if (selectedId === d.id) { deselect(node, link); return; }
  selectedId = d.id;

  const neighborIds = new Set([d.id]);
  link.each(e => {
    const sid = typeof e.source === "object" ? e.source.id : e.source;
    const tid = typeof e.target === "object" ? e.target.id : e.target;
    if (sid === d.id) neighborIds.add(tid);
    if (tid === d.id) neighborIds.add(sid);
  });

  node
    .classed("selected", n => n.id === d.id)
    .classed("dimmed", n => !neighborIds.has(n.id));

  link
    .classed("highlighted", e => {
      const sid = typeof e.source === "object" ? e.source.id : e.source;
      const tid = typeof e.target === "object" ? e.target.id : e.target;
      return sid === d.id || tid === d.id;
    })
    .classed("dimmed", e => {
      const sid = typeof e.source === "object" ? e.source.id : e.source;
      const tid = typeof e.target === "object" ? e.target.id : e.target;
      return sid !== d.id && tid !== d.id;
    });

  openPanel(d.id);
}

function deselect(node, link) {
  selectedId = null;
  node.classed("selected", false).classed("dimmed", false);
  link.classed("highlighted", false).classed("dimmed", false);
  closePanel();
}

/* ── Side Panel ─────────────────────────────────────────────────────────────── */
async function openPanel(id) {
  const panel = document.getElementById("panel");
  document.getElementById("panel-content").innerHTML =
    '<p style="color:var(--text-dim);font-size:12px">Chargement…</p>';
  panel.classList.add("open");

  const note = await fetch(`/api/note/${id}`).then(r => r.json());

  document.getElementById("panel-title").textContent = note.title || "(untitled)";
  document.getElementById("panel-date").textContent =
    note.created_at ? new Date(note.created_at).toLocaleDateString("fr-FR", { day: "numeric", month: "long", year: "numeric" }) : "";
  document.getElementById("panel-sources").textContent =
    `${(note.source_ids || []).length} capture${(note.source_ids||[]).length !== 1 ? "s" : ""} d'origine`;

  document.getElementById("panel-content").innerHTML =
    marked.parse(note.content || "*(contenu vide)*");
}

function closePanel() {
  document.getElementById("panel").classList.remove("open");
}

/* ── Tooltip ────────────────────────────────────────────────────────────────── */
const tooltip = document.getElementById("tooltip");

function showTooltip(event, d) {
  tooltip.style.opacity = "1";
  tooltip.querySelector(".tip-title").textContent = d.title;
  tooltip.querySelector(".tip-summary").textContent = d.summary || "";
  moveTooltip(event);
}
function moveTooltip(event) {
  const x = event.clientX + 14, y = event.clientY - 10;
  tooltip.style.left = (x + 270 > window.innerWidth ? x - 284 : x) + "px";
  tooltip.style.top  = y + "px";
}
function hideTooltip() { tooltip.style.opacity = "0"; }

/* ── Search ─────────────────────────────────────────────────────────────────── */
document.getElementById("search").addEventListener("input", function () {
  const q = this.value.trim().toLowerCase();
  if (!gNodes) return;

  gNodes.selectAll("g.node").each(function (d) {
    const match = !q || d.title.toLowerCase().includes(q) || d.summary.toLowerCase().includes(q);
    d3.select(this)
      .classed("dimmed", !match)
      .classed("selected", false);
  });
  if (gLinks) gLinks.selectAll("line.link").classed("dimmed", !!q).classed("highlighted", false);
  if (!q) deselect(gNodes.selectAll("g.node"), gLinks?.selectAll("line.link"));
});

/* ── Resize ─────────────────────────────────────────────────────────────────── */
window.addEventListener("resize", () => {
  if (!svg) return;
  const wrap = document.getElementById("graph-wrap");
  const W = wrap.offsetWidth, H = wrap.offsetHeight;
  svg.attr("viewBox", [0, 0, W, H]);
  if (simulation) simulation.force("center", d3.forceCenter(W / 2, H / 2)).alpha(0.3).restart();
});

document.getElementById("panel-close").addEventListener("click", () => {
  closePanel();
  if (gNodes) gNodes.selectAll("g.node").classed("selected", false).classed("dimmed", false);
  if (gLinks) gLinks.selectAll("line.link").classed("highlighted", false).classed("dimmed", false);
  selectedId = null;
});

/* ── Start ───────────────────────────────────────────────────────────────────── */
init();
