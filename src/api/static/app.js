// FinQA Engine demo: settings in place of the YAML, the workflow drawn as it runs.

const $ = (sel) => document.querySelector(sel);
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  mode: "live",
  options: null,
  settings: null,
  examples: [],
  example: null,
  filter: "calc",
  run: null,
  pinned: null,
  controller: null,
};

// --- diagram geometry -----------------------------------------------------------------

const ROW_Y = 112;
const NODE_W = 150;
const NODE_H = 100;
const NODES = [
  { id: "question", x: 10, y: ROW_Y, title: "Question", sub: () => clip(docLabel() || "tous les rapports", 19) },
  { id: "dense", x: 176, y: ROW_Y, title: "Recherche dense", sub: (s) => `BGE-M3, top ${s.reranker ? s.prefetch : s.k}` },
  { id: "rerank", x: 342, y: ROW_Y, title: "Reranker", sub: (s) => `garde le top ${s.k}` },
  { id: "grade", x: 508, y: ROW_Y, title: "Grader LLM", sub: (s) => `garde la note ≥ ${s.keep_threshold}` },
  { id: "expand", x: 674, y: ROW_Y, title: "Expansion", sub: (s) => `±${s.window} chunk voisin` },
  { id: "route", x: 840, y: ROW_Y, title: "Routage", sub: () => "une formule ?" },
  { id: "calculate", x: 1000, y: 6, title: "Calculatrice", sub: () => "vérifiée par le code" },
  { id: "generate", x: 1150, y: ROW_Y, title: "Génération", sub: (s) => clip(s.model, 19) },
];
const MID = ROW_Y + NODE_H / 2;
const EDGES = [
  { id: "question-dense", d: `M160 ${MID} H176` },
  { id: "dense-rerank", d: `M326 ${MID} H342` },
  { id: "rerank-grade", d: `M492 ${MID} H508` },
  { id: "grade-expand", d: `M658 ${MID} H674` },
  { id: "expand-route", d: `M824 ${MID} H840` },
  { id: "route-generate", d: `M990 ${MID} H1150` },
  { id: "route-calculate", d: `M915 ${ROW_Y} C915 70 950 56 1000 56` },
  { id: "calculate-generate", d: `M1150 56 C1200 56 1225 80 1225 ${ROW_Y}` },
];
const CHAIN = ["question", "dense", "rerank", "grade", "expand", "route"];
const TIME_Y = 254;

// Graph node (as the API names it) -> the diagram nodes that draw it.
const DRAWN = {
  retrieve: (s) => (s.reranker ? ["dense", "rerank"] : ["dense"]),
  grade: () => ["grade"],
  expand: () => ["expand"],
  route: () => ["route"],
  calculate: () => ["calculate"],
  generate: () => ["generate"],
};

const NODE_HELP = {
  question: "La question de l'utilisateur, et le rapport auquel la recherche est limitée quand il est choisi.",
  dense: "La question est encodée par BGE-M3 ; Qdrant renvoie les passages les plus proches (similarité cosinus) dans le rapport choisi.",
  rerank: "Un cross-encoder relit chaque paire question-passage et reclasse la présélection ; seuls les k meilleurs passent.",
  grade: "Un appel LLM par passage note sa pertinence de 0 à 3. Les passages au-dessus du seuil sont gardés ; le plancher complète par rang si la sélection est trop mince.",
  expand: "Chaque passage gardé est relu avec les chunks qui l'entourent dans son rapport : un tableau coupé entre deux chunks arrive entier. Aucun appel LLM.",
  route: "Le code lit la forme de la question : si elle donne une formule et demande une valeur, la calculatrice est appelée.",
  calculate: "Le modèle écrit la formule et désigne les lignes des tableaux ; il ne recopie aucun chiffre. Le code lit chaque valeur, vérifie l'état financier, l'échelle et l'année, puis calcule en décimal exact. Au moindre doute, il refuse.",
  generate: "Le contexte est coupé par la fin pour tenir dans la fenêtre, puis le modèle répond en citant ses sources. Un chiffre vérifié lui est transmis.",
};

const OUTCOMES = {
  good_job: "Correcte, avec la preuve dans le contexte",
  unverified: "Correcte, mais la page de preuve n'était pas dans le contexte",
  need_help: "Fausse alors que la preuve était dans le contexte",
  hallucinating: "Fausse, et la preuve n'était pas dans le contexte",
  dont_know: "Le modèle a refusé de répondre",
};

// A ladder: each preset adds one stage to the one before it.
const PRESETS = {
  naive: { reranker: false, grading: false, expansion: false, calculator: false },
  rerank: { reranker: true, grading: false, expansion: false, calculator: false },
  grade_expand: { reranker: true, grading: true, keep_threshold: 2, min_chunks: 3, expansion: true, window: 1, calculator: false },
  advanced: { reranker: true, grading: true, keep_threshold: 2, min_chunks: 3, expansion: true, window: 1, calculator: true },
};

// --- small helpers --------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const seconds = (s) => (s == null ? "" : s < 1 ? `${s.toFixed(2)} s` : s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`);
const thousands = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n));
const clip = (text, n) => (text.length > n ? `${text.slice(0, n - 1)}…` : text);
const plural = (n, word) => `${n} ${word}${n > 1 ? "s" : ""}`;
// replaceChildren writes a null as the text "null": drop the absent parts first.
const put = (node, ...children) => node.replaceChildren(...children.filter((c) => c != null));
const snippet = (text, n = 140) => text.replace(/\s+/g, " ").trim().slice(0, n);

function docLabel() {
  const value = $("#doc")?.value;
  return value ? value.replaceAll("_", " ") : "";
}

// Answers come back as Markdown with some LaTeX; render the common parts, escape the rest.
function renderAnswer(text) {
  let t = escapeHtml(text)
    .replace(/\\\[|\\\]|\\\(|\\\)/g, "")
    .replace(/\\text\{([^}]*)\}/g, "$1")
    .replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, "($1) / ($2)")
    .replace(/\\times/g, "×").replace(/\\approx/g, "≈").replace(/\\%/g, "%").replace(/\\,/g, " ")
    .replace(/\\boxed\{([^}]*)\}/g, "<strong>$1</strong>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return t
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
    .join("");
}

// --- settings rail --------------------------------------------------------------------

function segmented(container, values, label, onPick) {
  put(container,
    ...values.map((v) => {
      const b = el("button", { type: "button", "data-value": v, "aria-pressed": "false", text: label(v) });
      b.addEventListener("click", () => onPick(v));
      return b;
    }),
  );
}

function setPressed(container, value) {
  for (const b of container.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.value == value));
}

function buildRail() {
  const o = state.options;
  segmented($("#chunk_size"), [256, 512, 1024], (v) => `${v}`, (v) => update({ chunk_size: v }));
  for (const b of $("#chunk_size").querySelectorAll("button")) {
    b.disabled = !o.chunk_sizes.includes(Number(b.dataset.value));
    b.title = b.disabled ? "Collection non indexée" : `Chunks de ${b.dataset.value} tokens`;
  }
  segmented($("#keep_threshold"), [0, 1, 2, 3], (v) => `≥ ${v}`, (v) => update({ keep_threshold: v }));
  segmented($("#num_ctx"), [8192, 12288, 16384, 24576], (v) => `${v / 1024}K`, (v) => update({ num_ctx: v }));
  segmented($("#max_tokens"), [512, 1024, 2048], (v) => `${v} tokens`, (v) => update({ max_tokens: v }));
  const models = o.models.length ? o.models : [o.defaults.model];
  put($("#model"), ...models.map((m) => el("option", { value: m, text: m })));

  for (const id of ["reranker", "grading", "expansion", "calculator"]) {
    $(`#${id}`).addEventListener("change", (e) => update({ [id]: e.target.checked }));
  }
  for (const id of ["prefetch", "k", "min_chunks", "window"]) {
    $(`#${id}`).addEventListener("input", (e) => update({ [id]: Number(e.target.value) }));
  }
  $("#model").addEventListener("change", (e) => update({ model: e.target.value }));
  for (const b of document.querySelectorAll(".presets button")) {
    b.addEventListener("click", () => update(PRESETS[b.dataset.preset]));
  }
}

function update(patch) {
  if (state.mode !== "live") return;
  state.settings = { ...state.settings, ...patch };
  if (state.settings.k > state.settings.prefetch) state.settings.k = state.settings.prefetch;
  renderRail();
  if (!state.run || state.run.finished) drawIdle();
}

function renderRail() {
  const s = activeSettings();
  setPressed($("#chunk_size"), s.chunk_size);
  setPressed($("#keep_threshold"), s.keep_threshold);
  setPressed($("#num_ctx"), s.num_ctx);
  setPressed($("#max_tokens"), s.max_tokens ?? 1024);
  for (const id of ["reranker", "grading", "expansion", "calculator"]) $(`#${id}`).checked = s[id];
  for (const id of ["prefetch", "k", "min_chunks", "window"]) {
    $(`#${id}`).value = s[id];
    $(`#${id}-out`).textContent = id === "window" ? `±${s[id]}` : s[id];
  }
  $("#model").value = s.model;
  $("#g-grading").classList.toggle("off", !s.grading);
  $("#g-expansion").classList.toggle("off", !s.expansion);
  $("#prefetch").closest(".field").style.opacity = s.reranker ? "" : "0.4";
  for (const b of document.querySelectorAll(".presets button")) {
    const preset = PRESETS[b.dataset.preset];
    b.setAttribute("aria-pressed", String(Object.entries(preset).every(([k, v]) => s[k] === v)));
  }
  const locked = state.mode === "replay";
  $(".rail").classList.toggle("locked", locked);
  $("#rail-note").textContent = locked
    ? `Réglages du run enregistré : ${state.options.replay?.label ?? ""}.`
    : "Chaque réglage remplace une ligne du fichier de configuration.";
}

function activeSettings() {
  return state.mode === "replay" && state.options.replay ? state.options.replay.settings : state.settings;
}

// --- the diagram ----------------------------------------------------------------------

const nodeEls = {};
const edgeEls = {};
let timeLayer;

function buildDiagram() {
  const root = $("#diagram");
  put(root);
  for (const e of EDGES) {
    const path = svg("path", { d: e.d, class: "edge" });
    edgeEls[e.id] = path;
    root.append(path);
  }
  for (const n of NODES) {
    const g = svg("g", { class: "node", tabindex: "0", role: "button", "data-node": n.id });
    g.append(svg("rect", { class: "box", x: n.x, y: n.y, width: NODE_W, height: NODE_H, rx: 12 }));
    const title = svg("text", { class: "title", x: n.x + 14, y: n.y + 28 });
    title.textContent = n.title;
    const sub = svg("text", { class: "sub", x: n.x + 14, y: n.y + 47 });
    const stat = svg("text", { class: "stat", x: n.x + 14, y: n.y + 70 });
    g.append(title, sub, stat);
    if (n.id === "grade") g.append(svg("g", { class: "cells" }));
    g.addEventListener("click", () => select(n.id, true));
    g.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        select(n.id, true);
      }
    });
    nodeEls[n.id] = { g, sub, stat, def: n };
    root.append(g);
  }
  timeLayer = svg("g", { class: "timeline" });
  root.append(timeLayer);
}

function drawCells(k) {
  const g = nodeEls.grade.g.querySelector(".cells");
  put(g);
  const { x, y } = nodeEls.grade.def;
  const inner = NODE_W - 28;
  const gap = k > 24 ? 1 : 2;
  const w = (inner - gap * (k - 1)) / k;
  for (let i = 0; i < k; i++) {
    g.append(svg("rect", { class: "cell", x: x + 14 + i * (w + gap), y: y + 80, width: Math.max(w, 1), height: 10, rx: 1.5, "data-i": i }));
  }
}

function offNodes(s) {
  const off = new Set();
  if (!s.reranker) off.add("rerank");
  if (!s.grading) off.add("grade");
  if (!s.expansion) off.add("expand");
  if (!s.calculator) {
    off.add("route");
    off.add("calculate");
  }
  return off;
}

function setNode(id, status, stat) {
  const n = nodeEls[id];
  n.g.classList.remove("idle", "off", "running", "done", "skipped", "ok", "refused");
  for (const c of status.split(" ")) if (c) n.g.classList.add(c);
  if (stat !== undefined) n.stat.textContent = stat;
}

function status(id) {
  const c = nodeEls[id].g.classList;
  for (const s of ["running", "done", "off", "skipped"]) if (c.contains(s)) return s;
  return "idle";
}

function drawIdle() {
  const s = activeSettings();
  const off = offNodes(s);
  for (const n of NODES) {
    nodeEls[n.id].sub.textContent = n.sub(s);
    setNode(n.id, off.has(n.id) ? "off" : "idle", off.has(n.id) ? "désactivé" : "");
  }
  nodeEls.generate.stat.textContent = `fenêtre ${Math.round(s.num_ctx / 1024)}K`;
  drawCells(s.k);
  drawEdges();
  put(timeLayer);
}

// Main-chain edges follow the furthest node done and the node running: an edge into a
// disabled node is passed through like any other. The calculator branch is drawn apart.
function drawEdges() {
  let reached = -1;
  let running = -1;
  CHAIN.forEach((id, i) => {
    if (status(id) === "done") reached = i;
    if (status(id) === "running") running = i;
  });
  const genStatus = status("generate");
  const calcStatus = status("calculate");
  const calcRan = calcStatus === "done";
  if (genStatus === "running" || genStatus === "done" || calcStatus === "running" || calcRan) reached = CHAIN.length - 1;

  for (let i = 0; i < CHAIN.length - 1; i++) {
    const edge = edgeEls[`${CHAIN[i]}-${CHAIN[i + 1]}`];
    edge.classList.remove("passed", "flow", "dim");
    if (i + 1 <= reached) edge.classList.add("passed");
    else if (running > -1 && i + 1 <= running) edge.classList.add("flow");
  }
  const set = (id, cls) => {
    edgeEls[id].classList.remove("passed", "flow", "dim");
    if (cls) edgeEls[id].classList.add(cls);
  };
  set("route-calculate", calcRan ? "passed" : calcStatus === "running" ? "flow" : calcStatus === "off" ? "dim" : "");
  set("calculate-generate", calcRan && genStatus === "done" ? "passed" : calcRan && genStatus === "running" ? "flow" : calcStatus === "off" ? "dim" : "");
  set("route-generate", calcRan ? "dim" : genStatus === "done" ? "passed" : genStatus === "running" ? "flow" : "");
}

function drawTimeline(latencies) {
  put(timeLayer);
  const order = [
    ["retrieve", "Recherche"],
    ["grade", "Grader"],
    ["expand", "Expansion"],
    ["calculate", "Calculatrice"],
    ["generate", "Génération"],
  ];
  const parts = order.map(([id, label]) => [id, label, latencies[id] || 0]).filter(([, , s]) => s > 0.01);
  const total = parts.reduce((a, [, , s]) => a + s, 0);
  if (!total) return;
  const x0 = 10;
  const width = 1290;
  const label = svg("text", { x: x0, y: TIME_Y - 8, class: "sub" });
  label.textContent = `Temps passé par étape, ${seconds(total)} au total`;
  timeLayer.append(label);
  let x = x0;
  const shades = { retrieve: "#9fb3c8", grade: "#13233f", expand: "#7b8aa0", calculate: "#0e7a5a", generate: "#4f6e93" };
  for (const [id, name, s] of parts) {
    const w = Math.max((s / total) * width, 2);
    timeLayer.append(svg("rect", { x, y: TIME_Y, width: w - 2, height: 22, rx: 4, fill: shades[id] }));
    if (w > 110) {
      const t = svg("text", { x: x + 8, y: TIME_Y + 15, fill: "#fff", "font-size": 12, "font-weight": 600 });
      t.textContent = `${name} ${seconds(s)}`;
      timeLayer.append(t);
    }
    x += w;
  }
}

// --- a run ----------------------------------------------------------------------------

function newRun(settings) {
  return {
    settings,
    plan: [],
    passages: new Map(),
    retrieved: [],
    dense: [],
    grades: [],
    kept: new Set(),
    floor: new Set(),
    widened: [],
    neighbours: new Set(),
    context: [],
    isNumeric: null,
    calc: null,
    answer: null,
    promptTokens: null,
    dropped: 0,
    latencies: {},
    nodeSeconds: {},
    started: performance.now(),
    runningSince: performance.now(),
    running: [],
    finished: false,
    done: null,
  };
}

function startClock() {
  const clock = $("#clock");
  clock.classList.add("running");
  const tick = () => {
    const run = state.run;
    if (!run || run.finished) return;
    const total = (performance.now() - run.started) / 1000;
    clock.textContent = `${state.mode === "replay" ? "Rejeu en cours" : "En cours"}, ${seconds(total)}`;
    for (const id of run.running) {
      if (id === "grade" && run.grades.length) continue;
      nodeEls[id].stat.textContent = `en cours, ${seconds((performance.now() - run.runningSince) / 1000)}`;
    }
    requestAnimationFrame(() => setTimeout(tick, 100));
  };
  tick();
}

function markRunning(graphNode) {
  const run = state.run;
  // Retrieval runs its two stages in turn: the reranker lights up once the dense
  // shortlist is reported.
  run.running = graphNode === "retrieve" ? ["dense"] : DRAWN[graphNode](run.settings);
  run.runningSince = performance.now();
  for (const id of run.running) setNode(id, "running", "en cours");
  if (!state.pinned) select(run.running[run.running.length - 1]);
  drawEdges();
}

function nextAfter(graphNode) {
  const run = state.run;
  if (graphNode === "route") return run.isNumeric ? "calculate" : "generate";
  const i = run.plan.indexOf(graphNode);
  const next = run.plan[i + 1];
  return next === "calculate" ? "generate" : next;
}

function onEvent(ev) {
  const run = state.run;
  if (ev.type === "plan") {
    run.plan = ev.nodes;
    run.settings = ev.settings;
    setNode("question", "done", clip($("#question").value, 20));
    markRunning("retrieve");
    return;
  }
  if (ev.type === "progress" && ev.stage === "dense") {
    remember(ev.passages);
    run.dense = ev.passages.map((p) => p.id);
    ev.passages.forEach((p, i) => {
      const known = run.passages.get(p.id);
      known.denseRank = i + 1;
      known.denseScore = p.score;
    });
    setNode("dense", "done", `${ev.passages.length} candidats`);
    run.running = ["rerank"];
    run.runningSince = performance.now();
    setNode("rerank", "running", "en cours");
    if (!state.pinned || state.pinned === "dense") renderInspector();
    if (!state.pinned) select("rerank");
    drawEdges();
    return;
  }
  if (ev.type === "progress" && ev.node === "grade") {
    run.grades[ev.index] = ev.grade;
    const cell = nodeEls.grade.g.querySelector(`.cell[data-i="${ev.index}"]`);
    if (cell) cell.setAttribute("class", `cell on g${ev.grade}`);
    const notes = run.grades.filter((g) => g != null).length;
    nodeEls.grade.stat.textContent = `${notes} / ${run.retrieved.length} notés`;
    if (state.pinned === "grade" || !state.pinned) renderInspector();
    return;
  }
  if (ev.type === "node") return onNode(ev);
  if (ev.type === "done") return onDone(ev);
  if (ev.type === "error") return onError(ev.message);
  if (ev.type === "cancelled") return onError("Run interrompu : un autre run a démarré, ou il a été arrêté.");
}

function remember(passages) {
  for (const p of passages) if (!state.run.passages.has(p.id)) state.run.passages.set(p.id, p);
}

function onNode(ev) {
  const run = state.run;
  const d = ev.data;
  if (ev.seconds != null) run.nodeSeconds[ev.node] = ev.seconds;

  if (ev.node === "retrieve") {
    remember(d.passages);
    run.retrieved = d.passages.map((p) => p.id);
    d.passages.forEach((p, i) => {
      const known = run.passages.get(p.id);
      known.rank = i + 1;
      known.score = p.score;
    });
    if (!run.dense.length) {
      setNode("dense", "done", run.settings.reranker ? `${run.settings.prefetch} candidats` : `${d.passages.length} passages`);
    }
    if (run.settings.reranker) {
      const climbed = d.passages.filter((p, i) => (run.passages.get(p.id).denseRank ?? 0) > d.passages.length).length;
      setNode("rerank", "done", climbed ? `${plural(climbed, "remonté")} du dense` : `${d.passages.length} passages gardés`);
    }
    drawCells(d.passages.length);
  } else if (ev.node === "grade") {
    run.grades = d.grades;
    run.kept = new Set(d.kept);
    const byGrade = new Set(run.retrieved.filter((id, i) => d.grades[i] >= run.settings.keep_threshold));
    run.floor = new Set(d.kept.filter((id) => !byGrade.has(id)));
    d.grades.forEach((g, i) => {
      const cell = nodeEls.grade.g.querySelector(`.cell[data-i="${i}"]`);
      if (cell) cell.setAttribute("class", `cell on g${g}${run.kept.has(run.retrieved[i]) ? " kept" : ""}`);
    });
    setNode("grade", "done", `${plural(d.kept.length, "gardé")} sur ${d.grades.length}`);
  } else if (ev.node === "expand") {
    remember(d.added);
    run.widened = d.ids;
    const anchors = new Set(run.kept.size ? run.kept : run.retrieved);
    run.neighbours = new Set(d.ids.filter((id) => !anchors.has(id)));
    setNode("expand", "done", `+${plural(run.neighbours.size, "voisin")}`);
  } else if (ev.node === "route") {
    run.isNumeric = d.is_numeric;
    setNode("route", "done", d.is_numeric ? "formule détectée" : "pas de formule");
    if (!d.is_numeric) setNode("calculate", "skipped", "non appelée");
  } else if (ev.node === "calculate") {
    run.calc = d;
    if (d.computed) setNode("calculate", "done ok", `vérifié : ${d.computed}`);
    else setNode("calculate", "done refused", "refus, réponse libre");
    renderVerified();
  } else if (ev.node === "generate") {
    remember(d.added);
    run.context = d.sources;
    run.answer = d.answer;
    run.promptTokens = d.prompt_tokens;
    run.dropped = d.n_dropped_to_fit;
    run.truncated = d.truncated;
    if (d.truncated) setNode("generate", "done refused", `coupée à ${run.settings.max_tokens ?? 1024} tokens`);
    else setNode("generate", "done", `${thousands(d.prompt_tokens)} tokens lus`);
    renderAnswer_();
  }
  run.running = [];
  const next = ev.node === "generate" ? null : nextAfter(ev.node);
  if (next) markRunning(next);
  drawEdges();
  renderSources();
  if (!state.pinned || DRAWN[ev.node](run.settings).includes(state.pinned)) renderInspector();
}

function onDone(ev) {
  const run = state.run;
  run.finished = true;
  run.done = ev;
  run.latencies = ev.node_latencies || {};
  const clock = $("#clock");
  clock.classList.remove("running");
  clock.textContent =
    state.mode === "replay"
      ? `Run enregistré, ${seconds(ev.latency_s)} de calcul à l'origine`
      : `Terminé en ${seconds(ev.latency_s)}`;
  drawTimeline(run.latencies);
  renderMetrics();
  setRunning(false);
  if (!state.pinned) select("generate");
}

function onError(message) {
  const banner = $("#banner");
  banner.hidden = false;
  banner.textContent = message;
  if (state.run) {
    state.run.finished = true;
    for (const id of state.run.running) setNode(id, "idle", "interrompu");
  }
  $("#clock").classList.remove("running");
  setRunning(false);
}

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (line) yield JSON.parse(line.slice(6));
    }
  }
}

function setRunning(on) {
  const button = $("#run");
  button.textContent = on ? "Arrêter" : state.mode === "replay" ? "Rejouer" : "Lancer";
  button.classList.toggle("stop", on);
  updateRunButton();
}

function updateRunButton() {
  const button = $("#run");
  const running = state.run && !state.run.finished;
  if (running) {
    button.disabled = false;
    return;
  }
  if (state.mode === "replay") {
    button.disabled = !(state.example && state.example.recorded);
    button.title = button.disabled ? "Choisissez une question FinanceBench enregistrée" : "";
  } else {
    button.disabled = !$("#question").value.trim();
    button.title = "";
  }
}

async function run() {
  if (state.run && !state.run.finished) {
    state.controller?.abort();
    onError("Run arrêté.");
    return;
  }
  const question = $("#question").value.trim();
  const example = state.example && state.example.question === question ? state.example : null;
  const settings = { ...activeSettings() };
  $("#banner").hidden = true;
  state.pinned = null;
  state.run = newRun(settings);
  state.run.example = example;
  drawIdle();
  clearPanes();
  setRunning(true);
  startClock();
  state.controller = new AbortController();
  try {
    const response =
      state.mode === "replay"
        ? await fetch(`/demo/replay/${encodeURIComponent(example.id)}`, { signal: state.controller.signal })
        : await fetch("/demo/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, doc_id: $("#doc").value || null, settings }),
            signal: state.controller.signal,
          });
    if (!response.ok) throw new Error(`${response.status} : ${await response.text()}`);
    for await (const ev of readEvents(response)) onEvent(ev);
    if (!state.run.finished) onError("La connexion s'est terminée avant la fin du run.");
  } catch (err) {
    if (err.name !== "AbortError") onError(`Le run a échoué : ${err.message}`);
  }
}

// --- panes ----------------------------------------------------------------------------

function clearPanes() {
  $("#answer-text").innerHTML = "";
  $("#answer-text").classList.add("pending");
  $("#answer-text").textContent = "Le modèle répond à la fin du workflow.";
  $("#verified").hidden = true;
  $("#truncated").hidden = true;
  $("#metrics").hidden = true;
  renderGold();
  renderSources();
  renderInspector();
}

function renderVerified() {
  const run = state.run;
  const box = $("#verified");
  if (!run?.calc) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.classList.toggle("refused", !run.calc.computed);
  put(box,
    ...(run.calc.computed
      ? [el("strong", { text: run.calc.computed }), el("span", { text: "Calculé et vérifié par le code, transmis au générateur." })]
      : [el("span", { text: `Calculatrice : refus. ${run.calc.calc_error ?? ""} Le modèle répond sans chiffre vérifié.` })]),
  );
}

function renderAnswer_() {
  const run = state.run;
  const box = $("#answer-text");
  box.classList.remove("pending");
  box.innerHTML = renderAnswer(run.answer || "");
  const notice = $("#truncated");
  notice.hidden = !run.truncated;
  notice.textContent = run.truncated
    ? `Réponse coupée : le modèle a atteint la limite de ${run.settings.max_tokens ?? 1024} tokens de sortie avant de conclure. Le texte ci-dessous est son raisonnement inachevé ; augmentez « Longueur maximale de la réponse » ou choisissez un modèle plus concis.`
    : "";
  $("#answer-sub").textContent = `${run.settings.model}, à partir de ${run.context.length} passages.`;
}

function renderGold() {
  const box = $("#gold");
  const ex = state.run?.example ?? (state.example && state.example.question === $("#question").value.trim() ? state.example : null);
  if (!ex) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const outcome = state.run?.done?.outcome;
  put(box,
    el("h3", { text: "Réponse attendue (FinanceBench)" }),
    el("div", { class: "value", text: ex.gold_answer }),
    ex.justification ? el("details", {}, el("summary", { text: "Justification de la réponse attendue" }), el("p", { text: ex.justification })) : null,
    outcome ? el("span", { class: `outcome ${outcome}`, text: `Jugement : ${OUTCOMES[outcome] ?? outcome}` }) : null,
  );
}

function renderMetrics() {
  const run = state.run;
  const d = run.done;
  const box = $("#metrics");
  box.hidden = false;
  const item = (label, value) => el("div", {}, el("dt", { text: label }), el("dd", { text: value }));
  put(box,
    item(state.mode === "replay" ? "Durée enregistrée" : "Durée", seconds(d.latency_s)),
    item("Appels LLM", String(d.llm_calls)),
    item("Tokens lus", run.promptTokens != null ? thousands(run.promptTokens) : "—"),
  );
  renderGold();
}

function passageTags(id) {
  const run = state.run;
  const tags = [];
  const i = run.retrieved.indexOf(id);
  if (i >= 0 && run.grades[i] != null) tags.push(el("span", { class: "tag", text: `note ${run.grades[i]}` }));
  if (run.floor.has(id)) tags.push(el("span", { class: "tag floor", text: "plancher" }));
  else if (run.kept.has(id)) tags.push(el("span", { class: "tag kept", text: "gardé" }));
  if (run.neighbours.has(id)) tags.push(el("span", { class: "tag neighbour", text: "voisin" }));
  return tags;
}

function renderSources() {
  const list = $("#sources-list");
  const run = state.run;
  if (!run || !run.context.length) {
    put(list, el("p", { class: "empty", text: run ? "Le contexte final apparaît après la génération." : "Lancez une question pour voir le contexte." }));
    $("#sources-sub").textContent = "Les passages retenus, dans l'ordre où le modèle les lit.";
    return;
  }
  const inContext = new Set(run.context);
  const cut = (run.widened.length ? run.widened : run.kept.size ? [...run.kept] : run.retrieved).filter((id) => !inContext.has(id));
  $("#sources-sub").textContent =
    `${run.context.length} passages lus` + (cut.length ? `, ${cut.length} coupés pour tenir dans ${Math.round(run.settings.num_ctx / 1024)}K tokens.` : ".");
  const row = (id, n, isCut) => {
    const p = run.passages.get(id);
    if (!p) return null;
    const tags = passageTags(id);
    if (isCut) tags.push(el("span", { class: "tag cut", text: "coupé" }));
    return el(
      "details",
      { class: "source" },
      el(
        "summary",
        {},
        el("span", { class: "n", text: String(n) }),
        el("span", { class: "where", title: p.doc_id, text: `p. ${p.page}` }),
        el("span", { class: "snip", text: snippet(p.text) }),
        el("span", { class: "tags" }, ...tags),
      ),
      el("div", { class: "body", text: p.text }),
    );
  };
  put(list, ...run.context.map((id, i) => row(id, i + 1, false)), ...cut.map((id) => row(id, "—", true)));
}

function select(id, byUser = false) {
  if (byUser) state.pinned = id;
  state.selected = id;
  for (const [nid, n] of Object.entries(nodeEls)) n.g.classList.toggle("selected", nid === id);
  renderInspector();
}

function plist(ids, opts = {}) {
  const run = state.run;
  return el(
    "ul",
    { class: "plist" },
    ...ids.map((id, n) => {
      const p = run.passages.get(id);
      if (!p) return null;
      const i = run.retrieved.indexOf(id);
      const grade = i >= 0 ? run.grades[i] : null;
      const out = (opts.dimUnkept && run.kept.size && !run.kept.has(id)) || (opts.dimNotRetrieved && !run.retrieved.includes(id));
      const score = opts.score === "dense" ? p.denseScore : p.score;
      const side = opts.tags
        ? passageTags(id)
        : [
            opts.showDenseRank && p.denseRank != null
              ? el("span", { class: `tag${p.denseRank > run.retrieved.length ? " climb" : ""}`, title: "Rang dans la recherche dense", text: `dense ${p.denseRank}ᵉ` })
              : null,
            score != null ? el("span", { class: "tag", title: opts.score === "dense" ? "Similarité cosinus" : "Score du cross-encoder", text: score.toFixed(2) }) : null,
          ];
      return el(
        "li",
        { class: out ? "out" : "" },
        el("span", { class: "rank", text: String(opts.rankFromOrder ? n + 1 : (opts.score === "dense" ? p.denseRank : p.rank) ?? "") }),
        el("i", { class: `g g${grade ?? 0}`, style: grade == null ? "opacity:.25" : "" }),
        el("span", { class: "where", title: p.doc_id, text: `p. ${p.page}` }),
        el("span", { class: "snip", text: snippet(p.text, 160) }),
        el("span", { class: "side" }, ...side.filter((c) => c != null)),
      );
    }),
  );
}

function renderInspector() {
  const id = state.selected;
  const body = $("#inspector-body");
  const run = state.run;
  if (!id) {
    $("#inspector-title").textContent = "Inspecteur";
    $("#inspector-sub").textContent = "Cliquez un nœud du diagramme pour voir ce qu'il a fait.";
    put(body);
    return;
  }
  const def = nodeEls[id].def;
  const s = run?.settings ?? activeSettings();
  $("#inspector-title").textContent = def.title;
  $("#inspector-sub").textContent = def.sub(s);
  const parts = [el("p", { class: "explain", text: NODE_HELP[id] })];
  const st = status(id);
  if (st === "off") parts.push(el("p", { class: "explain", text: "Ce nœud est désactivé dans les réglages : le flux passe directement au suivant." }));

  if (run) {
    if (id === "question") {
      parts.push(el("dl", { class: "kv" }, el("dt", { text: "Question" }), el("dd", { text: $("#question").value }), el("dt", { text: "Rapport" }), el("dd", { text: docLabel() || "Tous les rapports indexés" })));
    }
    if (id === "dense" && s.reranker && run.dense.length) {
      const kept = run.retrieved.length;
      parts.push(el("p", { class: "explain", text: kept ? `${run.dense.length} candidats classés par similarité cosinus. Les ${kept} que le reranker a retenus sont en clair, les autres estompés.` : `${run.dense.length} candidats classés par similarité cosinus, transmis au reranker.` }));
      parts.push(plist(run.dense, { score: "dense", dimNotRetrieved: kept > 0 }));
    } else if (id === "dense" && s.reranker && run.retrieved.length) {
      parts.push(el("p", { class: "explain", text: "Les candidats de la recherche dense ne sont pas enregistrés pour ce run ; lancez la question en direct pour les voir." }));
    } else if (id === "dense" && run.retrieved.length) {
      parts.push(el("p", { class: "explain", text: `Sans reranker, les ${run.retrieved.length} premiers passages par similarité cosinus sont transmis tels quels.` }));
      parts.push(plist(run.retrieved));
    }
    if (id === "rerank" && run.retrieved.length) {
      const climbed = run.retrieved.filter((pid) => (run.passages.get(pid).denseRank ?? 0) > run.retrieved.length).length;
      const text = run.dense.length
        ? `Le cross-encoder a relu les ${run.dense.length} candidats et en garde ${run.retrieved.length}.${climbed ? ` ${climbed} d'entre eux étaient hors du top ${run.retrieved.length} de la recherche dense : sans reranker, ils n'auraient pas été transmis.` : ""}`
        : `${run.retrieved.length} passages retenus par le cross-encoder parmi ${s.prefetch} candidats.`;
      parts.push(el("p", { class: "explain", text }));
      parts.push(plist(run.retrieved, { showDenseRank: run.dense.length > 0 }));
    }
    if (id === "grade" && run.retrieved.length && run.grades.length) {
      const kept = run.kept.size;
      parts.push(el("p", { class: "explain", text: kept ? `${kept} passages gardés sur ${run.retrieved.length}${run.floor.size ? `, dont ${run.floor.size} ajoutés par le plancher` : ""}.` : `${run.grades.filter((g) => g != null).length} passages notés sur ${run.retrieved.length}.` }));
      parts.push(plist(run.retrieved, { dimUnkept: true, tags: true }));
    }
    if (id === "expand" && run.widened.length) {
      parts.push(el("p", { class: "explain", text: `${run.kept.size || run.retrieved.length} passages élargis à ${run.widened.length} : ${run.neighbours.size} chunks voisins ajoutés.` }));
      parts.push(plist(run.widened, { tags: true, rankFromOrder: true }));
    }
    if (id === "route" && run.isNumeric != null) {
      parts.push(el("p", { class: run.isNumeric ? "accepted" : "explain", text: run.isNumeric ? "La question donne sa formule et demande une valeur : la calculatrice est appelée." : "Aucune formule à vérifier dans la question : la génération répond directement." }));
    }
    if (id === "calculate" && run.calc) parts.push(...calcDetail(run.calc));
    if (id === "generate" && run.answer != null) parts.push(...generateDetail(run));
  }
  put(body, ...parts);
}

function calcDetail(calc) {
  const out = [];
  const c = calc.calculation;
  if (c) {
    out.push(el("div", { class: "formula", text: c.expression }));
    const rows = c.inputs.map((v) =>
      el(
        "tr",
        {},
        el("td", { class: "mono", text: v.name }),
        el("td", { text: v.label ? `L${v.line} : ${v.label}` : `L${v.line} (ligne absente)` }),
        el("td", { text: v.statement ? `${statementName(v.statement)}${v.scale ? `, en ${SCALES[v.scale] ?? v.scale}` : ""}, p. ${v.page}` : "" }),
        el("td", { class: "num", text: v.printed ?? "" }),
      ),
    );
    out.push(
      el(
        "table",
        { class: "vars" },
        el("thead", {}, el("tr", {}, el("th", { text: "Variable" }), el("th", { text: "Ligne désignée par le modèle" }), el("th", { text: "Lu par le code" }), el("th", { text: "Valeur" }))),
        el("tbody", {}, ...rows),
      ),
    );
  }
  if (calc.computed) out.push(el("p", { class: "accepted", text: `Résultat ${calc.computed} : toutes les vérifications passent, le chiffre est transmis au générateur.` }));
  else out.push(el("p", { class: "refusal", text: `Refus : ${calc.calc_error}. Le générateur répond sans chiffre vérifié ; la calculatrice ne peut pas introduire d'erreur.` }));
  if (!c && state.mode === "replay") out.push(el("p", { class: "explain", text: "Le détail des lignes n'est pas enregistré pour ce run ; lancez la question en direct pour le voir." }));
  return out;
}

const SCALES = { thousands: "milliers", millions: "millions", billions: "milliards", units: "unités" };

function statementName(s) {
  return { income_statement: "Compte de résultat", balance_sheet: "Bilan", cash_flow: "Flux de trésorerie", other: "Autre tableau" }[s] ?? s;
}

function generateDetail(run) {
  const budget = run.settings.num_ctx;
  const reserved = (run.settings.max_tokens ?? 1024) + 256;
  const used = run.promptTokens ?? 0;
  const pct = (n) => `${Math.min(100, (n / budget) * 100).toFixed(1)}%`;
  return [
    el("p", { class: "explain", text: `Le prompt fait environ ${thousands(used)} tokens sur une fenêtre de ${budget / 1024}K ; ${thousands(reserved)} sont réservés à la réponse et à la marge.${run.dropped ? ` ${run.dropped} passages ont été coupés par la fin pour tenir.` : ""}` }),
    el("div", { class: "gauge" }, el("i", { class: "used", style: `width:${pct(used)}` }), el("i", { class: "out", style: `width:${pct(reserved)}` })),
    el("div", { class: "gauge-legend" }, el("span", { text: "Contexte et instructions" }), el("span", { text: `Fenêtre de ${budget / 1024}K tokens` })),
    run.calc?.computed ? el("p", { class: "accepted", text: `Chiffre vérifié transmis : ${run.calc.computed}.` }) : null,
    run.truncated ? el("p", { class: "refusal", text: `La génération s'est arrêtée sur la limite de ${run.settings.max_tokens ?? 1024} tokens de sortie (done_reason "length" renvoyé par Ollama) : la réponse est inachevée.` }) : null,
  ];
}

// --- examples -------------------------------------------------------------------------

function renderExamples() {
  const query = $("#examples-search").value.trim().toLowerCase();
  const items = state.examples
    .filter((e) => (state.filter === "calc" ? e.computed : state.filter === "good" ? e.outcome === "good_job" : true))
    .filter((e) => !query || `${e.company} ${e.question}`.toLowerCase().includes(query))
    .sort((a, b) => Number(!!b.computed) - Number(!!a.computed) || Number(b.outcome === "good_job") - Number(a.outcome === "good_job"));
  put($("#examples-list"),
    ...items.map((e) => {
      const b = el(
        "button",
        { type: "button" },
        el("span", { class: "ex-company", text: `${e.company}, ${e.doc_id.replaceAll("_", " ")}` }),
        el("span", { class: "ex-q", text: e.question }),
        el(
          "span",
          { class: "ex-side" },
          el("span", { class: "ex-gold", text: e.gold_answer.length > 26 ? `${e.gold_answer.slice(0, 26)}…` : e.gold_answer }),
          e.computed ? el("span", { class: "tag kept", text: `calcul ${e.computed}` }) : null,
          e.outcome ? el("span", { class: `outcome ${e.outcome}`, text: OUTCOMES[e.outcome].split(",")[0] }) : null,
        ),
      );
      b.addEventListener("click", () => pickExample(e));
      return el("li", {}, b);
    }),
  );
  if (!items.length) put($("#examples-list"), el("li", { class: "empty", text: "Aucune question ne correspond au filtre." }));
}

function pickExample(e) {
  state.example = e;
  $("#question").value = e.question;
  $("#doc").value = e.doc_id;
  $("#example-line").hidden = false;
  $("#example-line").textContent = `Question FinanceBench sur ${e.company}. Réponse attendue : ${e.gold_answer}.`;
  $("#examples").close();
  nodeEls.question.sub.textContent = clip(docLabel(), 19);
  renderGold();
  updateRunButton();
}

// --- modes & boot ---------------------------------------------------------------------

// The results and production pages sit beside the demo rather than replacing a mode: a
// run in progress keeps going while one is open and is still there on the way back.
const PAGES = { results: "#results", prod: "#prod" };

function showPage(page) {
  $(".layout").hidden = page != null;
  for (const [name, sel] of Object.entries(PAGES)) {
    $(sel).hidden = name !== page;
    $(`#mode-${name}`).setAttribute("aria-selected", String(name === page));
  }
  $("#mode-live").setAttribute("aria-selected", String(page == null && state.mode === "live"));
  $("#mode-replay").setAttribute("aria-selected", String(page == null && state.mode === "replay"));
  if (page) window.scrollTo(0, 0);
}

// Hovering a workflow step lights the calls it makes and the services they reach.
function wireArchitecture() {
  for (const step of document.querySelectorAll(".arch .wf")) {
    const light = (on) => {
      for (const call of document.querySelectorAll(`.arch .call[data-from="${step.dataset.node}"]`)) call.classList.toggle("hot", on);
      for (const svc of step.dataset.calls.split(" ").filter(Boolean)) {
        document.querySelector(`.arch .svc[data-svc="${svc}"]`)?.classList.toggle("hot", on);
      }
    };
    for (const [ev, on] of [["mouseenter", true], ["mouseleave", false], ["focus", true], ["blur", false]]) {
      step.addEventListener(ev, () => light(on));
    }
  }
}

function setMode(mode) {
  showPage(null);
  if (state.run && !state.run.finished) return;
  state.mode = mode;
  $("#mode-live").setAttribute("aria-selected", String(mode === "live"));
  $("#mode-replay").setAttribute("aria-selected", String(mode === "replay"));
  $("#run").textContent = mode === "replay" ? "Rejouer" : "Lancer";
  renderRail();
  drawIdle();
  updateRunButton();
  if (mode === "replay" && !(state.example && state.example.recorded)) $("#examples").showModal();
}

async function boot() {
  buildDiagram();
  try {
    const [options, examples] = await Promise.all([
      fetch("/demo/options").then((r) => r.json()),
      fetch("/demo/examples").then((r) => r.json()),
    ]);
    state.options = options;
    state.examples = examples;
    state.settings = { ...options.defaults };
    const ollama = options.ollama;
    const ok = options.models.length > 0;
    $("#status-dot").className = `dot ${ok && !ollama?.on_cpu ? "ok" : "bad"}`;
    $("#status-text").textContent = !ok
      ? "Ollama injoignable"
      : ollama?.loaded
        ? `${ollama.loaded} chargé, ${ollama.gpu_share}% sur GPU`
        : "Ollama et Qdrant prêts";
    if (options.tracing) {
      const link = $("#phoenix-link");
      link.hidden = false;
      link.href = options.phoenix_url;
    }
    if (!options.replay) $("#mode-replay").disabled = true;
  } catch (err) {
    $("#status-dot").className = "dot bad";
    $("#status-text").textContent = "API injoignable";
    state.options = { models: [], chunk_sizes: [1024], defaults: {}, replay: null };
    state.settings = {};
    $("#banner").hidden = false;
    $("#banner").textContent = `Impossible de joindre l'API : ${err.message}`;
    return;
  }
  const docs = [...new Set(state.examples.map((e) => e.doc_id))].sort();
  put($("#doc"), el("option", { value: "", text: "Tous les rapports" }), ...docs.map((d) => el("option", { value: d, text: d.replaceAll("_", " ") })));
  buildRail();
  renderRail();
  drawIdle();
  clearPanes();
  select("question");
  state.pinned = null;

  $("#mode-live").addEventListener("click", () => setMode("live"));
  $("#mode-replay").addEventListener("click", () => setMode("replay"));
  $("#mode-results").addEventListener("click", () => showPage("results"));
  $("#mode-prod").addEventListener("click", () => showPage("prod"));
  wireArchitecture();
  $("#run").addEventListener("click", run);
  $("#question").addEventListener("input", () => {
    updateRunButton();
    $("#example-line").hidden = !(state.example && state.example.question === $("#question").value.trim());
  });
  $("#question").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey) && !$("#run").disabled) run();
  });
  $("#doc").addEventListener("change", () => (nodeEls.question.sub.textContent = clip(docLabel() || "tous les rapports", 19)));
  $("#open-examples").addEventListener("click", () => {
    renderExamples();
    $("#examples").showModal();
  });
  $("#close-examples").addEventListener("click", () => $("#examples").close());
  $("#examples-search").addEventListener("input", renderExamples);
  for (const b of $("#examples-filter").querySelectorAll("button")) {
    b.addEventListener("click", () => {
      state.filter = b.dataset.filter;
      for (const o of $("#examples-filter").querySelectorAll("button")) o.setAttribute("aria-pressed", String(o === b));
      renderExamples();
    });
  }
  renderExamples();
  updateRunButton();
}

boot();
