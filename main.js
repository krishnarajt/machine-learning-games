// =====================================================================
//  main.js — the app shell.
//
//  It reads the game registry (games/index.js), then builds:
//    • the sidebar navigation (grouped by category),
//    • the dashboard view (hero + idea cards + launcher + ML families),
//    • one <section.view> per game, lazily mounting the game on first visit.
//  Routing is hash-based (#key). Games expose onEnter/onLeave so their
//  animation loops pause when you navigate away.
// =====================================================================
import { GAMES } from "./games/index.js";
import { el, Loop, REDUCED_MOTION, classColor, withAlpha } from "./shared.js";

const content = document.getElementById("content");
const nav = document.getElementById("nav");

/* ----------------------------------------------------------------- *
 * Sidebar navigation (Dashboard + games grouped by category)
 * ----------------------------------------------------------------- */
nav.append(
  el("a", { class: "nav-link", href: "#dashboard", "data-view": "dashboard" },
    el("span", { class: "nav-ico", "aria-hidden": "true" }, "◎"), " Dashboard")
);

const categories = [];
for (const g of GAMES) if (!categories.includes(g.category)) categories.push(g.category);

for (const cat of categories) {
  nav.append(el("div", { class: "nav-group-label" }, cat));
  for (const g of GAMES.filter((x) => x.category === cat)) {
    nav.append(
      el("a", { class: "nav-link", href: "#" + g.key, "data-view": g.key },
        el("span", { class: "nav-ico", "aria-hidden": "true" }, g.icon),
        " " + g.name,
        g.tag ? el("span", { class: "nav-tag" }, g.tag) : null)
    );
  }
}

/* ----------------------------------------------------------------- *
 * Dashboard view
 * ----------------------------------------------------------------- */
function buildDashboard() {
  const heroCanvas = el("canvas", { id: "heroNet", width: 520, height: 320, role: "img",
    "aria-label": "An animated neural network with signals pulsing along its connections." });

  const launcher = el("div", { class: "launch-grid" });
  for (const g of GAMES) {
    launcher.append(
      el("button", { class: "launch-tile", "data-goto": g.key },
        el("span", { class: "launch-ico", "aria-hidden": "true" }, g.icon),
        el("span", { class: "lt-tag" }, g.category),
        el("span", { class: "lt-name" }, g.name),
        el("span", { class: "lt-desc" }, g.blurb))
    );
  }

  const families = [
    { h: "Supervised learning", p: "You show the model labelled examples — input → known answer — and it learns to predict the answer for new inputs.", list: "Regression (numbers) · Classification (categories)" },
    { h: "Unsupervised learning", p: "No labels at all. The model finds structure on its own — natural groups, or the directions the data really varies in.", list: "Clustering · Dimensionality reduction" },
    { h: "Reinforcement learning", p: "An agent acts in an environment, gets rewards or penalties, and learns a strategy that maximises long-run reward.", list: "Q-learning · policies & value functions" },
  ];

  const view = el("section", { class: "view active", "data-view": "dashboard" },
    el("div", { class: "hero" },
      el("div", { class: "hero-copy" },
        el("span", { class: "hero-kicker" }, "Learn machine learning by playing with it"),
        el("h1", { html: 'The famous algorithms, each turned into a <span class="grad">game</span>.' }),
        el("p", {}, "Machine learning sounds abstract until you can poke at it. Here every major algorithm is a small interactive game: drag the data, place the clusters, watch a neural network learn a curve, train an agent to cross a grid. Play first — then read exactly how each one works underneath."),
        el("div", { class: "hero-actions" },
          el("button", { "data-goto": GAMES[0].key }, "Start with " + GAMES[0].name + " →"),
          el("button", { class: "ghost", "data-goto": "neural-network" }, "Jump to the neural net"))),
      el("div", { class: "hero-canvas-wrap" },
        heroCanvas,
        el("span", { class: "hero-caption" }, "A neural network — the idea behind much of modern AI"))),

    el("div", { class: "idea-grid" },
      el("div", { class: "idea-card" },
        el("div", { class: "big" }, "17"),
        el("h3", {}, "Algorithms, 17 games"),
        el("p", {}, "From a line of best fit to a self-training agent — the canon of classic ML.")),
      el("div", { class: "idea-card" },
        el("div", { class: "big" }, "0"),
        el("h3", {}, "Servers involved"),
        el("p", {}, "Every model trains live in your browser. No uploads, no backend, no waiting.")),
      el("div", { class: "idea-card" },
        el("div", { class: "big" }, "∞"),
        el("h3", {}, "Times you can retrain"),
        el("p", {}, "Reset the data, change the settings, and watch the algorithm relearn from scratch."))),

    el("div", { class: "section-head" },
      el("h2", {}, "Pick a game"),
      el("p", {}, "Each one shows the live algorithm first, then explains the maths beneath it. No order required — but the list runs roughly simple → advanced.")),
    launcher,

    el("div", { class: "section-head" },
      el("h2", {}, "The three families of machine learning"),
      el("p", {}, "Almost every algorithm here belongs to one of these. Knowing which family a method is in tells you what problem it solves.")),
    el("div", { class: "family-grid" },
      ...families.map((f) =>
        el("div", { class: "family-card" },
          el("h3", {}, f.h),
          el("p", {}, f.p),
          el("div", { class: "fc-list" }, f.list)))),

    el("footer", {},
      el("p", {}, "Built as an interactive, honest introduction to machine learning. Models are deliberately small so you can watch every step — real systems use the same ideas at vastly larger scale."))
  );
  return view;
}

/* ----------------------------------------------------------------- *
 * Game views — header + mount point + lesson. The game itself is
 * mounted lazily the first time the view is shown.
 * ----------------------------------------------------------------- */
const mounted = new Map(); // key -> { onEnter, onLeave } | null

function buildGameView(g) {
  const root = el("div", { class: "game-root" });
  const view = el("section", { class: "view", "data-view": g.key },
    el("span", { class: "eyebrow" }, g.eyebrow || g.category),
    el("h1", { class: "view-title", html: g.title }),
    el("p", { class: "lede", html: g.lede }),
    root,
    el("div", { class: "lesson-intro" }, "How it works & the maths"),
    el("div", { class: "lesson", html: g.lesson })
  );
  view._gameRoot = root;
  view._game = g;
  return view;
}

content.append(buildDashboard());
const gameViews = new Map();
for (const g of GAMES) {
  const v = buildGameView(g);
  gameViews.set(g.key, v);
  content.append(v);
}

function ensureMounted(key) {
  if (mounted.has(key)) return mounted.get(key);
  const view = gameViews.get(key);
  let handle = null;
  if (view && view._game.mount) {
    try {
      handle = view._game.mount(view._gameRoot) || null;
    } catch (err) {
      console.error("Failed to mount game", key, err);
      view._gameRoot.append(el("p", { class: "verdict" }, "This game failed to load — check the console."));
    }
  }
  mounted.set(key, handle);
  return handle;
}

/* ----------------------------------------------------------------- *
 * Routing
 * ----------------------------------------------------------------- */
const views = [...document.querySelectorAll(".view")];
const navLinks = [...document.querySelectorAll(".nav-link")];
const validKeys = new Set(views.map((v) => v.dataset.view));
let current = null;

const TITLE = "Machine Learning Games";
function keyFromHash() {
  const k = (location.hash || "").replace(/^#/, "");
  return validKeys.has(k) ? k : "dashboard";
}

function setView(key) {
  const target = views.find((v) => v.dataset.view === key) || views[0];
  const k = target.dataset.view;

  // tear down the one we are leaving
  if (current && current !== k) {
    const prev = mounted.get(current);
    if (prev && prev.onLeave) try { prev.onLeave(); } catch (e) { console.error(e); }
  }
  if (current === "dashboard" && k !== "dashboard") heroLoop.stop();

  views.forEach((v) => v.classList.toggle("active", v === target));
  navLinks.forEach((n) => {
    const on = n.dataset.view === k;
    n.classList.toggle("active", on);
    if (on) n.setAttribute("aria-current", "page");
    else n.removeAttribute("aria-current");
  });

  const g = GAMES.find((x) => x.key === k);
  document.title = g ? `${g.name} · ${TITLE}` : TITLE;
  window.scrollTo({ top: 0, behavior: "auto" });
  closeNav();

  if (k === "dashboard") {
    if (!REDUCED_MOTION) heroLoop.start();
    else drawHero(0);
  } else {
    const handle = ensureMounted(k);
    if (handle && handle.onEnter) try { handle.onEnter(); } catch (e) { console.error(e); }
  }
  current = k;
}

document.addEventListener("click", (e) => {
  const goto = e.target.closest("[data-goto]");
  if (!goto) return;
  e.preventDefault();
  const key = goto.dataset.goto;
  if (location.hash.replace(/^#/, "") === key) setView(key);
  else location.hash = key;
});
window.addEventListener("hashchange", () => setView(keyFromHash()));

/* ----------------------------------------------------------------- *
 * Mobile drawer
 * ----------------------------------------------------------------- */
const menuBtn = document.getElementById("menuBtn");
const scrim = document.getElementById("scrim");
const sidebar = document.getElementById("sidebar");
function openNav() {
  document.body.classList.add("nav-open");
  if (menuBtn) menuBtn.setAttribute("aria-expanded", "true");
}
function closeNav() {
  document.body.classList.remove("nav-open");
  if (menuBtn) menuBtn.setAttribute("aria-expanded", "false");
}
if (menuBtn) menuBtn.addEventListener("click", openNav);
if (scrim) scrim.addEventListener("click", closeNav);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && document.body.classList.contains("nav-open")) closeNav();
});

/* ----------------------------------------------------------------- *
 * Hero — an animated little neural network with travelling signals
 * ----------------------------------------------------------------- */
const LAYERS = [3, 5, 4, 2];
let heroNodes = [];
let heroEdges = [];
function layoutHero(W, H) {
  heroNodes = [];
  heroEdges = [];
  const padX = 50, padY = 36;
  LAYERS.forEach((count, li) => {
    const x = padX + (li / (LAYERS.length - 1)) * (W - 2 * padX);
    for (let i = 0; i < count; i++) {
      const y = padY + ((i + 0.5) / count) * (H - 2 * padY);
      heroNodes.push({ x, y, layer: li });
    }
  });
  for (let li = 0; li < LAYERS.length - 1; li++) {
    const a = heroNodes.filter((n) => n.layer === li);
    const b = heroNodes.filter((n) => n.layer === li + 1);
    for (const na of a) for (const nb of b) heroEdges.push({ a: na, b: nb, phase: Math.random() });
  }
}

function drawHero(t) {
  const cv = document.getElementById("heroNet");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  if (!heroNodes.length) layoutHero(W, H);
  ctx.clearRect(0, 0, W, H);

  // edges
  for (const e of heroEdges) {
    ctx.strokeStyle = "rgba(139,123,255,0.13)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(e.a.x, e.a.y);
    ctx.lineTo(e.b.x, e.b.y);
    ctx.stroke();
    // travelling pulse
    const p = (t * 0.0004 + e.phase) % 1;
    const px = e.a.x + (e.b.x - e.a.x) * p;
    const py = e.a.y + (e.b.y - e.a.y) * p;
    ctx.beginPath();
    ctx.arc(px, py, 2.2, 0, Math.PI * 2);
    ctx.fillStyle = withAlpha("#5ad2ff", 0.9 * Math.sin(p * Math.PI));
    ctx.fill();
  }
  // nodes
  heroNodes.forEach((n, i) => {
    const pulse = 0.5 + 0.5 * Math.sin(t * 0.002 + i);
    const r = 7 + pulse * 2;
    const grad = ctx.createRadialGradient(n.x, n.y, 1, n.x, n.y, r + 6);
    const col = n.layer === LAYERS.length - 1 ? "#ffd166" : n.layer === 0 ? "#43e0c8" : "#5ad2ff";
    grad.addColorStop(0, col);
    grad.addColorStop(1, withAlpha(col, 0));
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(n.x, n.y, r + 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(n.x, n.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
  });
}
const heroLoop = new Loop((dt, t) => drawHero(t));

/* ----------------------------------------------------------------- *
 * Boot
 * ----------------------------------------------------------------- */
setView(keyFromHash());
