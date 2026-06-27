// =====================================================================
//  DBSCAN — density-based clustering.
//
//  Where k-Means assumes round blobs and forces you to pick k, DBSCAN
//  grows clusters wherever points are densely packed and leaves the
//  sparse stragglers labelled as noise.
//
//  Two knobs:
//     ε (epsilon)  — the neighbourhood radius
//     minPts       — how many points (incl. self) make a dense region
//
//  Vocabulary:
//     CORE   — a point with ≥ minPts neighbours within ε
//     BORDER — within ε of a core point, but not core itself
//     NOISE  — reached by nobody
//  Clusters are grown from core points by region-growing ("expandCluster").
// =====================================================================
import {
  el, ghostButton, slider, Plot,
  twoMoons, circles, blobs, randRange,
  dist, classColor, withAlpha,
} from "../shared.js";

export default {
  key: "dbscan",
  name: "DBSCAN",
  icon: "🫧",
  category: "Clustering",
  tag: "Unsup",
  blurb: "Find clusters of any shape — and flag the outliers — with no need to pick k.",
  eyebrow: "Clustering · Unsupervised",
  title: 'DBSCAN: clustering by <span class="grad">density</span>',
  lede:
    "k-Means assumes round blobs and makes you choose how many there are. DBSCAN does neither: it grows clusters " +
    "wherever points are <em>densely packed</em>, traces out arbitrary shapes like crescents and rings, and leaves " +
    "sparse stragglers labelled as noise. Tune the neighbourhood size and watch clusters form.",

  mount(root) {
    // ---- state ----
    let EPS = 0.18;          // ε neighbourhood radius (data units)
    let MINPTS = 4;          // minimum points (including self) for a core point
    let dataset = "moons";   // which generator
    let points = [];         // { x, y, cluster, type } — cluster -1 = noise/unassigned
    let hover = null;        // index of point under the pointer (for ε circle), or null
    let stats = { clusters: 0, noise: 0, core: 0, border: 0 };

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Scatter plot where points are coloured by the DBSCAN cluster they belong to; grey crosses mark noise." });
    const plot = new Plot(canvas, { xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2, pad: 16 });

    // dataset segmented control
    const dataDefs = [
      ["moons", "Moons"],
      ["circles", "Circles"],
      ["blobs", "Blobs"],
      ["random", "Random"],
    ];
    const dataBtns = {};
    const dataSeg = el("div", { class: "seg-group" },
      dataDefs.map(([key, label]) => {
        const b = el("button", { class: key === dataset ? "active" : "" }, label);
        b.addEventListener("click", () => setDataset(key));
        dataBtns[key] = b;
        return b;
      }));

    const epsSlider = slider({ min: 0.05, max: 0.6, step: 0.01, value: EPS, label: "ε (radius) =", format: (v) => v.toFixed(2) });
    const minPtsSlider = slider({ min: 2, max: 10, step: 1, value: MINPTS, label: "minPts =", format: (v) => v });
    const newDataBtn = ghostButton("New data");

    const clustersEl = el("span", { class: "v" }, "0");
    const noiseEl = el("span", { class: "v" }, "0");
    const coreEl = el("span", { class: "v" }, "0");
    const borderEl = el("span", { class: "v" }, "0");

    const legend = el("div", { class: "legend" },
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot" }), "Core (solid, large)"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { opacity: "0.55" } }), "Border (faint, small)"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: "#7c89b8" } }), "Noise (✕)"));

    const verdict = el("p", { class: "verdict" },
      "Drag ε and minPts and watch clusters appear. Hover a point to see its ε-neighbourhood circle.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" }, "Hover a point to draw its ε-radius. Click empty space to add a point.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Result"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Clusters found"), clustersEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Noise points"), noiseEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Core points"), coreEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Border points"), borderEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, dataSeg),
            el("div", { class: "controls" }, epsSlider.wrap),
            el("div", { class: "controls" }, minPtsSlider.wrap),
            el("div", { class: "controls" }, newDataBtn)),
          legend)),
      verdict
    );

    // ---- data ----
    function seedData() {
      let raw;
      if (dataset === "moons") raw = twoMoons(160, 0.1);
      else if (dataset === "circles") raw = circles(160, 0.06);
      else if (dataset === "blobs") raw = blobs([[-0.55, 0.45], [0.55, 0.5], [0.05, -0.55]], 53, 0.11);
      else raw = Array.from({ length: 160 }, () => ({ x: randRange(-1, 1), y: randRange(-1, 1) }));
      // strip incoming labels; DBSCAN is unsupervised
      points = raw.map((p) => ({ x: p.x, y: p.y, cluster: -1, type: "noise" }));
    }

    // ---- the algorithm: classic DBSCAN ----------------------------------
    // For each point compute its ε-neighbourhood (O(n²), fine here). A point
    // is CORE if its neighbourhood (including itself) has ≥ minPts members.
    // Region-grow from each unvisited core point: pull in every neighbour;
    // if a neighbour is itself core, enqueue ITS neighbours; non-core
    // neighbours join as BORDER but do not expand. Anything never reached
    // stays NOISE.
    function run() {
      const n = points.length;
      // reset
      for (const p of points) { p.cluster = -1; p.type = "noise"; }
      if (n === 0) { refresh(); return; }

      // precompute ε-neighbourhoods and core flags
      const neigh = new Array(n);
      const isCore = new Array(n).fill(false);
      for (let i = 0; i < n; i++) {
        const list = [];
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          if (dist(points[i].x, points[i].y, points[j].x, points[j].y) <= EPS) list.push(j);
        }
        neigh[i] = list;
        // neighbourhood size INCLUDING the point itself
        if (list.length + 1 >= MINPTS) isCore[i] = true;
      }

      const visited = new Array(n).fill(false);
      let clusterId = -1;

      for (let i = 0; i < n; i++) {
        if (visited[i]) continue;
        visited[i] = true;
        if (!isCore[i]) continue;        // only start clusters from core points
        // start a new cluster and region-grow (expandCluster)
        clusterId++;
        points[i].cluster = clusterId;
        points[i].type = "core";

        const queue = neigh[i].slice();
        for (let q = 0; q < queue.length; q++) {
          const j = queue[q];
          if (!visited[j]) {
            visited[j] = true;
            // a core neighbour extends the frontier
            if (isCore[j]) {
              for (const k of neigh[j]) queue.push(k);
            }
          }
          // density-reachable: claim it for this cluster if not already taken
          if (points[j].cluster === -1) {
            points[j].cluster = clusterId;
            points[j].type = isCore[j] ? "core" : "border";
          }
        }
      }

      // tally
      let noise = 0, core = 0, border = 0;
      for (const p of points) {
        if (p.cluster === -1) noise++;
        else if (p.type === "core") core++;
        else border++;
      }
      stats = { clusters: clusterId + 1, noise, core, border };
      refresh();
    }

    // ---- draw ----
    function draw() {
      plot.clear("#070d1a");
      plot.grid(0.5);

      // ε-radius of the hovered point (makes ε tangible)
      if (hover != null && points[hover]) {
        const p = points[hover];
        const rPix = plot.scaleX(EPS);
        const cx = plot.sx(p.x), cy = plot.sy(p.y);
        const col = p.cluster < 0 ? "#7c89b8" : classColor(p.cluster);
        plot.ctx.beginPath();
        plot.ctx.arc(cx, cy, rPix, 0, Math.PI * 2);
        plot.ctx.fillStyle = withAlpha(col, 0.1);
        plot.ctx.fill();
        plot.ctx.lineWidth = 1.5;
        plot.ctx.strokeStyle = withAlpha(col, 0.7);
        plot.ctx.setLineDash([4, 4]);
        plot.ctx.stroke();
        plot.ctx.setLineDash([]);
      }

      // points: core solid+large, border faint+small, noise grey ✕
      for (const p of points) {
        if (p.cluster < 0) {
          // noise — small grey cross
          const px = plot.sx(p.x), py = plot.sy(p.y);
          plot.ctx.strokeStyle = "rgba(124,137,184,0.85)";
          plot.ctx.lineWidth = 1.5;
          plot.ctx.beginPath();
          plot.ctx.moveTo(px - 3.5, py - 3.5); plot.ctx.lineTo(px + 3.5, py + 3.5);
          plot.ctx.moveTo(px + 3.5, py - 3.5); plot.ctx.lineTo(px - 3.5, py + 3.5);
          plot.ctx.stroke();
        } else if (p.type === "core") {
          plot.point(p.x, p.y, 5.5, classColor(p.cluster), "rgba(7,13,26,0.85)", 1.5);
        } else {
          // border — smaller and lighter
          plot.point(p.x, p.y, 3.5, withAlpha(classColor(p.cluster), 0.55), "rgba(7,13,26,0.5)", 1);
        }
      }
    }

    function refresh() {
      draw();
      clustersEl.textContent = stats.clusters;
      noiseEl.textContent = stats.noise;
      coreEl.textContent = stats.core;
      borderEl.textContent = stats.border;

      const n = points.length;
      const noiseFrac = n ? stats.noise / n : 0;
      if (n === 0) {
        verdict.innerHTML = "No points. Press <b>New data</b> or click the canvas.";
      } else if (stats.clusters === 0 || noiseFrac > 0.8) {
        verdict.innerHTML = `<b>ε too small.</b> Almost nothing has ${MINPTS} neighbours within ε = ${EPS.toFixed(2)}, ` +
          `so ${stats.noise} of ${n} points are flagged as <b>noise</b>. Widen ε or lower minPts.`;
      } else if (stats.clusters === 1 && noiseFrac < 0.05) {
        verdict.innerHTML = `<b>ε too big.</b> Every dense region now touches every other, so it all merged into ` +
          `<b>one</b> blob. Shrink ε so the gaps between groups reappear.`;
      } else {
        verdict.innerHTML = `Found <b>${stats.clusters}</b> cluster${stats.clusters === 1 ? "" : "s"} and ` +
          `<b>${stats.noise}</b> outlier${stats.noise === 1 ? "" : "s"} — no <code>k</code> required. ` +
          `${stats.core} core points anchor the clusters; ${stats.border} border points cling to their edges.`;
      }
    }

    // ---- events ----
    function setDataset(key) {
      dataset = key;
      for (const k in dataBtns) dataBtns[k].classList.toggle("active", k === key);
      hover = null;
      seedData();
      run();
    }

    epsSlider.input.addEventListener("input", () => { EPS = parseFloat(epsSlider.input.value); run(); });
    minPtsSlider.input.addEventListener("input", () => { MINPTS = parseInt(minPtsSlider.input.value, 10); run(); });
    newDataBtn.addEventListener("click", () => { hover = null; seedData(); run(); });

    // nearest point to the pointer (within a small pixel radius) for the ε circle
    function pickNearest(px, py) {
      let best = null, bd = 18 * 18; // 18px hit radius (squared)
      for (let i = 0; i < points.length; i++) {
        const dx = plot.sx(points[i].x) - px;
        const dy = plot.sy(points[i].y) - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < bd) { bd = d2; best = i; }
      }
      return best;
    }

    canvas.addEventListener("pointermove", (e) => {
      const { px, py } = plot.eventToData(e);
      const h = pickNearest(px, py);
      if (h !== hover) { hover = h; draw(); }
    });
    canvas.addEventListener("pointerleave", () => { if (hover != null) { hover = null; draw(); } });

    canvas.addEventListener("pointerdown", (e) => {
      const { x, y, px, py } = plot.eventToData(e);
      // clicking near an existing point just keeps the ε circle; clicking empty space adds one
      if (pickNearest(px, py) != null) return;
      if (x < plot.xmin || x > plot.xmax || y < plot.ymin || y > plot.ymax) return;
      points.push({ x, y, cluster: -1, type: "noise" });
      run();
    });

    // ---- init ----
    seedData();
    run();

    return {
      onEnter() { draw(); },
      onLeave() { hover = null; /* no Loop to stop; nothing animates */ },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🫧 The job: clusters of any shape, plus outliers</h3>
    <p>DBSCAN — <b>Density-Based Spatial Clustering of Applications with Noise</b> — is unsupervised clustering for the cases <b>k-Means</b> gets wrong. You don't tell it how many groups there are; it discovers that itself. It happily traces out crescents, rings and other non-round shapes, and instead of forcing every point into a cluster it labels the lonely ones as <em>noise</em>. The whole idea: a cluster is a region where points are <b>packed densely</b>, separated from other clusters by sparse gaps.</p>
  </div>
  <div class="lesson-block">
    <h3>🔤 Three kinds of point</h3>
    <p>Everything hinges on two knobs — a radius <code>ε</code> and a count <code>minPts</code> — and how many neighbours each point has inside that radius:</p>
    <ul class="bullets">
      <li><b>Core point</b> (solid, large): has at least <code>minPts</code> points within distance <code>ε</code> of it (counting itself). These sit in the dense interior of a cluster.</li>
      <li><b>Border point</b> (faint, small): not dense enough to be core, but lies within <code>ε</code> of some core point — it clings to the cluster's edge.</li>
      <li><b>Noise</b> (grey ✕): reached by nobody. Too far from any dense region to belong anywhere.</li>
    </ul>
    <p>Hover a point above to see its ε-circle: count how many other points fall inside and you can read off whether it's core.</p>
  </div>
  <div class="lesson-block">
    <h3>🌱 How clusters grow — density-reachability</h3>
    <ol class="steps">
      <li><b>Neighbourhood.</b> For every point, find all points within Euclidean distance <code>ε</code>.</li>
      <li><b>Flag cores.</b> Mark a point <em>core</em> if its neighbourhood (including itself) has ≥ <code>minPts</code> members.</li>
      <li><b>Expand.</b> Pick an unvisited core point, start a new cluster, and region-grow: pull in every neighbour. If a neighbour is <em>itself</em> core, also enqueue <em>its</em> neighbours — the cluster keeps spreading through chains of overlapping core circles. Non-core neighbours join as border but don't extend the frontier.</li>
      <li><b>Repeat</b> from the next unvisited core point. Anything never reached stays noise.</li>
    </ol>
    <p>Point <b>q</b> is <em>density-reachable</em> from <b>p</b> if you can hop from <b>p</b> to <b>q</b> through a chain of core points, each within <code>ε</code> of the next. That chaining is exactly why DBSCAN can wrap around a crescent: a long thin band of overlapping ε-circles is still one connected dense region.</p>
    <div class="formula">N<sub>ε</sub>(p) = { q : ‖p − q‖ ≤ ε } &nbsp;&nbsp; p is core ⇔ |N<sub>ε</sub>(p)| ≥ minPts</div>
  </div>
  <div class="lesson-block">
    <h3>🏆 Why it beats k-Means here</h3>
    <ul class="bullets">
      <li><b>Arbitrary shapes.</b> Try the <em>Moons</em> or <em>Circles</em> data. DBSCAN follows the curves. k-Means, which measures distance to a central mean, would slice each moon straight down the middle and split the ring into pie wedges.</li>
      <li><b>No <code>k</code>.</b> It returns however many dense regions exist. You never guess the number of clusters.</li>
      <li><b>Outliers stay outliers.</b> Sparse points are labelled noise instead of being dragged into the nearest cluster and distorting it.</li>
    </ul>
  </div>
  <div class="lesson-block">
    <h3>⚠️ The catches</h3>
    <ul class="bullets">
      <li><b>One global ε.</b> A single radius can't fit clusters of very different density — set ε for the loose cluster and the tight one merges; set it for the tight one and the loose one shatters into noise. (Variants like HDBSCAN address this.)</li>
      <li><b>Parameter sensitivity.</b> Nudge the sliders: <em>ε too small</em> → everything becomes noise; <em>ε too big</em> → all clusters merge into one blob. <code>minPts</code> controls how aggressively points are flagged as noise.</li>
      <li><b>High dimensions.</b> In many dimensions distances bunch up, so a meaningful ε is hard to find — DBSCAN is happiest in low-dimensional spaces like this 2-D plot.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> DBSCAN grows a cluster outward from any densely-surrounded "core" point through chains of overlapping ε-neighbourhoods — so it finds clusters of any shape, decides their number on its own, and calls the leftover sparse points noise.</p>
  </div>`,
};
