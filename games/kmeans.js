// =====================================================================
//  k-Means clustering — the reference game.
//
//  Lloyd's algorithm: pick k centroids, then repeat two steps until
//  nothing moves —
//     ASSIGN : colour each point by its nearest centroid
//     UPDATE : move each centroid to the mean of its colour
//  The user adds points, picks k, and steps or auto-runs the loop.
// =====================================================================
import {
  el, ghostButton, slider, Loop, Plot,
  blobs, dist2, classColor, withAlpha, range, randRange,
} from "../shared.js";

export default {
  key: "kmeans",
  name: "k-Means",
  icon: "⭕",
  category: "Clustering",
  tag: "Unsup",
  blurb: "Drop points and watch centroids hunt for the natural groups.",
  eyebrow: "Clustering · Unsupervised",
  title: 'k-Means: <span class="grad">finding groups</span> with no labels',
  lede:
    "Nobody tells k-Means which points belong together — it has to discover the groups itself. " +
    "It plants <em>k</em> centroids, repeatedly recolours every point by its nearest centroid, then slides each " +
    "centroid to the middle of its colour. Watch it settle. Click the canvas to add your own points.",

  mount(root) {
    // ---- state ----
    let K = 3;
    let points = [];          // { x, y, c } — c is current cluster index (-1 = none)
    let centroids = [];       // { x, y }
    let iter = 0;
    let phase = "assign";     // next step to perform
    let settled = false;
    let lastMoved = 0;

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Scatter plot where points are coloured by cluster and large rings mark the centroids." });
    const plot = new Plot(canvas, { xmin: -1.15, xmax: 1.15, ymin: -1.15, ymax: 1.15, pad: 16 });

    const stepBtn = el("button", {}, "Step ▸");
    const runBtn = el("button", {}, "Auto-run");
    const reseedBtn = ghostButton("Re-seed centroids");
    const newDataBtn = ghostButton("New data");
    const clearBtn = ghostButton("Clear");
    const kSlider = slider({ min: 2, max: 5, step: 1, value: K, label: "Clusters k =", format: (v) => v });

    const iterEl = el("span", { class: "v" }, "0");
    const moveEl = el("span", { class: "v" }, "—");
    const inertiaEl = el("span", { class: "v" }, "—");
    const phaseEl = el("span", { class: "v" }, "assign");

    const legend = el("div", { class: "legend" });
    const verdict = el("p", { class: "verdict" }, "Press <b>Step</b> to run one round, or <b>Auto-run</b> to let it settle. Click the canvas to add points.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" }, "Click empty space to add a point. ✕ marks where each centroid started this round.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Algorithm state"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Iteration"), iterEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Next step"), phaseEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Centroids moved"), moveEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Inertia (cost)"), inertiaEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, stepBtn, runBtn),
            el("div", { class: "controls" }, kSlider.wrap),
            el("div", { class: "controls" }, reseedBtn, newDataBtn, clearBtn)),
          legend)),
      verdict
    );

    // ---- helpers ----
    function seedData() {
      // three honest-to-goodness blobs so there is something real to find
      points = blobs([[-0.55, 0.45], [0.55, 0.55], [0.1, -0.55]], 26, 0.14)
        .map((p) => ({ x: p.x, y: p.y, c: -1 }));
    }

    function seedCentroids() {
      // k-means++-ish: first centroid random point, rest far from chosen ones
      centroids = [];
      if (!points.length) { centroids = range(K).map(() => ({ x: randRange(-0.8, 0.8), y: randRange(-0.8, 0.8) })); }
      else {
        const first = points[Math.floor(Math.random() * points.length)];
        centroids.push({ x: first.x, y: first.y });
        while (centroids.length < K) {
          // choose the point with the largest distance to its nearest centroid
          let best = null, bestD = -1;
          for (const p of points) {
            let d = Infinity;
            for (const c of centroids) d = Math.min(d, dist2(p.x, p.y, c.x, c.y));
            if (d > bestD) { bestD = d; best = p; }
          }
          centroids.push({ x: best.x + randRange(-0.02, 0.02), y: best.y + randRange(-0.02, 0.02) });
        }
      }
      centroids.forEach((c) => { c.startX = c.x; c.startY = c.y; });
      iter = 0; phase = "assign"; settled = false; lastMoved = 0;
      points.forEach((p) => (p.c = -1));
      refresh();
    }

    function assign() {
      points.forEach((c) => { c.startX = undefined; });
      for (const p of points) {
        let best = 0, bd = Infinity;
        for (let k = 0; k < centroids.length; k++) {
          const d = dist2(p.x, p.y, centroids[k].x, centroids[k].y);
          if (d < bd) { bd = d; best = k; }
        }
        p.c = best;
      }
      centroids.forEach((c) => { c.startX = c.x; c.startY = c.y; });
    }

    function update() {
      let moved = 0;
      for (let k = 0; k < centroids.length; k++) {
        const members = points.filter((p) => p.c === k);
        if (!members.length) continue;
        const mx = members.reduce((s, p) => s + p.x, 0) / members.length;
        const my = members.reduce((s, p) => s + p.y, 0) / members.length;
        moved += Math.hypot(mx - centroids[k].x, my - centroids[k].y);
        centroids[k].x = mx;
        centroids[k].y = my;
      }
      lastMoved = moved;
      if (moved < 1e-4) settled = true;
    }

    function step() {
      if (!points.length) return;
      if (phase === "assign") { assign(); phase = "update"; }
      else { update(); phase = "assign"; iter++; }
      refresh();
      if (settled && loop.running) { loop.stop(); runBtn.textContent = "Auto-run"; }
    }

    function inertia() {
      let s = 0;
      for (const p of points) {
        if (p.c < 0) continue;
        s += dist2(p.x, p.y, centroids[p.c].x, centroids[p.c].y);
      }
      return s;
    }

    // ---- draw ----
    function draw() {
      plot.clear("#070d1a");
      plot.grid(0.5);

      // points
      for (const p of points) {
        const col = p.c < 0 ? "#7c89b8" : classColor(p.c);
        plot.point(p.x, p.y, 4.5, col, "rgba(7,13,26,0.8)", 1);
      }
      // line from each centroid's start to its new position (the "update" move)
      centroids.forEach((c, k) => {
        if (c.startX !== undefined && (c.startX !== c.x || c.startY !== c.y)) {
          plot.line(c.startX, c.startY, c.x, c.y, withAlpha(classColor(k), 0.5), 2, [4, 4]);
          // start marker
          const sx = plot.sx(c.startX), sy = plot.sy(c.startY);
          plot.ctx.strokeStyle = withAlpha(classColor(k), 0.5);
          plot.ctx.lineWidth = 1.5;
          plot.ctx.beginPath();
          plot.ctx.moveTo(sx - 4, sy - 4); plot.ctx.lineTo(sx + 4, sy + 4);
          plot.ctx.moveTo(sx + 4, sy - 4); plot.ctx.lineTo(sx - 4, sy + 4);
          plot.ctx.stroke();
        }
      });
      // centroids
      centroids.forEach((c, k) => {
        const col = classColor(k);
        const cx = plot.sx(c.x), cy = plot.sy(c.y);
        plot.ctx.beginPath();
        plot.ctx.arc(cx, cy, 12, 0, Math.PI * 2);
        plot.ctx.fillStyle = withAlpha(col, 0.18);
        plot.ctx.fill();
        plot.ctx.lineWidth = 3;
        plot.ctx.strokeStyle = col;
        plot.ctx.stroke();
        plot.ctx.beginPath();
        plot.ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        plot.ctx.fillStyle = "#fff";
        plot.ctx.fill();
      });
    }

    function refresh() {
      draw();
      iterEl.textContent = iter;
      phaseEl.textContent = settled ? "settled ✓" : phase;
      moveEl.textContent = iter === 0 ? "—" : lastMoved.toFixed(4);
      inertiaEl.textContent = points.length ? inertia().toFixed(3) : "—";

      legend.replaceChildren();
      for (let k = 0; k < centroids.length; k++) {
        const n = points.filter((p) => p.c === k).length;
        legend.append(el("span", { class: "legend-item" },
          el("span", { class: "legend-dot", style: { background: classColor(k) } }),
          `Cluster ${k + 1} · ${n} pts`));
      }
      if (settled) {
        verdict.innerHTML = `Converged after <b>${iter}</b> iterations. No point can switch clusters and ` +
          `no centroid can improve — that is a <b>local</b> optimum. Re-seed to see it can land somewhere different.`;
      } else if (iter > 0) {
        verdict.innerHTML = `Round <b>${iter}</b>: centroids last moved <b>${lastMoved.toFixed(3)}</b>. ` +
          `Keep stepping until the movement hits zero.`;
      }
    }

    // ---- animation loop (auto-run): one full round every ~0.6s ----
    let acc = 0;
    const loop = new Loop((dt) => {
      acc += dt;
      if (acc >= 0.55) { acc = 0; step(); }
      else draw(); // keep it lively even between steps
    });

    // ---- events ----
    stepBtn.addEventListener("click", () => { if (loop.running) { loop.stop(); runBtn.textContent = "Auto-run"; } step(); });
    runBtn.addEventListener("click", () => {
      if (settled) seedCentroids();
      loop.toggle();
      runBtn.textContent = loop.running ? "Pause" : "Auto-run";
    });
    reseedBtn.addEventListener("click", () => { loop.stop(); runBtn.textContent = "Auto-run"; seedCentroids(); });
    newDataBtn.addEventListener("click", () => { loop.stop(); runBtn.textContent = "Auto-run"; seedData(); seedCentroids(); });
    clearBtn.addEventListener("click", () => { loop.stop(); runBtn.textContent = "Auto-run"; points = []; seedCentroids(); });
    kSlider.input.addEventListener("input", () => { K = parseInt(kSlider.input.value, 10); loop.stop(); runBtn.textContent = "Auto-run"; seedCentroids(); });

    canvas.addEventListener("pointerdown", (e) => {
      const { x, y } = plot.eventToData(e);
      if (x < plot.xmin || x > plot.xmax || y < plot.ymin || y > plot.ymax) return;
      points.push({ x, y, c: -1 });
      settled = false; phase = "assign"; iter = 0;
      refresh();
    });

    // ---- init ----
    seedData();
    seedCentroids();

    return {
      onEnter() { /* stay paused until the user runs it */ draw(); },
      onLeave() { loop.stop(); runBtn.textContent = "Auto-run"; },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: grouping without labels</h3>
    <p>k-Means is the poster child for <b>unsupervised learning</b>. You hand it a pile of points with <em>no labels</em> and a single number, <code>k</code> — how many groups you think are there. It returns <code>k</code> cluster centres and a group for every point. Customer segments, colour palettes, document topics: anywhere you suspect there are "types" but nobody has labelled them.</p>
  </div>
  <div class="lesson-block">
    <h3>🔁 Lloyd's algorithm — two steps, repeated</h3>
    <p>The whole method is a loop you can watch above:</p>
    <ol class="steps">
      <li><b>Initialise.</b> Place <code>k</code> centroids (this game uses a k-means++ style spread so they don't start on top of each other).</li>
      <li><b>Assign.</b> Colour every point by whichever centroid is nearest — squared Euclidean distance.</li>
      <li><b>Update.</b> Move each centroid to the mean (centre of gravity) of the points that chose it.</li>
      <li><b>Repeat</b> assign → update until no point changes colour and the centroids stop moving.</li>
    </ol>
    <p>That's it. The dashed lines above are each centroid sliding from where it was (✕) to the new mean of its members.</p>
  </div>
  <div class="lesson-block">
    <h3>📉 What it's actually minimising</h3>
    <p>Each round can only lower — never raise — the <b>inertia</b>: the total squared distance from points to their own centroid.</p>
    <div class="formula">J = Σ<sub>i</sub> ‖ x<sub>i</sub> − μ<sub>(cluster of i)</sub> ‖²</div>
    <p>The <em>assign</em> step lowers J by giving each point the closest centre; the <em>update</em> step lowers J because the mean is the single point with the smallest squared distance to a set. Two descents that alternate — so J marches down and the loop must stop. Watch the "Inertia" readout fall and flatten.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ The catches</h3>
    <ul class="bullets">
      <li><b>Local optima.</b> Where it ends up depends on where the centroids start. Hit "Re-seed centroids" a few times — sometimes it finds a worse grouping. Real implementations run it several times and keep the lowest-inertia result.</li>
      <li><b>You must choose k.</b> The algorithm can't tell you how many groups exist. People use the "elbow" of the inertia curve or the silhouette score to guess.</li>
      <li><b>It likes round blobs.</b> Because it uses distance to a mean, k-Means assumes roughly spherical, similar-sized clusters. Long thin or nested shapes fool it.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> k-Means alternates "assign points to the nearest centre" and "move each centre to its points' average," and that alternation is guaranteed to keep shrinking the within-cluster spread until it settles.</p>
  </div>`,
};
