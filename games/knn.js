// =====================================================================
//  k-Nearest Neighbours — the laziest classifier in the room.
//
//  There is NO training step. The "model" is just the dataset itself.
//  To classify a query point we:
//     1. measure the (Euclidean) distance to every labelled point,
//     2. take the k closest of them,
//     3. let those k neighbours vote — the majority label wins.
//
//  The decision boundary is drawn by running that same vote at every
//  cell of a background grid. Because that is expensive, we render it
//  once to an OFFSCREEN canvas and only recompute when the data or k
//  changes — never on a plain mouse-move.
// =====================================================================
import {
  el, ghostButton, slider, Plot,
  blobs, dist, dist2, classColor, withAlpha, clamp,
} from "../shared.js";

export default {
  key: "knn",
  name: "k-Nearest Neighbours",
  icon: "👥",
  category: "Classification",
  tag: "Lazy",
  blurb: "Move a query point and let its k closest neighbours vote on its label.",
  eyebrow: "Classification · Supervised",
  title: 'k-Nearest Neighbours: <span class="grad">ask the neighbours</span>',
  lede:
    "The laziest algorithm here — it does no training at all. To classify a new point it just looks at the <em>k</em> labelled points nearest to it and takes a vote. Drag the query point around, change k, and watch the decision boundary breathe between jagged and smooth.",

  mount(root) {
    // ---- state ----
    const NCLASSES = 3;
    let K = 5;
    let points = [];                 // { x, y, label }
    let query = { x: 0.1, y: 0.15 }; // the movable point being classified
    let addClass = 0;                // which class a canvas click drops
    let dragging = false;

    // ---- DOM ----
    const canvas = el("canvas", {
      width: 760, height: 460, role: "img",
      "aria-label": "Scatter plot of labelled points with a movable query marker; the background is tinted by the predicted class.",
    });
    const plot = new Plot(canvas, { xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2, pad: 16 });

    // offscreen canvas holds the cached decision-region shading
    const regionCanvas = document.createElement("canvas");
    regionCanvas.width = canvas.width;
    regionCanvas.height = canvas.height;
    const regionCtx = regionCanvas.getContext("2d");

    const kSlider = slider({ min: 1, max: 15, step: 1, value: K, label: "Neighbours k =", format: (v) => v });

    const classBtns = [0, 1, 2].map((i) =>
      el("button", { class: i === addClass ? "active" : "", dataset: { cls: i } }, `Class ${i + 1}`));
    const classSeg = el("div", { class: "seg-group" }, ...classBtns);

    const newDataBtn = ghostButton("New data");
    const clearBtn = ghostButton("Clear");

    const predEl = el("span", { class: "v" }, "—");
    const kEl = el("span", { class: "v" }, String(K));
    const distEl = el("span", { class: "v" }, "—");

    const bars = el("div", { class: "bars" });
    const legend = el("div", { class: "legend" });
    const verdict = el("p", { class: "verdict" },
      "Drag the white ring around the canvas. Its k nearest neighbours light up and vote on its colour.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" },
            "Drag the white ring to classify it. Click empty space to add a labelled point of the selected class.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Prediction"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Neighbours k"), kEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Predicted class"), predEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Nearest distance"), distEl),
            el("h3", { style: { marginTop: "1rem" } }, "Vote tally"),
            bars),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, kSlider.wrap),
            el("div", { class: "controls" }, el("span", { class: "toggle", style: { cursor: "default" } }, "Add as"), classSeg),
            el("div", { class: "controls" }, newDataBtn, clearBtn)),
          legend)),
      verdict
    );

    // ---- core k-NN ----------------------------------------------------
    // Return { winner, counts, neighbours } for a point (qx, qy).
    // neighbours is the list of the k closest dataset points (by distance).
    function classify(qx, qy) {
      const counts = new Array(NCLASSES).fill(0);
      if (!points.length) return { winner: -1, counts, neighbours: [] };
      // squared distance to every point (sqrt is monotonic, so squared
      // distance gives the same ordering and is cheaper).
      const scored = points.map((p) => ({ p, d2: dist2(qx, qy, p.x, p.y) }));
      scored.sort((a, b) => a.d2 - b.d2);
      const k = Math.min(K, scored.length);
      const neighbours = scored.slice(0, k);
      for (const n of neighbours) counts[n.p.label]++;
      // majority vote; ties broken toward the lower class index, then
      // (implicitly) toward the closer neighbour because we scanned the
      // sorted list — a deterministic, defensible tie-break.
      let winner = 0, best = -1;
      for (let c = 0; c < NCLASSES; c++) {
        if (counts[c] > best) { best = counts[c]; winner = c; }
      }
      return { winner, counts, neighbours };
    }

    // Light-weight winner-only classification for the background grid.
    function winnerAt(qx, qy) {
      if (!points.length) return -1;
      // partial selection of k smallest distances without a full sort:
      // keep a small array of the current k best (k is tiny, ≤15).
      const k = Math.min(K, points.length);
      const bestD = new Array(k).fill(Infinity);
      const bestL = new Array(k).fill(-1);
      for (const p of points) {
        const d = dist2(qx, qy, p.x, p.y);
        // find the worst of the current best and replace if d beats it
        let worstIdx = 0;
        for (let i = 1; i < k; i++) if (bestD[i] > bestD[worstIdx]) worstIdx = i;
        if (d < bestD[worstIdx]) { bestD[worstIdx] = d; bestL[worstIdx] = p.label; }
      }
      const counts = new Array(NCLASSES).fill(0);
      for (const l of bestL) if (l >= 0) counts[l]++;
      let winner = 0, best = -1;
      for (let c = 0; c < NCLASSES; c++) if (counts[c] > best) { best = counts[c]; winner = c; }
      return winner;
    }

    // ---- decision-region cache ---------------------------------------
    // Recompute ONLY when the dataset or k changes. We shade an offscreen
    // canvas cell-by-cell, then drawImage it each frame.
    function recomputeRegions() {
      regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height);
      if (!points.length) return;
      const cols = 76;                                   // grid resolution
      const rows = Math.round(cols * (plot.plotH / plot.plotW));
      const cw = plot.plotW / cols;
      const ch = plot.plotH / rows;
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const px = plot.padL + (i + 0.5) * cw;
          const py = plot.padT + (j + 0.5) * ch;
          const w = winnerAt(plot.ix(px), plot.iy(py));
          if (w < 0) continue;
          regionCtx.fillStyle = withAlpha(classColor(w), 0.16);
          regionCtx.fillRect(plot.padL + i * cw, plot.padT + j * ch, cw + 1, ch + 1);
        }
      }
    }

    // ---- draw (cheap; runs on every interaction) ----------------------
    function draw() {
      plot.clear("#070d1a");
      // cached regions first
      plot.ctx.drawImage(regionCanvas, 0, 0);
      plot.grid(0.5, "rgba(255,255,255,0.05)");

      const res = classify(query.x, query.y);

      // lines + rings to the k chosen neighbours (drawn under the points)
      for (const n of res.neighbours) {
        plot.line(query.x, query.y, n.p.x, n.p.y, withAlpha(classColor(n.p.label), 0.55), 1.5);
      }

      // all dataset points
      const chosen = new Set(res.neighbours.map((n) => n.p));
      for (const p of points) {
        plot.point(p.x, p.y, 4.5, classColor(p.label), "rgba(7,13,26,0.8)", 1);
        if (chosen.has(p)) {
          // halo ring around the voting neighbours
          const cx = plot.sx(p.x), cy = plot.sy(p.y);
          plot.ctx.beginPath();
          plot.ctx.arc(cx, cy, 8.5, 0, Math.PI * 2);
          plot.ctx.lineWidth = 2;
          plot.ctx.strokeStyle = withAlpha(classColor(p.label), 0.95);
          plot.ctx.stroke();
        }
      }

      // the query marker — a distinct white ring tinted by its prediction
      const qx = plot.sx(query.x), qy = plot.sy(query.y);
      const predCol = res.winner < 0 ? "#7c89b8" : classColor(res.winner);
      plot.ctx.beginPath();
      plot.ctx.arc(qx, qy, 11, 0, Math.PI * 2);
      plot.ctx.fillStyle = withAlpha(predCol, 0.25);
      plot.ctx.fill();
      plot.ctx.lineWidth = 3;
      plot.ctx.strokeStyle = "#ffffff";
      plot.ctx.stroke();
      plot.ctx.beginPath();
      plot.ctx.arc(qx, qy, 3, 0, Math.PI * 2);
      plot.ctx.fillStyle = "#ffffff";
      plot.ctx.fill();

      return res;
    }

    // ---- refresh (draw + side panel + verdict) ------------------------
    const CLASS_NAME = ["cyan", "coral", "violet"];

    function refresh() {
      const res = draw();
      kEl.textContent = String(K);

      const total = res.counts.reduce((s, v) => s + v, 0);
      const winnerCount = res.winner >= 0 ? res.counts[res.winner] : 0;

      predEl.textContent = res.winner < 0 ? "—" : `Class ${res.winner + 1} (${CLASS_NAME[res.winner]})`;
      predEl.style.color = res.winner < 0 ? "" : classColor(res.winner);
      distEl.textContent = res.neighbours.length
        ? dist(query.x, query.y, res.neighbours[0].p.x, res.neighbours[0].p.y).toFixed(3)
        : "—";

      // vote-tally bars
      bars.replaceChildren();
      for (let c = 0; c < NCLASSES; c++) {
        const n = res.counts[c];
        const pct = total ? (n / total) * 100 : 0;
        const track = el("div", { class: "bar-track" },
          el("div", { class: "bar-fill", style: { width: pct + "%", background: classColor(c) } }));
        bars.append(el("div", { class: "bar-row" },
          el("span", { class: "bl", style: { color: classColor(c) } }, `Class ${c + 1}`),
          track,
          el("span", { class: "bar-val" }, `${n}/${total || 0}`)));
      }

      // legend
      legend.replaceChildren();
      for (let c = 0; c < NCLASSES; c++) {
        const n = points.filter((p) => p.label === c).length;
        legend.append(el("span", { class: "legend-item" },
          el("span", { class: "legend-dot", style: { background: classColor(c) } }),
          `Class ${c + 1} · ${n} pts`));
      }

      // verdict in words
      if (!points.length) {
        verdict.innerHTML = "No data yet — click the canvas to add labelled points, then drag the white ring to classify it.";
      } else {
        // check for a tie at the winning count
        const tied = res.counts.filter((v) => v === winnerCount).length > 1;
        const tieNote = tied
          ? ` It is a <b>tie</b> at the top — broken toward the lower class index, so the boundary here is fragile.`
          : "";
        verdict.innerHTML =
          `<b>${winnerCount}</b> of the ${total} nearest are <b>${CLASS_NAME[res.winner]}</b>, ` +
          `so the query is classified <b>Class ${res.winner + 1}</b>.${tieNote}`;
      }
    }

    // ---- data seeding -------------------------------------------------
    function seedData() {
      points = blobs([[-0.5, 0.4], [0.5, 0.5], [0, -0.5]], 18, 0.16)
        .map((p) => ({ x: p.x, y: p.y, label: p.label }));
      query = { x: 0.1, y: 0.15 };
      recomputeRegions();
      refresh();
    }

    // ---- events -------------------------------------------------------
    kSlider.input.addEventListener("input", () => {
      K = parseInt(kSlider.input.value, 10);
      recomputeRegions();   // k changed → regions change
      refresh();
    });

    classBtns.forEach((b) => b.addEventListener("click", () => {
      addClass = parseInt(b.dataset.cls, 10);
      classBtns.forEach((x, i) => x.classList.toggle("active", i === addClass));
    }));

    newDataBtn.addEventListener("click", () => seedData());
    clearBtn.addEventListener("click", () => { points = []; recomputeRegions(); refresh(); });

    // distance (in pixels) from a pointer to the query marker
    function nearQuery(d) {
      const dx = plot.sx(query.x) - plot.sx(d.x);
      const dy = plot.sy(query.y) - plot.sy(d.y);
      return Math.hypot(dx, dy) <= 16;
    }

    canvas.addEventListener("pointerdown", (e) => {
      const d = plot.eventToData(e);
      if (d.x < plot.xmin || d.x > plot.xmax || d.y < plot.ymin || d.y > plot.ymax) return;
      if (nearQuery(d)) {
        // begin dragging the query point
        dragging = true;
        canvas.setPointerCapture?.(e.pointerId);
        e.preventDefault();
        return;
      }
      // otherwise: add a labelled point of the selected class
      points.push({ x: clamp(d.x, plot.xmin, plot.xmax), y: clamp(d.y, plot.ymin, plot.ymax), label: addClass });
      recomputeRegions();   // dataset changed → regions change
      refresh();
    }, { passive: false });

    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      e.preventDefault();
      const d = plot.eventToData(e);
      query.x = clamp(d.x, plot.xmin, plot.xmax);
      query.y = clamp(d.y, plot.ymin, plot.ymax);
      refresh();            // cheap — no region recompute on move
    }, { passive: false });

    const endDrag = (e) => {
      if (!dragging) return;
      dragging = false;
      canvas.releasePointerCapture?.(e.pointerId);
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // ---- init ----
    seedData();

    return {
      onEnter() { recomputeRegions(); refresh(); },
      onLeave() { dragging = false; },  // no Loop to stop in this game
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: classify by company kept</h3>
    <p>k-Nearest Neighbours answers one question: <em>"given a new point, what label should it have?"</em> Its whole philosophy is that points close together in feature space tend to share a label — so to label something new, just look at the labelled examples nearest to it and copy the majority. Recommending similar movies, spotting handwritten digits, flagging anomalies: anywhere "similar things behave similarly" holds.</p>
  </div>
  <div class="lesson-block">
    <h3>😴 No training — it just memorises ("lazy" / instance-based)</h3>
    <p>Almost every other model spends effort up front <em>fitting</em> parameters, then throws the data away. k-NN does the opposite: <b>training is just storing the dataset.</b> All the work happens at <em>predict</em> time. That is why it is called a <b>lazy</b> or <b>instance-based</b> learner — there is no boundary equation, no weights, nothing learned. The decision boundary you see shaded above is not stored anywhere; it <em>emerges</em> from the data every time you ask.</p>
  </div>
  <div class="lesson-block">
    <h3>📐 The steps you can see</h3>
    <ol class="steps">
      <li><b>Measure distance.</b> From the query (white ring) to every labelled point — straight-line Euclidean distance.</li>
      <li><b>Pick the k closest.</b> Those are ringed and joined to the query by a line.</li>
      <li><b>Vote.</b> Count the labels among those k. The majority wins — watch the tally bars.</li>
      <li><b>Paint the map.</b> Run that same vote at every background cell and you get the decision regions. Drag k and watch them redraw.</li>
    </ol>
  </div>
  <div class="lesson-block">
    <h3>🧮 The only maths it needs</h3>
    <p>Distance between the query <b>q</b> and a stored point <b>x</b> in 2-D is plain Pythagoras:</p>
    <div class="formula">d(q, x) = √[ (q<sub>1</sub> − x<sub>1</sub>)<sup>2</sup> + (q<sub>2</sub> − x<sub>2</sub>)<sup>2</sup> ]</div>
    <p>(We actually compare <em>squared</em> distances — sorting by d² gives the same order and skips the square root.) Then the prediction is just the most common label among the k closest:</p>
    <div class="formula">ŷ(q) = mode{ label(x) : x ∈ k-nearest(q) }</div>
    <p>Set <code>k = 1</code> above and every point owns a little island of its own colour — the boundary turns jagged and wraps around lone outliers. Crank <code>k</code> up and the regions fuse into smooth blobs. <b>k is a smoothing knob:</b> small k = low bias, high variance (overfit, noise-sensitive); large k = high bias, low variance (oversmoothed, ignores local structure). The sweet spot is somewhere in between.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ Where it bites</h3>
    <ul class="bullets">
      <li><b>k = 1 overfits.</b> A single mislabelled point creates its own wrong island. Try it: set k to 1 and drag the query near a stray point.</li>
      <li><b>Ties.</b> With an even split (e.g. k = 4 giving 2 vs 2) the vote is ambiguous; people use odd k, distance-weighted votes, or a fixed tie-break. The verdict above warns you when a tie occurs.</li>
      <li><b>Slow at predict time.</b> Every prediction must scan <em>all</em> stored points — O(n) per query, O(n·d) in d dimensions. A million points means a million distances for every single answer. Real systems use KD-trees, ball-trees, or approximate nearest-neighbour indexes to cheat.</li>
      <li><b>The curse of dimensionality.</b> In high dimensions distances concentrate — everything becomes roughly equidistant, so "nearest" stops meaning much. k-NN quietly stops working well past a handful of informative features unless you reduce dimensions first.</li>
      <li><b>Scale matters.</b> Euclidean distance treats every axis equally, so a feature measured in thousands will drown one measured in fractions. Always standardise your features first.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> k-NN doesn't learn a model — it stores every example and, to classify a new point, lets its k nearest neighbours vote; k is the dial between jagged overfitting and oversmoothed blur.</p>
  </div>`,
};
