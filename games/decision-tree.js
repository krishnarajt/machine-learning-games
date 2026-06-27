// =====================================================================
//  Decision Tree (CART) — greedy axis-aligned splits.
//
//  A binary tree is grown top-down. Each node holds a subset of the
//  training points and the bounding box it "owns" inside the window.
//  To split a node we try BOTH axes; for each axis we sort the points
//  and test every midpoint between consecutive distinct values as a
//  candidate threshold, scoring each by the weighted Gini impurity of
//  the two children. The split that minimises that impurity wins.
//  Leaves predict their majority class and paint their rectangle.
// =====================================================================
import {
  el, button, ghostButton, slider, Plot,
  blobs, xor, circles, classColor, withAlpha, clamp,
} from "../shared.js";

export default {
  key: "decision-tree",
  name: "Decision Tree",
  icon: "🌳",
  category: "Classification",
  tag: "CART",
  blurb: "Watch greedy yes/no splits carve the plane into pure rectangles.",
  eyebrow: "Classification · Supervised",
  title: 'Decision Tree: <span class="grad">twenty questions</span> for data',
  lede:
    "A decision tree classifies by asking a series of simple yes/no questions — <em>is x &gt; 0.3?</em> — " +
    "each one splitting the data to make the groups purer. Grow it one split at a time and watch the plane " +
    "get chopped into boxes, then see how more depth means sharper — and more overfit — boundaries.",

  mount(root) {
    // ---- window / state ----
    const WIN = { xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2 };
    let dataset = "blobs";
    let points = [];          // { x, y, label }
    let maxDepth = 4;         // the slider value (the cap on depth)
    let effDepth = 4;         // the depth limit actually used to build (for "grow one level")
    let tree = null;          // root node
    let nClasses = 3;

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Scatter plot of labelled points with the plane partitioned into coloured rectangles by a decision tree." });
    const plot = new Plot(canvas, { ...WIN, pad: 16 });

    const growBtn = button("Grow one level ▸");
    const fullBtn = button("Build full");
    const resetBtn = ghostButton("Reset");
    const depthSlider = slider({ min: 1, max: 7, step: 1, value: maxDepth, label: "max depth =", format: (v) => v });

    // dataset segmented control
    const segDefs = [["blobs", "Blobs"], ["xor", "XOR"], ["circles", "Circles"]];
    const segButtons = {};
    const seg = el("div", { class: "seg-group" },
      segDefs.map(([k, lbl]) => {
        const b = el("button", { class: k === dataset ? "active" : "" }, lbl);
        b.addEventListener("click", () => setDataset(k));
        segButtons[k] = b;
        return b;
      }));

    const depthEl = el("span", { class: "v" }, "—");
    const leavesEl = el("span", { class: "v" }, "—");
    const accEl = el("span", { class: "v" }, "—");
    const giniEl = el("span", { class: "v" }, "—");

    const treeBox = el("div", { class: "tree-dom",
      style: { font: '0.8rem "JetBrains Mono", monospace', color: "#d4daee", lineHeight: "1.5", overflowX: "auto" } });

    const legend = el("div", { class: "legend" });
    const verdict = el("p", { class: "verdict" },
      "Press <b>Grow one level</b> to add a layer of questions, or <b>Build full</b> to grow to the depth cap. Watch the plane split into boxes.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" }, "Filled boxes are leaf regions (coloured by the predicted class). Bright lines are the split thresholds. Dots are the true labels.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Tree state"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Depth used"), depthEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Leaves"), leavesEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Train accuracy"), accEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Root Gini"), giniEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, growBtn, fullBtn),
            el("div", { class: "controls" }, depthSlider.wrap),
            el("div", { class: "controls" }, el("span", { class: "toggle", style: { gap: "0.6rem" } }, "Dataset", seg)),
            el("div", { class: "controls" }, resetBtn)),
          el("div", { class: "panel" },
            el("h3", {}, "The tree"),
            treeBox),
          legend)),
      verdict
    );

    // =================================================================
    //  CART algorithm
    // =================================================================

    // Gini impurity of a set of points: 1 - Σ p_c²
    function gini(pts) {
      if (!pts.length) return 0;
      const counts = new Array(nClasses).fill(0);
      for (const p of pts) counts[p.label]++;
      let g = 1;
      for (const c of counts) { const p = c / pts.length; g -= p * p; }
      return g;
    }

    function majority(pts) {
      const counts = new Array(nClasses).fill(0);
      for (const p of pts) counts[p.label]++;
      let best = 0, bestC = -1;
      for (let c = 0; c < nClasses; c++) if (counts[c] > bestC) { bestC = counts[c]; best = c; }
      return best;
    }

    // Find the best (axis, threshold) minimising weighted child Gini.
    // axis 0 = split on x, axis 1 = split on y.
    function bestSplit(pts) {
      const n = pts.length;
      let best = null;            // { axis, thr, score, leftIdx... }
      let bestScore = Infinity;
      for (let axis = 0; axis < 2; axis++) {
        const key = axis === 0 ? "x" : "y";
        // sort points by this coordinate
        const sorted = pts.slice().sort((a, b) => a[key] - b[key]);
        // candidate thresholds = midpoints between consecutive DISTINCT values
        for (let i = 0; i < n - 1; i++) {
          const v0 = sorted[i][key], v1 = sorted[i + 1][key];
          if (v1 - v0 < 1e-9) continue;          // identical coord, no split here
          const thr = (v0 + v1) / 2;
          // left = first i+1 points, right = rest (because sorted)
          const left = sorted.slice(0, i + 1);
          const right = sorted.slice(i + 1);
          const score = (left.length * gini(left) + right.length * gini(right)) / n;
          if (score < bestScore - 1e-12) {
            bestScore = score;
            best = { axis, thr, score, left, right };
          }
        }
      }
      return best;
    }

    // Recursively build the tree. box = { xmin, xmax, ymin, ymax } the node owns.
    function build(pts, box, depth, depthLimit) {
      const node = { pts, box, gini: gini(pts), majority: majority(pts), depth };
      // stop conditions: pure, hit depth limit, too few points
      const pure = node.gini < 1e-9;
      if (pure || depth >= depthLimit || pts.length < 2) {
        node.leaf = true;
        return node;
      }
      const split = bestSplit(pts);
      // a split that doesn't actually reduce impurity (or none found) → leaf
      if (!split || split.score >= node.gini - 1e-12) {
        node.leaf = true;
        return node;
      }
      node.leaf = false;
      node.axis = split.axis;
      node.thr = split.thr;
      // carve the box for each child
      const lbox = { ...box }, rbox = { ...box };
      if (split.axis === 0) { lbox.xmax = split.thr; rbox.xmin = split.thr; }
      else { lbox.ymax = split.thr; rbox.ymin = split.thr; }
      node.left = build(split.left, lbox, depth + 1, depthLimit);   // ≤ thr
      node.right = build(split.right, rbox, depth + 1, depthLimit); // > thr
      return node;
    }

    function rebuild() {
      tree = build(points, { ...WIN }, 0, effDepth);
      refresh();
    }

    // ---- tree queries ----
    function leaves(node, acc = []) {
      if (!node) return acc;
      if (node.leaf) acc.push(node);
      else { leaves(node.left, acc); leaves(node.right, acc); }
      return acc;
    }
    function usedDepth(node) {
      if (!node || node.leaf) return node ? node.depth : 0;
      return Math.max(usedDepth(node.left), usedDepth(node.right));
    }
    function predict(node, p) {
      if (node.leaf) return node.majority;
      const v = node.axis === 0 ? p.x : p.y;
      return v <= node.thr ? predict(node.left, p) : predict(node.right, p);
    }
    function accuracy() {
      if (!points.length || !tree) return 0;
      let ok = 0;
      for (const p of points) if (predict(tree, p) === p.label) ok++;
      return ok / points.length;
    }

    // =================================================================
    //  Drawing
    // =================================================================
    function drawNodeSplits(node) {
      if (!node || node.leaf) return;
      const c = plot.ctx;
      const b = node.box;
      // bright split segment, clipped to this node's box
      c.save();
      c.strokeStyle = "rgba(234,238,251,0.85)";
      c.lineWidth = 2;
      c.beginPath();
      if (node.axis === 0) {
        const px = plot.sx(node.thr);
        c.moveTo(px, plot.sy(b.ymin)); c.lineTo(px, plot.sy(b.ymax));
      } else {
        const py = plot.sy(node.thr);
        c.moveTo(plot.sx(b.xmin), py); c.lineTo(plot.sx(b.xmax), py);
      }
      c.stroke();
      c.restore();
      drawNodeSplits(node.left);
      drawNodeSplits(node.right);
    }

    function draw() {
      plot.clear("#070d1a");

      if (tree) {
        // fill each leaf's rectangle with its predicted-class colour
        for (const lf of leaves(tree)) {
          if (!lf.pts.length) continue;
          const b = lf.box;
          const x0 = plot.sx(clamp(b.xmin, WIN.xmin, WIN.xmax));
          const x1 = plot.sx(clamp(b.xmax, WIN.xmin, WIN.xmax));
          const y0 = plot.sy(clamp(b.ymax, WIN.ymin, WIN.ymax)); // ymax → top pixel
          const y1 = plot.sy(clamp(b.ymin, WIN.ymin, WIN.ymax)); // ymin → bottom pixel
          plot.ctx.fillStyle = withAlpha(classColor(lf.majority), 0.18);
          plot.ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        }
        // split lines on top of the fills
        drawNodeSplits(tree);
      }

      plot.grid(0.5);

      // data points coloured by TRUE label
      for (const p of points) {
        plot.point(p.x, p.y, 4.5, classColor(p.label), "rgba(7,13,26,0.85)", 1.2);
      }
    }

    // ---- nested DOM rendering of the tree ----
    function renderTreeDOM(node) {
      if (!node) return el("div", {}, "—");
      if (node.leaf) {
        return el("div", { style: { padding: "1px 0" } },
          el("span", { style: { color: classColor(node.majority), fontWeight: "700" } }, "■ "),
          `leaf → class ${node.majority + 1} (${node.pts.length} pts)`);
      }
      const axis = node.axis === 0 ? "x" : "y";
      const q = `${axis} > ${node.thr.toFixed(2)} ?`;
      return el("div", { style: { padding: "1px 0" } },
        el("div", { style: { color: "#aee9ff" } }, q),
        el("div", { style: { borderLeft: "1px solid rgba(255,255,255,0.14)", marginLeft: "6px", paddingLeft: "10px" } },
          el("div", { style: { color: "#9aa4c6", fontSize: "0.72rem" } }, "no (≤):"),
          renderTreeDOM(node.left),
          el("div", { style: { color: "#9aa4c6", fontSize: "0.72rem", marginTop: "3px" } }, "yes (>):"),
          renderTreeDOM(node.right)));
    }

    function refresh() {
      draw();

      const lvs = tree ? leaves(tree) : [];
      const d = tree ? usedDepth(tree) : 0;
      const acc = accuracy();
      depthEl.textContent = tree ? `${d} / ${effDepth}` : "—";
      leavesEl.textContent = lvs.length || "—";
      accEl.textContent = points.length ? `${(acc * 100).toFixed(1)}%` : "—";
      giniEl.textContent = tree ? tree.gini.toFixed(3) : "—";

      treeBox.replaceChildren(renderTreeDOM(tree));

      legend.replaceChildren();
      for (let c = 0; c < nClasses; c++) {
        legend.append(el("span", { class: "legend-item" },
          el("span", { class: "legend-dot", style: { background: classColor(c) } }),
          `Class ${c + 1}`));
      }

      // verdict
      if (!tree) {
        verdict.innerHTML = "Build a tree to begin.";
      } else if (d === 0) {
        verdict.innerHTML =
          `Depth 0: one box, predicting the single most common class — accuracy <b>${(acc * 100).toFixed(1)}%</b>. ` +
          `Grow it to start asking questions.`;
      } else {
        const pure = lvs.every((l) => l.gini < 1e-9 || l.pts.length < 2);
        const deep = effDepth >= 6;
        let tail;
        if (acc > 0.985) {
          tail = deep
            ? `It now classifies almost every training point correctly — but look at the thin islands: a deep tree is <b>memorising noise</b>, not learning the shape. That is overfitting.`
            : `Nearly every training point is correct. Crank the depth higher and watch boxes shatter into tiny islands.`;
        } else if (pure) {
          tail = `Most leaves are pure. Increase the depth cap to split the rest.`;
        } else {
          tail = `Each new level adds a question that makes the boxes purer. Keep growing or push the depth cap higher.`;
        }
        verdict.innerHTML =
          `Depth <b>${d}</b> · <b>${lvs.length}</b> leaves · train accuracy <b>${(acc * 100).toFixed(1)}%</b>. ${tail}`;
      }
    }

    // =================================================================
    //  Data
    // =================================================================
    function seedData() {
      if (dataset === "blobs") {
        // 3 well-separated blobs of 3 classes
        points = blobs([[-0.6, 0.55], [0.6, 0.6], [0.0, -0.6]], 32, 0.14)
          .map((p) => ({ x: p.x, y: p.y, label: p.label }));
        nClasses = 3;
      } else if (dataset === "xor") {
        points = xor(150, 0.11).map((p) => ({ x: p.x, y: p.y, label: p.label }));
        nClasses = 2;
      } else {
        points = circles(150, 0.06).map((p) => ({ x: p.x, y: p.y, label: p.label }));
        nClasses = 2;
      }
    }

    function setDataset(k) {
      dataset = k;
      for (const key in segButtons) segButtons[key].classList.toggle("active", key === k);
      maxDepth = parseInt(depthSlider.input.value, 10);
      effDepth = maxDepth;
      seedData();
      rebuild();
    }

    // =================================================================
    //  Events
    // =================================================================
    growBtn.addEventListener("click", () => {
      // advance the build limit by one, measured from whichever is deeper:
      // the depth currently used or the current limit. This never goes backward
      // and always climbs toward the cap (clamped to maxDepth).
      const base = tree ? Math.max(usedDepth(tree), effDepth) : effDepth;
      effDepth = Math.min(maxDepth, base + 1);
      rebuild();
    });
    fullBtn.addEventListener("click", () => { effDepth = maxDepth; rebuild(); });
    resetBtn.addEventListener("click", () => { effDepth = 0; rebuild(); });
    depthSlider.input.addEventListener("input", () => {
      maxDepth = parseInt(depthSlider.input.value, 10);
      // keep the effective build depth within the new cap
      effDepth = Math.min(effDepth, maxDepth);
      rebuild();
    });

    // ---- init ----
    seedData();
    effDepth = maxDepth;
    rebuild();

    return {
      onEnter() { draw(); },
      onLeave() { /* no Loop running */ },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: classify by a chain of yes/no questions</h3>
    <p>A decision tree is exactly the game of <b>twenty questions</b>. To label a new point it asks one simple question — <em>is x &gt; 0.31?</em> — follows the matching branch, asks the next question there, and keeps going until it hits a <b>leaf</b>, which just votes for the most common class it saw. Each question is a single axis-aligned cut, so the tree is human-readable: you can literally trace the path. CART (Classification And Regression Trees) is the standard recipe for growing one.</p>
  </div>
  <div class="lesson-block">
    <h3>🔪 What you can see: greedy top-down splitting</h3>
    <ol class="steps">
      <li><b>Start.</b> The root holds every point in one box and predicts the majority class.</li>
      <li><b>Score every cut.</b> For each axis, sort the points and try a threshold between each pair of neighbours. Each candidate splits the box into a left and right child.</li>
      <li><b>Pick the purest.</b> Keep the single cut whose two children are the <em>purest</em> (lowest combined impurity). Draw it as a bright line.</li>
      <li><b>Recurse.</b> Repeat inside each child box. Stop when a box is pure, you hit the depth cap, or too few points remain.</li>
    </ol>
    <p>"Grow one level" performs one more round of this; "Build full" runs it to the depth cap. The plane fills with coloured rectangles — each is one leaf's prediction.</p>
  </div>
  <div class="lesson-block">
    <h3>📉 The maths: impurity and information gain</h3>
    <p>A split is good if it makes the groups <b>purer</b>. CART measures purity with the <b>Gini impurity</b> of a set — the chance you'd mislabel a random point if you guessed by the set's class proportions:</p>
    <div class="formula">Gini(S) = 1 − Σ<sub>c</sub> p<sub>c</sub><sup>2</sup></div>
    <p>It is 0 when every point shares one class and rises as the mix gets even. A candidate split is scored by the impurity it <em>removes</em> — the <b>information gain</b>, the parent's impurity minus the size-weighted impurity of its children:</p>
    <div class="formula">ΔGini = Gini(S) − ( |S<sub>L</sub>|/|S| · Gini(S<sub>L</sub>) + |S<sub>R</sub>|/|S| · Gini(S<sub>R</sub>) )</div>
    <p>The tree greedily takes the split with the biggest drop. (Some trees use <b>entropy</b>, −Σ p<sub>c</sub> log p<sub>c</sub>, instead of Gini — nearly the same boundaries, slightly more compute.) "Greedy" matters: it never revisits an earlier split, even if a worse first cut would have enabled a better tree later.</p>
  </div>
  <div class="lesson-block">
    <h3>📦 Why the boundaries are blocky</h3>
    <p>Every question compares one feature to a number, so every cut is parallel to an axis. The decision regions are therefore unions of <b>rectangles</b> — never a diagonal or a curve. On the <b>Circles</b> dataset you can watch the tree approximate a round boundary with a jagged staircase of boxes; a single diagonal line would do better, but the tree can only build steps.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ Overfitting, and how to fight it</h3>
    <ul class="bullets">
      <li><b>Deep trees memorise noise.</b> Push max depth to 6–7 and watch lonely islands appear around stray points. Training accuracy races toward 100%, but those islands describe individual samples, not the real pattern — they won't generalise.</li>
      <li><b>Pruning / max depth is the cure.</b> Capping depth (or requiring a minimum number of points per leaf, or pruning weak splits back afterward) trades a little training accuracy for a simpler, smoother boundary that holds up on new data.</li>
      <li><b>Instability.</b> A single tree is twitchy — move a few points and the top split can flip, redrawing everything below it.</li>
      <li><b>Ensembles fix this.</b> <b>Random forests</b> average many trees grown on random subsets of data and features; <b>gradient boosting</b> adds small trees that each correct the previous ones' mistakes. Averaging tames the variance and gives state-of-the-art results on tabular data.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> a decision tree greedily slices the feature space with axis-aligned cuts that each maximise the drop in Gini impurity — readable and powerful, but prone to overfit, so we cap its depth and average many of them.</p>
  </div>`,
};
