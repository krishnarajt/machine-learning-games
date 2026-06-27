// =====================================================================
//  Random Forest — the wisdom of crowds for decision trees.
//
//  One deep tree overfits (low bias, HIGH variance): its boundary is
//  jagged and unstable. A random forest fights the variance with a crowd:
//    • BAGGING        — each tree trains on a bootstrap resample (n draws
//                       WITH replacement) of the data, so trees disagree.
//    • FEATURE RANDOM — each split only considers a random subset of the
//                       features, which DECORRELATES the trees.
//    • VOTING         — the forest's prediction is the average over trees.
//  Averaging many decorrelated, overfit trees cancels their noise and
//  leaves a smooth, robust boundary. The ~37% of rows a tree never drew
//  (its out-of-bag set) give a free validation estimate (OOB accuracy).
//
//  Trees are grown incrementally so you watch the boundary settle as the
//  crowd grows.
// =====================================================================
import {
  el, ghostButton, slider, Loop, Plot,
  circles, twoMoons, xor, spiral,
  classColor, hexToRgb, lerp, clamp, randInt, range, mean,
} from "../shared.js";

export default {
  key: "random-forest",
  name: "Random Forest",
  icon: "🌲",
  category: "Classification",
  tag: "Ensemble",
  blurb: "Average a crowd of overfit trees into one smooth, robust boundary.",
  eyebrow: "Classification · Ensemble",
  title: 'Random Forest: the <span class="grad">wisdom of crowds</span>',
  lede:
    "One decision tree is twitchy — grow it deep and it memorises noise. A random forest fixes that with a crowd: " +
    "train many trees, each on a random resample of the data and a random subset of features, then let them vote. " +
    "The individual trees are jagged and overconfident; their <em>average</em> is smooth and reliable.",

  mount(root) {
    const WIN = { xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2 };
    const C0 = hexToRgb(classColor(0)); // class 0 (cyan)
    const C1 = hexToRgb(classColor(1)); // class 1 (coral)

    // ---- state ----
    let datasetKey = "circles";
    let data = [];            // { x, y, label(0/1) }
    let targetTrees = 40;
    let maxDepth = 7;         // deep so single trees visibly overfit
    let featureBag = true;    // consider a random subset of features per split
    let view = "forest";      // "forest" | "tree"
    let spotlight = 0;        // which individual tree to show in "tree" view
    let forest = [];          // { tree, inBag:Set<int> }

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Decision-boundary plot: the forest's averaged vote (smooth) or a single tree's jagged boundary, with the data points on top." });
    const plot = new Plot(canvas, { ...WIN, pad: 14 });

    const dsSeg = seg(["circles", "moons", "xor", "spiral"], ["Circles", "Moons", "XOR", "Spiral"], datasetKey, (k) => { datasetKey = k; newData(); });
    const viewSeg = seg(["forest", "tree"], ["Forest vote", "One tree"], view, (k) => { view = k; spotWrap.style.display = k === "tree" ? "" : "none"; draw(); refresh(); });

    const treesSlider = slider({ min: 1, max: 80, step: 1, value: targetTrees, label: "Trees =", format: (v) => v });
    const depthSlider = slider({ min: 2, max: 9, step: 1, value: maxDepth, label: "Max depth =", format: (v) => v });
    const spotSlider = slider({ min: 0, max: 79, step: 1, value: 0, label: "Show tree #", format: (v) => Number(v) + 1 });
    const spotWrap = el("div", { class: "controls", style: { display: "none" } }, spotSlider.wrap);

    const fbToggle = el("input", { type: "checkbox", checked: featureBag });
    const fbLabel = el("label", { class: "toggle" }, fbToggle, " Feature randomness (split on 1 random axis)");

    const regrowBtn = el("button", {}, "Regrow forest");
    const newBtn = ghostButton("New data");

    const treesEl = el("span", { class: "v" }, "0");
    const depthEl = el("span", { class: "v" }, maxDepth);
    const trainEl = el("span", { class: "v" }, "—");
    const oobEl = el("span", { class: "v" }, "—");

    const legend = el("div", { class: "legend" },
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { background: classColor(0) } }), "class A"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { background: classColor(1) } }), "class B"));
    const verdict = el("p", { class: "verdict" }, "Growing the forest…");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" }, "Shaded by the forest's vote — strong colour = confident, pale = a close call. Flip to “One tree” to see how jagged a single overfit tree is.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "The forest"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Trees grown"), treesEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Max depth"), depthEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Train accuracy"), trainEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Out-of-bag accuracy"), oobEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Data"),
            el("div", { class: "controls" }, dsSeg.node)),
          el("div", { class: "panel" },
            el("h3", {}, "View"),
            el("div", { class: "controls" }, viewSeg.node),
            spotWrap),
          el("div", { class: "panel" },
            el("h3", {}, "Forest settings"),
            el("div", { class: "controls" }, treesSlider.wrap),
            el("div", { class: "controls" }, depthSlider.wrap),
            el("div", { class: "controls" }, fbLabel),
            el("div", { class: "controls" }, regrowBtn, newBtn)),
          legend)),
      verdict
    );

    // ---- segmented control helper ----
    function seg(keys, labels, active, onPick) {
      const node = el("div", { class: "seg-group" });
      const btns = keys.map((k, i) => {
        const b = el("button", { class: k === active ? "active" : "" }, labels[i]);
        b.addEventListener("click", () => {
          btns.forEach((x) => x.classList.remove("active"));
          b.classList.add("active");
          onPick(k);
        });
        return b;
      });
      btns.forEach((b) => node.append(b));
      return { node, btns };
    }

    // ---------------------------------------------------------------
    //  CART tree with bagging + per-split feature subsampling
    // ---------------------------------------------------------------
    const giniOf = (pts) => {
      if (!pts.length) return 0;
      const p1 = mean(pts.map((p) => p.label));
      return 1 - p1 * p1 - (1 - p1) * (1 - p1);
    };

    function buildTree(pts, depth) {
      const node = { leaf: true, prob: pts.length ? mean(pts.map((p) => p.label)) : 0.5 };
      if (depth >= maxDepth || pts.length < 4) return node;
      const g0 = giniOf(pts);
      if (g0 < 1e-9) return node; // pure

      // which axes may we split on this time?
      const axes = featureBag ? [randInt(2) === 0 ? "x" : "y"] : ["x", "y"];
      let best = null;
      for (const ax of axes) {
        const sorted = [...pts].sort((a, b) => a[ax] - b[ax]);
        for (let i = 1; i < sorted.length; i++) {
          if (sorted[i][ax] === sorted[i - 1][ax]) continue;
          const thr = (sorted[i][ax] + sorted[i - 1][ax]) / 2;
          const L = sorted.slice(0, i), R = sorted.slice(i);
          const wg = (L.length * giniOf(L) + R.length * giniOf(R)) / pts.length;
          if (!best || wg < best.wg) best = { ax, thr, L, R, wg };
        }
      }
      if (!best || best.wg >= g0 - 1e-9) return node; // no useful split
      return { leaf: false, ax: best.ax, thr: best.thr, prob: mean(pts.map((p) => p.label)),
        left: buildTree(best.L, depth + 1), right: buildTree(best.R, depth + 1) };
    }

    const predictTree = (node, x, y) => {
      let n = node;
      while (!n.leaf) n = (n.ax === "x" ? x : y) < n.thr ? n.left : n.right;
      return n.prob; // P(class 1) at this leaf (0 or 1 for pure leaves)
    };

    function growOneTree() {
      const n = data.length;
      const idx = range(n).map(() => randInt(n)); // bootstrap: n draws with replacement
      const inBag = new Set(idx);
      const sample = idx.map((i) => data[i]);
      forest.push({ tree: buildTree(sample, 0), inBag });
    }

    // ---- ensemble prediction (average of tree votes) ----
    function ensembleProb(x, y) {
      if (!forest.length) return 0.5;
      let s = 0;
      for (const f of forest) s += predictTree(f.tree, x, y);
      return s / forest.length;
    }

    function metrics() {
      if (!forest.length || !data.length) return { train: null, oob: null };
      let trainOk = 0;
      let oobOk = 0, oobN = 0;
      for (let i = 0; i < data.length; i++) {
        const p = data[i];
        // training accuracy (full forest)
        if ((ensembleProb(p.x, p.y) >= 0.5 ? 1 : 0) === p.label) trainOk++;
        // out-of-bag: average only trees that did NOT draw row i
        let s = 0, c = 0;
        for (const f of forest) if (!f.inBag.has(i)) { s += predictTree(f.tree, p.x, p.y); c++; }
        if (c) { oobN++; if ((s / c >= 0.5 ? 1 : 0) === p.label) oobOk++; }
      }
      return { train: trainOk / data.length, oob: oobN ? oobOk / oobN : null };
    }

    // ---------------------------------------------------------------
    //  drawing
    // ---------------------------------------------------------------
    const fieldColor = (p) => {
      const conf = Math.abs(p - 0.5) * 2;           // 0 at the boundary, 1 when certain
      const r = Math.round(lerp(C0.r, C1.r, p)), g = Math.round(lerp(C0.g, C1.g, p)), b = Math.round(lerp(C0.b, C1.b, p));
      return `rgba(${r},${g},${b},${(0.12 + 0.5 * conf).toFixed(3)})`;
    };

    function draw() {
      plot.clear("#070d1a");
      if (forest.length) {
        const probeAt = view === "tree" && forest[spotlight]
          ? (x, y) => predictTree(forest[spotlight].tree, x, y)
          : (x, y) => ensembleProb(x, y);
        plot.shade((x, y) => fieldColor(probeAt(x, y)), 48);
      }
      plot.grid(0.5);
      for (const p of data) plot.point(p.x, p.y, 4.5, classColor(p.label), "rgba(7,13,26,0.85)", 1.2);
    }

    function refresh() {
      treesEl.textContent = forest.length;
      depthEl.textContent = maxDepth;
      const m = metrics();
      trainEl.textContent = m.train == null ? "—" : Math.round(m.train * 100) + "%";
      oobEl.textContent = m.oob == null ? "—" : Math.round(m.oob * 100) + "%";

      if (view === "tree") {
        verdict.innerHTML = `Showing tree <b>#${spotlight + 1}</b> on its own — note the <b>blocky, jagged</b> boundary and ` +
          `the overconfident hard edges. Each tree saw a different bootstrap sample, so every one looks different. ` +
          `Switch back to <b>Forest vote</b> to see the crowd average them out.`;
      } else if (m.train != null) {
        const gap = m.oob != null ? Math.round((m.train - m.oob) * 100) : null;
        verdict.innerHTML = `<b>${forest.length}</b> trees voting. The shaded field is their average — smoother and more ` +
          `confident in the middle of each class, hesitant near the border. Train accuracy <b>${Math.round(m.train * 100)}%</b>` +
          (m.oob != null ? `, and the honest <b>out-of-bag</b> estimate is <b>${Math.round(m.oob * 100)}%</b>${gap != null && gap > 8 ? " — the gap is the overfitting the crowd still couldn't fully cancel." : "."}` : ".");
      }
    }

    // ---------------------------------------------------------------
    //  incremental growth loop — watch the boundary settle
    // ---------------------------------------------------------------
    const loop = new Loop(() => {
      if (forest.length >= targetTrees) { loop.stop(); return; }
      growOneTree();
      draw();
      refresh();
    });

    function startGrowing() {
      if (forest.length < targetTrees && !loop.running) loop.start();
    }
    function rebuild() {
      loop.stop();
      forest = [];
      draw();
      refresh();
      startGrowing();
    }

    function newData() {
      const gens = {
        circles: () => circles(150, 0.07),
        moons: () => twoMoons(150, 0.1),
        xor: () => xor(170, 0.13),
        spiral: () => spiral(90, 0.05),
      };
      data = gens[datasetKey]().map((p) => ({ x: p.x, y: p.y, label: p.label }));
      rebuild();
    }

    // ---- events ----
    treesSlider.input.addEventListener("input", () => {
      targetTrees = parseInt(treesSlider.input.value, 10);
      if (forest.length > targetTrees) forest = forest.slice(0, targetTrees); // shrink instantly
      draw(); refresh();
      startGrowing(); // grow up to the new target
    });
    depthSlider.input.addEventListener("input", () => { maxDepth = parseInt(depthSlider.input.value, 10); rebuild(); });
    spotSlider.input.addEventListener("input", () => {
      spotlight = clamp(parseInt(spotSlider.input.value, 10), 0, Math.max(0, forest.length - 1));
      if (view === "tree") { draw(); refresh(); }
    });
    fbToggle.addEventListener("change", () => { featureBag = fbToggle.checked; rebuild(); });
    regrowBtn.addEventListener("click", rebuild);
    newBtn.addEventListener("click", newData);

    // ---- init ----
    newData();

    return {
      onEnter() { draw(); refresh(); startGrowing(); },
      onLeave() { loop.stop(); },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The problem: one tree is a loose cannon</h3>
    <p>A single deep decision tree has <b>low bias but high variance</b>. It can fit almost anything — which means it also fits the <em>noise</em>. Move one training point and the whole boundary can lurch. Flip to "One tree" above and step through them: each is blocky, overconfident, and quite different from its neighbours. Individually, none of them is trustworthy.</p>
  </div>
  <div class="lesson-block">
    <h3>🌲 The fix: build a diverse crowd and average it</h3>
    <p>A random forest trains hundreds of trees and lets them vote. Two deliberate injections of randomness make the trees <em>different</em> from each other — which is the whole point:</p>
    <ol class="steps">
      <li><b>Bagging (bootstrap aggregating).</b> Each tree is trained on a <em>bootstrap sample</em>: draw <code>n</code> rows from the data <em>with replacement</em>. On average each sample includes only ~63% of the unique rows, with some repeated — so every tree sees a slightly different world.</li>
      <li><b>Feature randomness.</b> At each split, the tree may only consider a random subset of the features (here, with two features, it often looks at just one). This stops every tree from leaning on the same dominant feature and makes them <em>decorrelated</em>.</li>
      <li><b>Vote.</b> To predict, average the trees (for classes, take the majority). The shaded field above is that average.</li>
    </ol>
  </div>
  <div class="lesson-block">
    <h3>📉 Why averaging works: variance falls, bias doesn't</h3>
    <p>If you average <code>T</code> estimators each with variance <code>σ²</code> and average pairwise correlation <code>ρ</code>, the variance of the average is:</p>
    <div class="formula">Var = ρ·σ² + (1 − ρ)·σ² / T</div>
    <p>The second term shrinks as you add trees — that's bagging. The first term is killed by lowering <code>ρ</code> — that's the feature randomness. Crucially, averaging leaves the <b>bias</b> alone, so you cancel the noise without making the model dumber. Slide "Trees" up and watch the boundary stop twitching and go smooth.</p>
  </div>
  <div class="lesson-block">
    <h3>🎁 A free validation set: out-of-bag error</h3>
    <p>Each tree never saw the ~37% of rows its bootstrap left out — that row's <b>out-of-bag</b> set. To score a row honestly, average only the trees that didn't train on it. Do this for every row and you get the <b>OOB accuracy</b> — a built-in estimate of how the forest generalises, with no separate validation split needed. It's the more trustworthy number in the panel.</p>
  </div>
  <div class="lesson-block">
    <h3>⚙️ Why it's a workhorse</h3>
    <ul class="bullets">
      <li>Strong accuracy with almost no tuning — it's many people's default first model.</li>
      <li>Handles non-linear boundaries and feature interactions for free (it's still trees underneath — see the <b>Decision Tree</b> game).</li>
      <li>Gives <b>feature importance</b> (how much each feature reduced impurity across the forest).</li>
      <li>Trade-off: slower and far less interpretable than one tree — you've swapped a readable flowchart for a committee. Boosting (e.g. gradient-boosted trees) is the other big tree-ensemble idea: build trees in <em>sequence</em>, each fixing the last one's mistakes.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> a random forest grows many overfit trees on random resamples and random feature subsets so their errors are independent, then averages them — cancelling variance into one smooth, robust boundary, and handing you a free out-of-bag accuracy estimate along the way.</p>
  </div>`,
};
