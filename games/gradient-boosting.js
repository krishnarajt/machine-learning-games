// =====================================================================
//  Gradient Boosting — learning from your mistakes, one tree at a time.
//
//  Random Forest builds many trees in PARALLEL and averages them to cut
//  variance. Boosting does the opposite: it builds shallow trees in
//  SEQUENCE, each one trained to fix the leftover errors (residuals) of
//  all the trees so far. That cuts BIAS.
//
//  For squared-error loss the residual r = y − F is exactly the negative
//  gradient of the loss, so each tree is a gradient-descent step in
//  function space:
//      F0(x) = mean(y)
//      r_i   = y_i − F_{m-1}(x_i)          (= −∂L/∂F)
//      h_m   = small CART regression tree fit to (x_i, r_i)
//      F_m   = F_{m-1} + ν · h_m           (ν = shrinkage / learning rate)
//
//  Kept strictly 1-D so the residual staircase is the whole story: the
//  prediction crawls toward the data stage by stage, and the little
//  vertical residual sticks visibly shrink.
// =====================================================================
import {
  el, ghostButton, slider, Loop, Plot,
  mean, clamp, range, randRange, gaussRandom, withAlpha, classColor,
} from "../shared.js";

export default {
  key: "gradient-boosting",
  name: "Gradient Boosting",
  icon: "🚀",
  category: "Regression",
  tag: "Boost",
  blurb: "Stack shallow trees in sequence, each one fixing the errors the last one left behind.",
  eyebrow: "Regression · Ensemble",
  title: 'Gradient Boosting: learning from your <span class="grad">mistakes</span>',
  lede:
    "Random Forest builds many trees in parallel and averages them. Boosting does the opposite: it builds trees " +
    "<em>one at a time</em>, and each new tree is trained to fix the leftover errors — the residuals — of all the trees so far. " +
    "Watch the prediction crawl toward the data, stage by stage.",

  mount(root) {
    const WIN = { xmin: -1.1, xmax: 1.1, ymin: -1.4, ymax: 1.4 };

    // ---- state ----
    let data = [];            // { x, y }  ~45 noisy 1-D points
    let nu = 0.3;             // learning rate / shrinkage
    let maxDepth = 1;         // stump..depth-3 regression trees
    let targetStages = 30;    // how many stages "Run" will build up to
    let F0 = 0;               // initial constant prediction = mean(y)
    let stages = [];          // array of fitted trees h_m (predict added with ν)
    const POINTS = 45;

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "1-D regression plot: noisy data points, the current piecewise-constant boosted prediction drawn as a staircase, and thin vertical residual lines from each point to the prediction." });
    const plot = new Plot(canvas, { ...WIN, padL: 30, padR: 16, padT: 18, padB: 26 });

    const runBtn = el("button", {}, "Run ▸");
    const addBtn = el("button", {}, "Add stage");
    const resetBtn = ghostButton("Reset");
    const newDataBtn = ghostButton("New data");

    const targetSlider = slider({ min: 1, max: 60, step: 1, value: targetStages, label: "Run to stage =", format: (v) => v });
    const lrSlider = slider({ min: 0.05, max: 1, step: 0.05, value: nu, label: "Learning rate ν =", format: (v) => Number(v).toFixed(2) });
    const depthSlider = slider({ min: 1, max: 3, step: 1, value: maxDepth, label: "Tree depth =", format: (v) => v });

    const stageEl = el("span", { class: "v" }, "0");
    const mseEl = el("span", { class: "v" }, "—");
    const lrEl = el("span", { class: "v" }, nu.toFixed(2));
    const depthEl = el("span", { class: "v" }, maxDepth);

    const legend = el("div", { class: "legend" },
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { background: classColor(0) } }), "data"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-line", style: { background: classColor(3) } }), "prediction Fₘ(x)"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-line", style: { background: withAlpha(classColor(1), 0.7) } }), "residuals"));

    const verdict = el("p", { class: "verdict" },
      "Stage 0 is just the flat mean of the data. Press <b>Add stage</b> to fit one tree to the residuals, or <b>Run</b> to watch the staircase crawl toward the points.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" }, "Thin vertical lines are residuals — the error still left at each point. Each new tree carves the staircase to shrink them.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Ensemble state"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Stages (trees)"), stageEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Train MSE"), mseEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Learning rate ν"), lrEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Tree depth"), depthEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Boost"),
            el("div", { class: "controls" }, runBtn, addBtn),
            el("div", { class: "controls" }, resetBtn, newDataBtn)),
          el("div", { class: "panel" },
            el("h3", {}, "Settings"),
            el("div", { class: "controls" }, targetSlider.wrap),
            el("div", { class: "controls" }, lrSlider.wrap),
            el("div", { class: "controls" }, depthSlider.wrap)),
          legend)),
      verdict
    );

    // ---------------------------------------------------------------
    //  CART regression tree (1-D) fit to (x, residual) by minimising
    //  the squared error of the two children. Leaves predict the MEAN
    //  residual of their points. Grown to a hard depth cap.
    // ---------------------------------------------------------------
    function buildTree(pts, depth) {
      const vals = pts.map((p) => p.r);
      const node = { leaf: true, value: vals.length ? mean(vals) : 0 };
      if (depth >= maxDepth || pts.length < 2) return node;

      // sort by x once, then try every midpoint split; the best split is the
      // one minimising  SSE_left + SSE_right.  Use prefix sums of r and r² so
      // each child's SSE is O(1):  SSE = Σr² − (Σr)²/n.
      const sorted = [...pts].sort((a, b) => a.x - b.x);
      const n = sorted.length;
      const preS = new Array(n + 1).fill(0);   // prefix sum of r
      const preQ = new Array(n + 1).fill(0);   // prefix sum of r²
      for (let i = 0; i < n; i++) {
        preS[i + 1] = preS[i] + sorted[i].r;
        preQ[i + 1] = preQ[i] + sorted[i].r * sorted[i].r;
      }
      const sse = (s, q, c) => (c > 0 ? q - (s * s) / c : 0);
      const totalSSE = sse(preS[n], preQ[n], n);

      let best = null;
      for (let i = 1; i < n; i++) {
        if (sorted[i].x === sorted[i - 1].x) continue; // can't split between equal x
        const sL = preS[i], qL = preQ[i], cL = i;
        const sR = preS[n] - sL, qR = preQ[n] - qL, cR = n - i;
        const cost = sse(sL, qL, cL) + sse(sR, qR, cR);
        if (!best || cost < best.cost) {
          best = { cost, thr: (sorted[i].x + sorted[i - 1].x) / 2, i };
        }
      }
      // no useful split (all equal x, or the split doesn't reduce error)
      if (!best || best.cost >= totalSSE - 1e-12) return node;

      const L = sorted.slice(0, best.i);
      const R = sorted.slice(best.i);
      return {
        leaf: false,
        thr: best.thr,
        left: buildTree(L, depth + 1),
        right: buildTree(R, depth + 1),
      };
    }

    function predictTree(treeNode, x) {
      let n = treeNode;
      while (!n.leaf) n = x < n.thr ? n.left : n.right;
      return n.value;
    }

    // ensemble prediction  F_m(x) = F0 + ν · Σ h_k(x)
    function predict(x) {
      let s = F0;
      for (const t of stages) s += nu * predictTree(t, x);
      return s;
    }

    function trainMSE() {
      if (!data.length) return 0;
      let s = 0;
      for (const p of data) {
        const e = p.y - predict(p.x);
        s += e * e;
      }
      return s / data.length;
    }

    // fit ONE more stage to the current residuals
    function addStage() {
      if (!data.length) return;
      const withResid = data.map((p) => ({ x: p.x, r: p.y - predict(p.x) }));
      stages.push(buildTree(withResid, 0));
    }

    // ---------------------------------------------------------------
    //  drawing
    // ---------------------------------------------------------------
    function draw() {
      plot.clear("#070d1a");
      plot.grid(0.5);
      plot.axes();

      // residual segments first (under everything), thin vertical sticks
      plot.ctx.save();
      plot.clipToPlot();
      for (const p of data) {
        const f = predict(p.x);
        plot.line(p.x, p.y, p.x, f, withAlpha(classColor(1), 0.55), 1.25);
      }
      plot.ctx.restore();

      // dense staircase of the current prediction Fₘ(x)
      const SAMPLES = 320;
      const pts = [];
      for (let i = 0; i <= SAMPLES; i++) {
        const x = WIN.xmin + (i / SAMPLES) * (WIN.xmax - WIN.xmin);
        pts.push([x, clamp(predict(x), WIN.ymin, WIN.ymax)]);
      }
      plot.ctx.save();
      plot.clipToPlot();
      plot.path(pts, classColor(3), 2.6);
      plot.ctx.restore();

      // data points on top
      for (const p of data) plot.point(p.x, p.y, 4.5, classColor(0), "rgba(7,13,26,0.85)", 1.2);

      plot.label(`stage ${stages.length}`, plot.padL + 8, plot.padT + 16, withAlpha(classColor(3), 0.95));
    }

    function refresh() {
      draw();
      stageEl.textContent = stages.length;
      mseEl.textContent = data.length ? trainMSE().toFixed(4) : "—";
      lrEl.textContent = nu.toFixed(2);
      depthEl.textContent = maxDepth;

      const m = trainMSE();
      if (stages.length === 0) {
        verdict.innerHTML =
          "Stage <b>0</b>: the prediction is the flat <b>mean</b> of <code>y</code> — the best constant guess. " +
          `Its MSE is <b>${m.toFixed(4)}</b>. Every vertical stick is an error waiting for a tree to fix it.`;
      } else if (loop.running) {
        verdict.innerHTML =
          `Boosting… stage <b>${stages.length}</b>, train MSE <b>${m.toFixed(4)}</b> and falling. ` +
          `Each tree fits the current residuals; the staircase is creeping toward the data.`;
      } else {
        verdict.innerHTML =
          `<b>${stages.length}</b> trees stacked (depth ${maxDepth}, ν = ${nu.toFixed(2)}). Train MSE is <b>${m.toFixed(4)}</b>. ` +
          (nu < 0.2
            ? "With a <b>small</b> learning rate the staircase barely nudges each stage — it needs many trees, but it generalises better."
            : nu >= 0.8
              ? "With a <b>large</b> learning rate it lunges at the data — fast, but it can chase the noise and overfit."
              : "Add more stages to keep shrinking the residuals — but watch for the curve hugging the noise.");
      }
    }

    // ---------------------------------------------------------------
    //  animation loop — one stage every ~0.25s
    // ---------------------------------------------------------------
    let acc = 0;
    const loop = new Loop((dt) => {
      acc += dt;
      if (stages.length >= targetStages) { loop.stop(); runBtn.textContent = "Run ▸"; refresh(); return; }
      if (acc >= 0.25) { acc = 0; addStage(); refresh(); }
    });

    // ---------------------------------------------------------------
    //  data + lifecycle helpers
    // ---------------------------------------------------------------
    // wiggly target with structure a stump can chip at:
    //   g(x) = 0.7·sin(3x) + 0.3·sign(x)
    function g(x) { return 0.7 * Math.sin(3 * x) + 0.3 * Math.sign(x); }

    function newData() {
      data = range(POINTS).map(() => {
        const x = randRange(-1, 1);
        return { x, y: g(x) + gaussRandom(0, 0.12) };
      });
      reset();
    }

    function reset() {
      loop.stop();
      runBtn.textContent = "Run ▸";
      acc = 0;
      stages = [];
      F0 = data.length ? mean(data.map((p) => p.y)) : 0;
      refresh();
    }

    // ---- events ----
    runBtn.addEventListener("click", () => {
      if (!loop.running && stages.length >= targetStages) {
        // already at target — bump target up so Run does something visible
        targetStages = clamp(stages.length + 10, 1, 60);
        targetSlider.input.value = targetStages;
        targetSlider.value.textContent = targetStages;
      }
      loop.toggle();
      runBtn.textContent = loop.running ? "Pause" : "Run ▸";
      if (!loop.running) refresh();
    });
    addBtn.addEventListener("click", () => {
      if (loop.running) { loop.stop(); runBtn.textContent = "Run ▸"; }
      addStage();
      refresh();
    });
    resetBtn.addEventListener("click", reset);
    newDataBtn.addEventListener("click", newData);

    targetSlider.input.addEventListener("input", () => {
      targetStages = parseInt(targetSlider.input.value, 10);
      if (loop.running && stages.length >= targetStages) { loop.stop(); runBtn.textContent = "Run ▸"; }
    });
    lrSlider.input.addEventListener("input", () => {
      nu = parseFloat(lrSlider.input.value);
      // ν changes every stage's contribution — rebuild from scratch so the
      // residuals each tree saw stay consistent with the new ν.
      const keep = stages.length;
      stages = [];
      for (let i = 0; i < keep; i++) addStage();
      refresh();
    });
    depthSlider.input.addEventListener("input", () => {
      maxDepth = parseInt(depthSlider.input.value, 10);
      const keep = stages.length;
      stages = [];
      for (let i = 0; i < keep; i++) addStage();
      refresh();
    });

    // ---- init ----
    newData();

    return {
      onEnter() { draw(); refresh(); },
      onLeave() { loop.stop(); runBtn.textContent = "Run ▸"; },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: fix your own mistakes, in sequence</h3>
    <p>Gradient boosting is <b>stagewise additive modelling</b>. Instead of one big model, it builds a sum of many tiny ones — here, shallow regression trees (often just <em>stumps</em>, one split deep). The trick: each new tree isn't trained on the original target. It's trained on what's <em>left over</em> after every previous tree — the <b>residuals</b>, the errors so far. You watch the staircase above crawl toward the data because each tree only ever patches the gaps the last ones missed.</p>
  </div>
  <div class="lesson-block">
    <h3>🔁 The steps you can see</h3>
    <ol class="steps">
      <li><b>Start flat.</b> The first prediction is just the <b>mean</b> of <code>y</code> — the best single constant. Every point's vertical stick is its initial error.</li>
      <li><b>Compute residuals.</b> For each point, <code>r = y − F(x)</code>: how far the current prediction misses.</li>
      <li><b>Fit a tree to the residuals.</b> A small CART regression tree splits <code>x</code> to best explain those residuals; each leaf predicts the <em>mean residual</em> of its region.</li>
      <li><b>Add a shrunken step.</b> <code>F ← F + ν·h(x)</code>. The learning rate <code>ν</code> scales the tree down so no single step overshoots.</li>
      <li><b>Repeat.</b> New residuals, new tree, again. The sticks shrink; the staircase tightens to the curve.</li>
    </ol>
  </div>
  <div class="lesson-block">
    <h3>📉 Why it's literally gradient descent — in function space</h3>
    <p>For squared-error loss <code>L = ½(y − F)²</code>, the gradient with respect to the prediction <code>F</code> is:</p>
    <div class="formula">∂L/∂F = −(y − F) &nbsp;⟹&nbsp; −∂L/∂F = y − F = r</div>
    <p>So the residual <b>is</b> the negative gradient. Fitting a tree to the residuals and adding it is one descent step — except we're not nudging numbers, we're adding a whole function:</p>
    <div class="formula">F<sub>m</sub>(x) = F<sub>m−1</sub>(x) + ν · h<sub>m</sub>(x),&nbsp;&nbsp; h<sub>m</sub> ≈ r = y − F<sub>m−1</sub></div>
    <p>That's why it's called <em>gradient</em> boosting — it's gradient descent where the "parameter" being optimised is the function itself. (Compare the parameter-space version in the <b>Gradient Descent</b> game.) Swap the loss for something else and the residual changes, but the recipe is identical.</p>
  </div>
  <div class="lesson-block">
    <h3>🎚️ Shrinkage: the speed-vs-generalisation dial</h3>
    <p>The learning rate <code>ν</code> shrinks each tree's contribution. Crank it to 1 and the model lunges at the data — fast, but it starts memorising the noise after only a few stages. Drop it to 0.05 and each tree barely moves the curve, so you need <em>many</em> more stages — but the result is smoother and generalises better. The standard wisdom: <b>small ν + many trees beats large ν + few trees</b>. Slide ν down, then add stages, and watch the staircase approach the same curve more gently.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ The failure modes</h3>
    <ul class="bullets">
      <li><b>Overfitting from too many stages.</b> Boosting reduces <em>bias</em> relentlessly — keep adding trees and it will eventually thread every noisy point, carving spikes that don't generalise. There's no automatic stop; you tune the number of stages (early stopping on a validation set).</li>
      <li><b>Trees too deep.</b> Bump the depth to 3 with a high ν and few stages overfit fast — deep trees are strong learners, and boosting wants <em>weak</em> ones it can correct gradually.</li>
      <li><b>Sensitive to noise/outliers.</b> Because squared-error residuals grow with the miss, a wild point pulls hard on every subsequent tree.</li>
      <li><b>Sequential = slow.</b> Trees depend on the ones before them, so you can't build them in parallel the way a forest can.</li>
    </ul>
  </div>
  <div class="lesson-block">
    <h3>🌲 Boosting vs. bagging — opposite cures</h3>
    <p>This is the clearest contrast with the <b>Random Forest</b> game. Both are tree ensembles, but they attack different problems:</p>
    <table class="mini-table">
      <tr><th></th><th>Random Forest (bagging)</th><th>Gradient Boosting</th></tr>
      <tr><td>Trees built</td><td>in <b>parallel</b>, independently</td><td>in <b>sequence</b>, each on the last's residuals</td></tr>
      <tr><td>Combine by</td><td><b>averaging</b> / voting</td><td><b>adding</b> shrunken steps</td></tr>
      <tr><td>Tree size</td><td>deep, overfit</td><td>shallow, weak</td></tr>
      <tr><td>Mainly reduces</td><td><b>variance</b></td><td><b>bias</b></td></tr>
    </table>
    <p>A forest grows strong, jittery trees and averages away their <em>variance</em>; boosting grows weak <b>stumps</b> (see the <b>Decision Tree</b> game for one split) and stacks them to chip away the <em>bias</em>. Use a forest when single trees overfit; reach for boosting when you want to squeeze out the last bit of accuracy.</p>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> gradient boosting adds shallow trees one at a time, each fitted to the residuals (= the negative gradient) of the trees so far and shrunk by ν, turning "fix the last mistake" into gradient descent in function space.</p>
  </div>`,
};
