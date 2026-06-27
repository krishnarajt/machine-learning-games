// =====================================================================
//  Overfitting & Regularization — the central tension of ML.
//
//  1-D polynomial regression with a train/test split and ridge (L2)
//  regularisation. We fit a degree-d polynomial to ~16 TRAIN points by
//  ridge least squares — solving (XᵀX + λI)·w = Xᵀy with our own dense
//  Gaussian-elimination solver — then measure RMSE on a held-out TEST
//  set. Crank the degree and the test error U-curves: the model memorises
//  the training noise and flunks new data. Ridge λ shrinks the weights
//  back down to curb that variance.
// =====================================================================
import {
  el, button, ghostButton, slider, Plot,
  gaussRandom, classColor, range, clamp, lerp,
} from "../shared.js";

export default {
  key: "overfitting",
  name: "Overfitting & Regularization",
  icon: "〰️",
  category: "Foundations",
  tag: "Core",
  blurb: "Crank up a polynomial's degree and watch it ace the training points but flunk new ones.",
  eyebrow: "Foundations · Model fit",
  title: 'Overfitting: when a model <span class="grad">memorises</span> instead of learning',
  lede:
    "The whole game of machine learning is doing well on data you <em>haven't seen</em>. Fit a too-simple model and it underfits; " +
    "fit a too-flexible one and it memorises the noise. Slide the polynomial degree and the regularisation strength, and watch the " +
    "training error and the <em>test</em> error pull apart — the central tension in all of ML.",

  mount(root) {
    // ---- config ----
    const X_LO = -1, X_HI = 1;          // domain the data is sampled over
    const N_TRAIN = 16, N_TEST = 16;
    const MAX_DEG = 12;

    // the smooth "true" function the data is generated from (plus noise)
    const trueFn = (x) => Math.sin(2.3 * x) + 0.4 * x;

    // ---- state ----
    let degree = 4;
    let lambdaSlider = -6;              // -6 means "off" → λ = 0; else λ = 10^slider
    let noise = 0.18;
    let showTrue = true;

    let train = [];                    // [{x, y}]
    let test = [];                     // [{x, y}]
    let weights = [];                  // fitted coefficients [w0..wd]
    let trainRMSE = 0, testRMSE = 0;

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 430, role: "img",
      "aria-label": "Polynomial fit through training points, with held-out test points and the true function." });
    const plot = new Plot(canvas, { xmin: -1.1, xmax: 1.1, ymin: -1.8, ymax: 1.8, pad: 22 });

    const curveCanvas = el("canvas", { width: 760, height: 210, role: "img",
      "aria-label": "Training and test error plotted against polynomial degree." });
    const curvePlot = new Plot(curveCanvas, { xmin: 1, xmax: MAX_DEG, ymin: 0, ymax: 1, padL: 44, padR: 16, padT: 18, padB: 30 });

    const newDataBtn = button("New data ⟳");
    const trueBtn = ghostButton("Hide true function");

    const degSlider = slider({ min: 1, max: MAX_DEG, step: 1, value: degree, label: "Polynomial degree d =", format: (v) => v });
    const lamSlider = slider({ min: -6, max: 0, step: 0.25, value: lambdaSlider, label: "Ridge λ =",
      format: (v) => (v <= -6 ? "0 (off)" : fmtLambda(Math.pow(10, v))) });
    const noiseSlider = slider({ min: 0, max: 0.45, step: 0.01, value: noise, label: "Noise σ =", format: (v) => v.toFixed(2) });

    const degEl = el("span", { class: "v" }, "—");
    const lamEl = el("span", { class: "v" }, "—");
    const trainEl = el("span", { class: "v" }, "—");
    const testEl = el("span", { class: "v" }, "—");
    const gapEl = el("span", { class: "v" }, "—");

    const legend = el("div", { class: "legend" },
      el("span", { class: "legend-item" }, el("span", { class: "legend-line", style: { background: "#ffd166" } }), "fitted polynomial"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { background: classColor(0) } }), "train points"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { background: "transparent", border: `2px solid ${classColor(1)}` } }), "test points (held out)"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-line", style: { background: "rgba(255,255,255,0.35)" } }), "true function"));

    const verdict = el("p", { class: "verdict" },
      "Slide the <b>degree</b> up and watch the curve start chasing individual points — the test error tells the real story.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" }, "Filled dots are training data; hollow rings are the held-out test set the model never sees while fitting."),
          el("div", { class: "canvas-stage", style: { marginTop: "10px" } }, curveCanvas),
          el("p", { class: "caption" }, "Train (cyan) vs test (coral) RMSE across every degree 1…12 — the classic U-shaped test curve. The dotted line marks the degree you picked.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Fit & error"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "degree d"), degEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "λ (ridge)"), lamEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "train RMSE"), trainEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "test RMSE"), testEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "test − train gap"), gapEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, degSlider.wrap),
            el("div", { class: "controls" }, lamSlider.wrap),
            el("div", { class: "controls" }, noiseSlider.wrap),
            el("div", { class: "controls" }, newDataBtn, trueBtn)),
          legend)),
      verdict
    );

    // ---- formatting ----
    function fmtLambda(v) {
      if (v === 0) return "0";
      if (v >= 0.001) return v.toFixed(v >= 0.1 ? 3 : 4).replace(/0+$/, "").replace(/\.$/, "");
      return v.toExponential(1);
    }
    function currentLambda() {
      return lambdaSlider <= -6 ? 0 : Math.pow(10, lambdaSlider);
    }

    // ---- data ----
    function makeData() {
      // x spread evenly across the domain (jittered) so train and test cover it
      const sample = (n) => range(n).map((i) => {
        const t = (i + 0.5) / n;                       // even coverage
        const x = clamp(lerp(X_LO, X_HI, t) + gaussRandom(0, 0.02), X_LO, X_HI);
        return { x, y: trueFn(x) + gaussRandom(0, noise) };
      });
      train = sample(N_TRAIN);
      test = sample(N_TEST);
    }
    // re-roll only the noise, keeping the same true function ("New data")
    function reseedNoise() { makeData(); }

    // ---- linear algebra: solve A·w = b by Gaussian elimination + partial pivot ----
    // A is n×n (array of rows), b is length n. Returns w (length n). Mutates copies.
    function solveLinear(A, b) {
      const n = b.length;
      // build an augmented copy so we never trample the caller's matrices
      const M = A.map((row, i) => row.slice().concat(b[i]));
      for (let col = 0; col < n; col++) {
        // partial pivot: find the row with the largest |value| in this column
        let piv = col, best = Math.abs(M[col][col]);
        for (let r = col + 1; r < n; r++) {
          const v = Math.abs(M[r][col]);
          if (v > best) { best = v; piv = r; }
        }
        if (piv !== col) { const tmp = M[piv]; M[piv] = M[col]; M[col] = tmp; }
        const diag = M[col][col];
        if (Math.abs(diag) < 1e-14) continue;          // singular column — skip (ridge floor avoids this)
        // eliminate this column from every lower row
        for (let r = col + 1; r < n; r++) {
          const f = M[r][col] / diag;
          if (f === 0) continue;
          for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k];
        }
      }
      // back-substitution
      const w = new Array(n).fill(0);
      for (let i = n - 1; i >= 0; i--) {
        let s = M[i][n];
        for (let k = i + 1; k < n; k++) s -= M[i][k] * w[k];
        const diag = M[i][i];
        w[i] = Math.abs(diag) < 1e-14 ? 0 : s / diag;
      }
      return w;
    }

    // build the design-matrix ROW for one x: [1, x, x², …, x^d]
    function basis(x, d) {
      const row = new Array(d + 1);
      let p = 1;
      for (let j = 0; j <= d; j++) { row[j] = p; p *= x; }
      return row;
    }

    // ridge least squares: (XᵀX + λI)·w = Xᵀy  (bias term w0 NOT regularised)
    function fitRidge(data, d, lambda) {
      const m = d + 1;
      // accumulate XᵀX (m×m) and Xᵀy (m) directly — no need to materialise X
      const XtX = range(m).map(() => new Array(m).fill(0));
      const Xty = new Array(m).fill(0);
      for (const p of data) {
        const phi = basis(p.x, d);
        for (let a = 0; a < m; a++) {
          Xty[a] += phi[a] * p.y;
          for (let bb = a; bb < m; bb++) {
            const v = phi[a] * phi[bb];
            XtX[a][bb] += v;
            if (a !== bb) XtX[bb][a] += v;
          }
        }
      }
      // add λ on the diagonal (skip index 0 = bias) plus a tiny floor for stability
      const FLOOR = 1e-8;
      for (let a = 0; a < m; a++) {
        XtX[a][a] += FLOOR + (a === 0 ? 0 : lambda);
      }
      return solveLinear(XtX, Xty);
    }

    function evalPoly(w, x) {
      let s = 0, p = 1;
      for (let j = 0; j < w.length; j++) { s += w[j] * p; p *= x; }
      return s;
    }

    function rmse(data, w) {
      if (!data.length) return 0;
      let s = 0;
      for (const p of data) { const e = evalPoly(w, p.x) - p.y; s += e * e; }
      return Math.sqrt(s / data.length);
    }

    // ---- recompute everything for the current sliders ----
    function recompute() {
      const lambda = currentLambda();
      weights = fitRidge(train, degree, lambda);
      trainRMSE = rmse(train, weights);
      testRMSE = rmse(test, weights);
    }

    // ---- the error-vs-degree curve (recomputed when data/λ change) ----
    let trainCurve = [], testCurve = [];
    function recomputeCurve() {
      const lambda = currentLambda();
      trainCurve = []; testCurve = [];
      for (let d = 1; d <= MAX_DEG; d++) {
        const w = fitRidge(train, d, lambda);
        trainCurve.push(rmse(train, w));
        testCurve.push(rmse(test, w));
      }
    }

    // ---- drawing: main fit plot ----
    function draw() {
      plot.clear("#070d1a");
      plot.grid(0.5);
      plot.axes();

      plot.ctx.save();
      plot.clipToPlot();   // a wild high-degree curve can't scribble outside

      // faint dashed true function
      if (showTrue) {
        const pts = [];
        for (let i = 0; i <= 200; i++) {
          const x = lerp(plot.xmin, plot.xmax, i / 200);
          pts.push([x, trueFn(x)]);
        }
        plot.path(pts, "rgba(255,255,255,0.32)", 2, [7, 6]);
      }

      // fitted polynomial — bright, densely sampled
      if (weights.length) {
        const pts = [];
        for (let i = 0; i <= 320; i++) {
          const x = lerp(plot.xmin, plot.xmax, i / 320);
          pts.push([x, evalPoly(weights, x)]);
        }
        plot.path(pts, "#ffd166", 2.75);
      }

      // test points first (hollow rings, behind train)
      for (const p of test) {
        const px = plot.sx(p.x), py = plot.sy(p.y);
        plot.ctx.beginPath();
        plot.ctx.arc(px, py, 5, 0, Math.PI * 2);
        plot.ctx.lineWidth = 2;
        plot.ctx.strokeStyle = classColor(1);
        plot.ctx.stroke();
      }
      // train points (filled)
      for (const p of train) {
        plot.point(p.x, p.y, 4.5, classColor(0), "rgba(7,13,26,0.85)", 1.5);
      }

      plot.ctx.restore();
    }

    // ---- drawing: error-vs-degree curve ----
    function drawCurve() {
      curvePlot.clear("#070d1a");
      // autoscale y to the data with a little headroom
      let hi = 0;
      for (const v of trainCurve) hi = Math.max(hi, isFinite(v) ? v : 0);
      for (const v of testCurve) hi = Math.max(hi, isFinite(v) ? v : 0);
      hi = Math.max(0.2, isFinite(hi) ? hi : 0.2) * 1.12;
      curvePlot.setWindow({ ymin: 0, ymax: hi });

      curvePlot.grid(hi > 1.2 ? 0.5 : 0.2);

      const c = curvePlot.ctx;

      // marker for the currently selected degree
      c.save();
      c.strokeStyle = "rgba(255,255,255,0.28)";
      c.lineWidth = 1.5;
      c.setLineDash([4, 4]);
      c.beginPath();
      c.moveTo(curvePlot.sx(degree), curvePlot.padT);
      c.lineTo(curvePlot.sx(degree), curvePlot.padT + curvePlot.plotH);
      c.stroke();
      c.restore();

      // two error lines (clamp to window; map a non-finite refit to the top)
      const clampY = (v) => Math.min(isFinite(v) ? v : hi, hi);
      const mkPts = (arr) => arr.map((v, i) => [i + 1, clampY(v)]);
      curvePlot.path(mkPts(trainCurve), classColor(0), 2.25);
      curvePlot.path(mkPts(testCurve), classColor(1), 2.25);

      // dots at each degree
      trainCurve.forEach((v, i) => curvePlot.point(i + 1, clampY(v), 2.8, classColor(0), null));
      testCurve.forEach((v, i) => curvePlot.point(i + 1, clampY(v), 2.8, classColor(1), null));

      // axis labels
      curvePlot.label("RMSE", curvePlot.padL - 36, curvePlot.padT + 4, "rgba(234,238,251,0.6)", "left", "11px 'JetBrains Mono', monospace");
      curvePlot.label("degree →", curvePlot.padL + curvePlot.plotW - 4, curvePlot.padT + curvePlot.plotH + 22, "rgba(234,238,251,0.6)", "right", "11px 'JetBrains Mono', monospace");
      // x ticks
      for (let d = 1; d <= MAX_DEG; d++) {
        if (d % 2 === 1 || d === MAX_DEG) {
          curvePlot.label(String(d), curvePlot.sx(d), curvePlot.padT + curvePlot.plotH + 14, "rgba(234,238,251,0.45)", "center", "10px 'JetBrains Mono', monospace");
        }
      }
    }

    // ---- regime classifier ----
    function regime() {
      // heuristics on absolute and relative error
      const gap = testRMSE - trainRMSE;
      const ratio = trainRMSE > 1e-6 ? testRMSE / trainRMSE : (testRMSE > 0.05 ? 99 : 1);
      if (trainRMSE > 0.45 && testRMSE > 0.45) return "underfit";
      if (trainRMSE < 0.12 && (gap > 0.25 || ratio > 2.2)) return "overfit";
      if (gap > 0.3 || ratio > 2.5) return "overfit";
      return "good";
    }

    // ---- refresh: numbers + verdict ----
    function refresh() {
      recompute();
      recomputeCurve();
      draw();
      drawCurve();

      const lambda = currentLambda();
      degEl.textContent = degree;
      lamEl.textContent = lambda === 0 ? "0 (off)" : fmtLambda(lambda);
      trainEl.textContent = trainRMSE.toFixed(3);
      testEl.textContent = testRMSE.toFixed(3);
      const gap = testRMSE - trainRMSE;
      gapEl.textContent = (gap >= 0 ? "+" : "") + gap.toFixed(3);

      const r = regime();
      if (r === "underfit") {
        verdict.innerHTML = `<b>Underfitting</b> — both errors are high (train ${trainRMSE.toFixed(2)}, test ${testRMSE.toFixed(2)}). ` +
          `The model is too rigid to capture the curve. Raise the <b>degree</b> (or lower <b>λ</b>) to give it more flexibility.`;
      } else if (r === "overfit") {
        verdict.innerHTML = `<b>Overfitting</b> — train error is tiny (${trainRMSE.toFixed(2)}) but test error has ballooned to ${testRMSE.toFixed(2)}. ` +
          `The polynomial is memorising the training <em>noise</em>, not the signal. Lower the <b>degree</b> or crank up <b>λ</b> to rein it in.`;
      } else {
        verdict.innerHTML = `<b>Good fit</b> — test ≈ train (${testRMSE.toFixed(2)} vs ${trainRMSE.toFixed(2)}) and both are low. ` +
          `The model captures the underlying function and <em>generalises</em> to data it never saw. This is the sweet spot the U-curve below is pointing at.`;
      }
    }

    // ---- events ----
    degSlider.input.addEventListener("input", () => {
      degree = parseInt(degSlider.input.value, 10);
      refresh();
    });
    lamSlider.input.addEventListener("input", () => {
      lambdaSlider = parseFloat(lamSlider.input.value);
      refresh();
    });
    noiseSlider.input.addEventListener("input", () => {
      noise = parseFloat(noiseSlider.input.value);
      makeData();
      refresh();
    });
    newDataBtn.addEventListener("click", () => { reseedNoise(); refresh(); });
    trueBtn.addEventListener("click", () => {
      showTrue = !showTrue;
      trueBtn.textContent = showTrue ? "Hide true function" : "Show true function";
      draw();
    });

    // ---- init ----
    makeData();
    refresh();

    return {
      onEnter() { refresh(); },
      onLeave() { /* no Loop running — nothing to stop */ },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: do well on data you haven't seen</h3>
    <p>Every model has one real test: <b>generalisation</b> — accuracy on <em>new</em> data, not the data it trained on. It is trivial to ace the training set (just memorise it), so we hide some examples in a <b>test set</b> the model never touches while fitting. Above, the filled cyan dots are the training points and the hollow coral rings are the held-out test points. The whole drama of machine learning lives in the gap between how well a model does on each.</p>
  </div>
  <div class="lesson-block">
    <h3>🔭 What you can see on screen</h3>
    <ol class="steps">
      <li><b>Fit.</b> We fit a polynomial <code>ŷ = w<sub>0</sub> + w<sub>1</sub>x + w<sub>2</sub>x² + … + w<sub>d</sub>x<sup>d</sup></code> to the training points by least squares.</li>
      <li><b>Score.</b> We measure the root-mean-square error on the training points and, separately, on the untouched test points.</li>
      <li><b>Slide the degree.</b> At <code>d = 1</code> you get a straight line (exactly the <em>Linear Regression</em> game) — too stiff. Crank <code>d</code> up and the curve starts wiggling through every training dot.</li>
      <li><b>Watch the U.</b> The lower panel plots both errors against degree. Train error falls forever; test error dips, bottoms out, then climbs — the classic <b>U-shaped test curve</b>.</li>
    </ol>
  </div>
  <div class="lesson-block">
    <h3>⚖️ Bias and variance — the core tradeoff</h3>
    <p>Two opposite failure modes sit at the ends of the slider:</p>
    <ul class="bullets">
      <li><b>Underfitting (high bias).</b> The model is too simple to represent the truth. A line through a sine wave is wrong no matter how you tune it — train <em>and</em> test error are both high.</li>
      <li><b>Overfitting (high variance).</b> The model is so flexible it threads through every training point, noise and all. Train error drops to nearly zero while test error explodes, because the wiggles it learned were peculiar to this one noisy sample.</li>
    </ul>
    <p>The expected test error decomposes into exactly these pieces, plus the irreducible noise you can never beat:</p>
    <div class="formula">E[(y − ŷ)²] = bias² + variance + σ²<sub>noise</sub></div>
    <p>Simple models trade variance for bias; complex models do the reverse. The best degree is the bottom of the U, where the two are balanced.</p>
  </div>
  <div class="lesson-block">
    <h3>🪢 Ridge regularisation — shrinking the weights</h3>
    <p>You don't have to throw away flexibility to fight overfitting. <b>Ridge regression</b> keeps the high degree but adds a penalty on the <em>size</em> of the coefficients, so the optimiser only uses big weights if the data really earns them:</p>
    <div class="formula">J(w) = Σ<sub>i</sub> ( y<sub>i</sub> − ŷ<sub>i</sub> )² + λ Σ<sub>j≥1</sub> w<sub>j</sub>²</div>
    <p>The extra term is L2 (squared) penalty with strength <code>λ</code>. We leave the bias <code>w<sub>0</sub></code> out of the penalty — shifting everything up or down shouldn't be punished. Setting the gradient to zero gives a closed form, the <b>ridge normal equations</b>:</p>
    <div class="formula">(XᵀX + λI) w = Xᵀy</div>
    <p>where <code>X</code> is the design matrix whose rows are <code>[1, x, x², …, x<sup>d</sup>]</code>. This game builds <code>XᵀX</code> and <code>Xᵀy</code> directly, adds <code>λ</code> to the diagonal (a tiny <code>10⁻⁸</code> floor too, for numerical stability at high degree), and solves the small dense system with <b>Gaussian elimination and partial pivoting</b> — the same closed-form route the <em>Linear Regression</em> game uses, just generalised to many features. Turn <code>λ</code> up and watch a degree-12 curve relax from a frantic squiggle back toward the smooth truth.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ The catches</h3>
    <ul class="bullets">
      <li><b>λ too large → underfit.</b> Crank it and the weights are squeezed toward zero; the curve flattens to a near-constant and both errors rise again. Too small → no help, and you overfit. There is a sweet spot, just like with degree.</li>
      <li><b>The test set can lie too.</b> With only 16 test points, "New data" can swing the numbers around. Press it a few times — the U-curve's bottom wanders. Never tune endlessly against one fixed test set or you start overfitting <em>it</em>.</li>
      <li><b>More data beats fancier models.</b> The same degree-12 polynomial that overfits 16 points behaves far better with hundreds — more data tames variance for free.</li>
    </ul>
  </div>
  <div class="lesson-block">
    <h3>🧪 How you'd actually pick d and λ</h3>
    <p>You can't peek at the real test set to choose hyperparameters. In practice you use <b>cross-validation</b>: carve the training data into <code>k</code> folds, train on <code>k−1</code> and validate on the held-out fold, rotate, and average. Pick the <code>d</code> and <code>λ</code> with the best average validation error, then retrain on everything. The picture above — fit on train, judge on a fold you didn't fit — is exactly what one round of cross-validation looks like. The gradient-based optimisers in the <em>Gradient Descent</em> game minimise the very same ridge cost <code>J(w)</code> when the matrix solve is too big.</p>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> a model that's too simple underfits and one that's too flexible memorises the noise — so we judge models on held-out data, read the U-shaped test curve, and use regularisation (and cross-validation) to land at the degree and λ that <em>generalise</em>.</p>
  </div>`,
};
