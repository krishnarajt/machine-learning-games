// =====================================================================
//  Logistic Regression — a binary linear classifier.
//
//  Model:    z = w1*x + w2*y + b ;  p = sigmoid(z)
//  Predict:  class 1 if p >= 0.5  (i.e. z >= 0)
//  Train:    gradient descent on the mean cross-entropy (log loss).
//            For each point the gradient of the loss is (p - y) times the
//            feature, averaged over the dataset:
//               dL/dw1 = mean((p - y) * x)
//               dL/dw2 = mean((p - y) * y_coord)
//               dL/db  = mean(p - y)
//            then step weights by -lr * grad.
//
//  The background shows the probability field (one class colour for p<0.5,
//  the other for p>0.5, opacity = how confident); the white line is the
//  decision boundary where z = 0.
// =====================================================================
import {
  el, button, ghostButton, slider, Loop, Plot,
  blobs, sigmoid, clamp, classColor, hexToRgb, lerp,
} from "../shared.js";

const COL_A = classColor(0); // class 0  (cyan)
const COL_B = classColor(1); // class 1  (coral)
const RGB_A = hexToRgb(COL_A);
const RGB_B = hexToRgb(COL_B);

// Mix the two class colours by t∈[0,1] and return an rgba() string with alpha a.
// (mixHex returns "rgb(...)", which withAlpha cannot re-parse — so build it here.)
function fieldColor(t, a) {
  const r = Math.round(lerp(RGB_A.r, RGB_B.r, t));
  const g = Math.round(lerp(RGB_A.g, RGB_B.g, t));
  const b = Math.round(lerp(RGB_A.b, RGB_B.b, t));
  return `rgba(${r},${g},${b},${a})`;
}

export default {
  key: "logistic-regression",
  name: "Logistic Regression",
  icon: "🔀",
  category: "Classification",
  tag: "Linear",
  blurb: "Paint two classes and watch a straight boundary and a probability gradient form.",
  eyebrow: "Classification · Supervised",
  title: 'Logistic Regression: drawing a <span class="grad">probability line</span>',
  lede:
    "Linear regression predicts a number; logistic regression bends that number into a <em>probability</em> with the sigmoid, " +
    "then draws a line between two classes. Place points of each colour and watch it learn where one class ends and the other " +
    "begins — and how sure it is everywhere in between.",

  mount(root) {
    // ---- state ----
    let points = [];          // { x, y, label } — label is 0 or 1 (TRUE class)
    let w1 = 0, w2 = 0, b = 0;
    let epoch = 0;
    let paintClass = 0;       // which class a canvas click adds

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Scatter plot of two classes with a probability field and a straight decision boundary." });
    const plot = new Plot(canvas, { xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2, pad: 16 });

    const trainBtn = button("Train ▸");
    const stepBtn = button("Step");
    const resetBtn = ghostButton("Reset weights");
    const newDataBtn = ghostButton("New data");
    const clearBtn = ghostButton("Clear");
    const lrSlider = slider({ min: 0.05, max: 3, step: 0.05, value: 0.8, label: "Learning rate η =", format: (v) => v.toFixed(2) });

    // segmented control: which class clicks add
    const segA = el("button", { class: "active" }, "Class A");
    const segB = el("button", {}, "Class B");
    const seg = el("div", { class: "seg-group" }, segA, segB);

    const w1El = el("span", { class: "v" }, "0.000");
    const w2El = el("span", { class: "v" }, "0.000");
    const bEl = el("span", { class: "v" }, "0.000");
    const lossEl = el("span", { class: "v" }, "—");
    const accEl = el("span", { class: "v" }, "—");
    const epochEl = el("span", { class: "v" }, "0");

    const legend = el("div", { class: "legend" },
      el("span", { class: "legend-item" },
        el("span", { class: "legend-dot", style: { background: COL_A } }), "Class A (label 0)"),
      el("span", { class: "legend-item" },
        el("span", { class: "legend-dot", style: { background: COL_B } }), "Class B (label 1)"),
      el("span", { class: "legend-item" },
        el("span", { class: "legend-line", style: { background: "#fff" } }), "boundary p = 0.5"));

    const verdict = el("p", { class: "verdict" },
      "Press <b>Train</b> to descend the log-loss, or <b>Step</b> for one gradient step. " +
      "Click the canvas to drop a point of the selected class.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" },
            "Background = predicted probability (deeper colour = more confident). " +
            "Click empty space to add a point of the chosen class.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Model & fit"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "w₁ (x weight)"), w1El),
            el("div", { class: "kv" }, el("span", { class: "k" }, "w₂ (y weight)"), w2El),
            el("div", { class: "kv" }, el("span", { class: "k" }, "b (bias)"), bEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Mean log-loss"), lossEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Accuracy"), accEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Epochs"), epochEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, trainBtn, stepBtn),
            el("div", { class: "controls" }, lrSlider.wrap),
            el("div", { class: "controls" }, el("span", { class: "toggle" }, "Add as "), seg),
            el("div", { class: "controls" }, resetBtn, newDataBtn, clearBtn)),
          legend)),
      verdict
    );

    // ---- model maths ----
    const score = (x, y) => w1 * x + w2 * y + b;        // z
    const prob = (x, y) => sigmoid(score(x, y));        // p = σ(z)

    function resetWeights() {
      // small random weights so the boundary starts in a generic place
      w1 = (Math.random() - 0.5) * 0.4;
      w2 = (Math.random() - 0.5) * 0.4;
      b = (Math.random() - 0.5) * 0.2;
      epoch = 0;
      refresh();
    }

    function seedData() {
      // two roughly linearly separable Gaussian blobs (label 0 and 1)
      points = blobs([[-0.45, -0.35], [0.5, 0.4]], 30, 0.18)
        .map((p) => ({ x: p.x, y: p.y, label: p.label }));
    }

    // ---- one gradient-descent step over the whole dataset ----
    function step() {
      if (!points.length) return;
      const lr = parseFloat(lrSlider.input.value);
      let g1 = 0, g2 = 0, gb = 0;
      for (const p of points) {
        const pred = prob(p.x, p.y);
        const err = pred - p.label;   // (p - y)
        g1 += err * p.x;
        g2 += err * p.y;
        gb += err;
      }
      const n = points.length;
      g1 /= n; g2 /= n; gb /= n;
      w1 -= lr * g1;
      w2 -= lr * g2;
      b -= lr * gb;
      epoch++;
      refresh();
    }

    // ---- metrics ----
    function logLoss() {
      if (!points.length) return 0;
      let s = 0;
      for (const p of points) {
        const pred = clamp(prob(p.x, p.y), 1e-7, 1 - 1e-7);
        s += -(p.label * Math.log(pred) + (1 - p.label) * Math.log(1 - pred));
      }
      return s / points.length;
    }

    function accuracy() {
      if (!points.length) return 0;
      let correct = 0;
      for (const p of points) {
        const yhat = prob(p.x, p.y) >= 0.5 ? 1 : 0;
        if (yhat === p.label) correct++;
      }
      return correct / points.length;
    }

    // ---- draw ----
    function draw() {
      plot.clear("#070d1a");

      // probability field: mix the two class colours by p, opacity by confidence
      plot.shade((x, y) => {
        const p = prob(x, y);                    // p=0 -> A, p=1 -> B
        const conf = Math.abs(p - 0.5) * 2;      // 0 at the boundary, 1 far away
        return fieldColor(p, 0.12 + 0.5 * conf);
      }, 46);

      plot.grid(0.5, "rgba(255,255,255,0.05)");

      // decision boundary: line where z = 0  ->  w1*x + w2*y + b = 0
      drawBoundary();

      // data points coloured by TRUE class, on top
      for (const p of points) {
        const col = p.label === 0 ? COL_A : COL_B;
        const correct = (prob(p.x, p.y) >= 0.5 ? 1 : 0) === p.label;
        plot.point(p.x, p.y, 5, col, correct ? "rgba(7,13,26,0.85)" : "#ffffff", correct ? 1.5 : 2.5);
      }
    }

    function drawBoundary() {
      // Solve for the line z=0 inside the window. Prefer y = -(w1*x + b)/w2,
      // but guard against w2 ≈ 0 (near-vertical boundary): then x = -(w2*y + b)/w1.
      const { xmin, xmax, ymin, ymax } = plot;
      let pa = null, pb = null;
      if (Math.abs(w2) > Math.abs(w1)) {
        const yAt = (x) => -(w1 * x + b) / w2;
        pa = [xmin, yAt(xmin)];
        pb = [xmax, yAt(xmax)];
      } else if (Math.abs(w1) > 1e-9) {
        const xAt = (y) => -(w2 * y + b) / w1;
        pa = [xAt(ymin), ymin];
        pb = [xAt(ymax), ymax];
      } else {
        return; // both weights ~0: no defined boundary yet
      }
      plot.ctx.save();
      plot.clipToPlot();
      plot.line(pa[0], pa[1], pb[0], pb[1], "rgba(255,255,255,0.92)", 2.5);
      plot.ctx.restore();
    }

    function refresh() {
      draw();
      w1El.textContent = w1.toFixed(3);
      w2El.textContent = w2.toFixed(3);
      bEl.textContent = b.toFixed(3);
      epochEl.textContent = epoch;
      if (!points.length) {
        lossEl.textContent = "—";
        accEl.textContent = "—";
        verdict.innerHTML = "No data — click the canvas to paint points, or hit <b>New data</b> for two blobs.";
        return;
      }
      const loss = logLoss();
      const acc = accuracy();
      lossEl.textContent = loss.toFixed(4);
      accEl.textContent = (acc * 100).toFixed(1) + "%";

      const sure = Math.hypot(w1, w2); // steeper z = sharper, more confident boundary
      if (epoch === 0) {
        verdict.innerHTML =
          `Untrained: log-loss <b>${loss.toFixed(3)}</b>, accuracy <b>${(acc * 100).toFixed(0)}%</b>. ` +
          `Press <b>Train</b> and watch the boundary swing toward the gap between the colours.`;
      } else if (acc >= 0.999) {
        verdict.innerHTML =
          `Separated cleanly after <b>${epoch}</b> epochs — accuracy <b>100%</b>, log-loss <b>${loss.toFixed(3)}</b>. ` +
          `The two blobs are linearly separable, so the line keeps getting steeper (more confident) as it trains.`;
      } else {
        verdict.innerHTML =
          `Epoch <b>${epoch}</b>: log-loss <b>${loss.toFixed(3)}</b>, accuracy <b>${(acc * 100).toFixed(0)}%</b>. ` +
          `Each step nudges the boundary toward misclassified points; the colour gradient steepens as |w| grows ` +
          `(currently <b>${sure.toFixed(2)}</b>).`;
      }
    }

    // ---- animation loop: several gradient steps per frame ----
    const loop = new Loop(() => {
      for (let i = 0; i < 4; i++) step();   // step() calls refresh() each time
    });

    function setTrainLabel() {
      trainBtn.textContent = loop.running ? "Pause" : "Train ▸";
    }

    // ---- events ----
    trainBtn.addEventListener("click", () => { loop.toggle(); setTrainLabel(); });
    stepBtn.addEventListener("click", () => {
      if (loop.running) { loop.stop(); setTrainLabel(); }
      step();
    });
    resetBtn.addEventListener("click", () => { loop.stop(); setTrainLabel(); resetWeights(); });
    newDataBtn.addEventListener("click", () => { loop.stop(); setTrainLabel(); seedData(); resetWeights(); });
    clearBtn.addEventListener("click", () => { loop.stop(); setTrainLabel(); points = []; resetWeights(); });
    lrSlider.input.addEventListener("input", refresh);

    segA.addEventListener("click", () => { paintClass = 0; segA.classList.add("active"); segB.classList.remove("active"); });
    segB.addEventListener("click", () => { paintClass = 1; segB.classList.add("active"); segA.classList.remove("active"); });

    canvas.addEventListener("pointerdown", (e) => {
      const { x, y } = plot.eventToData(e);
      if (x < plot.xmin || x > plot.xmax || y < plot.ymin || y > plot.ymax) return;
      points.push({ x, y, label: paintClass });
      refresh();
    });

    // ---- init ----
    seedData();
    resetWeights();

    return {
      onEnter() { draw(); },
      onLeave() { loop.stop(); setTrainLabel(); },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: turn a line into a yes/no answer</h3>
    <p>Linear regression gives you a number that runs off to ±∞ — useless if the answer is "spam or not", "tumour or benign", "will churn or won't". Logistic regression keeps the same linear core but squashes its output into a <b>probability</b> between 0 and 1, then calls anything above 0.5 class 1. It's the workhorse classifier: fast, interpretable, and the literal last layer of most neural networks.</p>
  </div>
  <div class="lesson-block">
    <h3>📐 From score to probability — the sigmoid</h3>
    <p>First compute a linear <b>score</b> from the features (here the two coordinates):</p>
    <div class="formula">z = w<sub>1</sub>·x + w<sub>2</sub>·y + b</div>
    <p>Then bend it through the <b>sigmoid</b> σ, an S-shaped curve that maps any real number into (0, 1):</p>
    <div class="formula">σ(z) = 1 / (1 + e<sup>−z</sup>)</div>
    <p>At <code>z = 0</code> the sigmoid is exactly 0.5; large positive <code>z</code> → ~1, large negative → ~0. So <code>p = σ(z)</code> is the model's confidence that the point is class 1. On the canvas that's the colour gradient: deep coral where it's sure of B, deep cyan where it's sure of A, washed-out in the uncertain middle band.</p>
  </div>
  <div class="lesson-block">
    <h3>📏 Why the boundary is a straight line</h3>
    <p>The model flips its vote exactly where <code>p = 0.5</code>, which is where <code>z = 0</code>:</p>
    <div class="formula">w<sub>1</sub>·x + w<sub>2</sub>·y + b = 0</div>
    <p>That's the equation of a straight line (a hyperplane in higher dimensions) — the white line above. Logistic regression can <em>only</em> draw straight boundaries. Give it two interleaved moons or an XOR pattern and no single line can separate them: accuracy stalls. That curved-boundary problem is exactly what stacking sigmoids into <b>neural networks</b> solves — each layer warps the space until the final logistic layer <em>can</em> cut it with a line.</p>
  </div>
  <div class="lesson-block">
    <h3>📉 The cost: cross-entropy (log loss), not squared error</h3>
    <p>We score the fit with <b>cross-entropy</b>: punish being confident <em>and wrong</em>.</p>
    <div class="formula">L = −(1/n) Σ [ y·log(p) + (1−y)·log(1−p) ]</div>
    <p>If the true label is 1 and the model says <code>p≈0</code>, then <code>log(p)→−∞</code> — a huge penalty. Why not plain squared error like linear regression? Because <code>(p−y)²</code> composed with the sigmoid is <em>non-convex</em> — full of flat spots and local dips that trap gradient descent. Cross-entropy paired with the sigmoid is convex and its gradient is beautifully simple (next block).</p>
  </div>
  <div class="lesson-block">
    <h3>⛰️ Learning: gradient descent, push toward the mistakes</h3>
    <p>The gradient of the log-loss collapses to one clean rule — the <b>error</b> <code>(p − y)</code> times each feature, averaged over the data:</p>
    <div class="formula">∂L/∂w<sub>1</sub> = mean((p−y)·x)&nbsp;&nbsp; ∂L/∂w<sub>2</sub> = mean((p−y)·y)&nbsp;&nbsp; ∂L/∂b = mean(p−y)</div>
    <p>Then take a small step downhill: <code>w ← w − η·∂L/∂w</code>. Intuition: a point we got wrong has a big <code>(p−y)</code>, so it tugs the weights hard in the direction that fixes it — the boundary swings toward misclassified points and settles in the gap. The <b>learning rate η</b> is the step size: too small and it crawls; too large and the weights overshoot and the boundary jitters or diverges.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ The catches</h3>
    <ul class="bullets">
      <li><b>Linear only.</b> One straight cut. Non-linearly-separable data (moons, rings, XOR) caps its accuracy no matter how long you train.</li>
      <li><b>Perfectly separable → exploding weights.</b> When a line splits the classes cleanly, the loss keeps dropping by making <code>|w|</code> bigger (sharper sigmoid), so weights grow without bound. Watch w₁/w₂ climb after it hits 100%. Real implementations add L2 regularisation to rein them in.</li>
      <li><b>Probabilities are only roughly calibrated.</b> The output is a genuine probability, not just a label — but it trusts that the world is logistic. It can be over- or under-confident, especially with little data or after the weights blow up.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> logistic regression runs a linear score through the sigmoid to get a probability, places its decision line where that probability is 0.5, and learns by nudging the weights against the error <code>(p − y)</code> until the cross-entropy bottoms out.</p>
  </div>`,
};
