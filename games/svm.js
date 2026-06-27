// =====================================================================
//  Support Vector Machine — the widest street.
//
//  A linear soft-margin SVM trained by Pegasos (stochastic sub-gradient
//  descent on the hinge-loss objective). We visualise the decision line
//  f(x)=w·x+b=0, the two margin lines f=±1 that bound the "street", and
//  ring the support vectors (points on or inside the margin) that alone
//  determine where the line sits.
// =====================================================================
import {
  el, button, ghostButton, slider, Loop, Plot,
  blobs, classColor, withAlpha, clamp, randInt,
} from "../shared.js";

export default {
  key: "svm",
  name: "Support Vector Machine",
  icon: "⚔️",
  category: "Classification",
  tag: "Margin",
  blurb: "Find the separating line with the biggest possible margin between classes.",
  eyebrow: "Classification · Supervised",
  title: 'Support Vector Machine: the <span class="grad">widest street</span>',
  lede:
    "Lots of lines can separate two classes — an SVM picks the one with the most breathing room, " +
    "the widest empty street between them. Only the points right on the kerb (the <em>support vectors</em>) " +
    "matter. Add points and watch the street, and the support vectors that define it, snap into place.",

  mount(root) {
    // ---- model state ----
    // f(x) = w·x + b,  prediction = sign(f).  Labels y in {-1,+1}.
    let w = [0, 0];
    let b = 0;
    let t = 0;                // Pegasos step counter (drives the schedule)
    let C = 1.0;             // soft-margin strength; bigger C = harder margin
    const EPS = 1e-3;        // tolerance for "on the margin" test

    let points = [];         // { x, y, label }  label in {-1,+1}
    let addClass = +1;       // which class new clicks create

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Scatter plot of two classes with a separating line, two dashed margin lines bounding the street, and rings around the support vectors." });
    const plot = new Plot(canvas, { xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2, pad: 16 });

    const trainBtn = button("Train ▸");
    const resetBtn = ghostButton("Reset");
    const newDataBtn = ghostButton("New data");

    const cSlider = slider({ min: -1, max: 2, step: 0.05, value: 0,
      label: "C (margin hardness) =", format: (v) => (10 ** v).toFixed(2) });

    // class picker seg-group
    const segNeg = el("button", { class: "" }, "Class −1");
    const segPos = el("button", { class: "active" }, "Class +1");
    const seg = el("div", { class: "seg-group" }, segNeg, segPos);

    const marginEl = el("span", { class: "v" }, "—");
    const svEl = el("span", { class: "v" }, "—");
    const normEl = el("span", { class: "v" }, "—");
    const biasEl = el("span", { class: "v" }, "—");
    const stepEl = el("span", { class: "v" }, "0");

    const legend = el("div", { class: "legend" },
      el("span", { class: "legend-item" },
        el("span", { class: "legend-dot", style: { background: classColor(0) } }), "Class +1"),
      el("span", { class: "legend-item" },
        el("span", { class: "legend-dot", style: { background: classColor(1) } }), "Class −1"),
      el("span", { class: "legend-item" },
        el("span", { class: "legend-line", style: { background: "#eaeefb" } }), "decision f=0"),
      el("span", { class: "legend-item" },
        el("span", { class: "legend-line", style: { background: "rgba(234,238,251,0.55)" } }), "margins f=±1"),
      el("span", { class: "legend-item" },
        el("span", { class: "legend-dot", style: { background: "transparent", border: "2px solid #ffd166", boxSizing: "border-box" } }), "support vectors"));

    const verdict = el("p", { class: "verdict" },
      "Press <b>Train</b> to let Pegasos roll the street out between the two classes. " +
      "Click the canvas to add a point of the selected class.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" },
            "Click empty space to add a point (of the chosen class). The dashed lines are the margins; ringed points are the support vectors that hold the street open.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Model state"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Margin width 2/‖w‖"), marginEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Support vectors"), svEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "‖w‖"), normEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "bias b"), biasEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Pegasos steps"), stepEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Add points as"),
            el("div", { class: "controls" }, seg)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, trainBtn, resetBtn, newDataBtn),
            el("div", { class: "controls" }, cSlider.wrap)),
          legend)),
      verdict
    );

    // ---- maths helpers ----
    const f = (px, py) => w[0] * px + w[1] * py + b;
    const norm = () => Math.hypot(w[0], w[1]);

    // λ for Pegasos. Objective: (λ/2)‖w‖² + (1/n)Σ hinge.  Bigger C ⇒ smaller λ
    // ⇒ less regularisation ⇒ harder margin. λ = 1/(C·n).
    function lambda() {
      const n = Math.max(1, points.length);
      return 1 / (C * n);
    }

    // ---- data ----
    function seedData() {
      // two fairly separable blobs → labels -1 / +1
      const raw = blobs([[-0.5, -0.3], [0.5, 0.35]], 22, 0.14);
      points = raw.map((p) => ({ x: p.x, y: p.y, label: p.label === 0 ? -1 : +1 }));
    }

    function reset() {
      w = [0, 0]; b = 0; t = 0;
    }

    // ---- training: one Pegasos step ----
    // η = 1/(λ·t). Pick random i. If yᵢ·f(xᵢ) < 1 (margin violated):
    //   w ← (1-ηλ)w + η·yᵢ·xᵢ ;  b ← b + η·yᵢ
    // else: w ← (1-ηλ)w   (just shrink toward zero).
    function pegasosStep() {
      if (points.length < 2) return;
      const lam = lambda();
      t += 1;
      const eta = 1 / (lam * t);
      const i = randInt(points.length);
      const p = points[i];
      const margin = p.label * f(p.x, p.y);
      const shrink = 1 - eta * lam;
      if (margin < 1) {
        w[0] = shrink * w[0] + eta * p.label * p.x;
        w[1] = shrink * w[1] + eta * p.label * p.y;
        b = b + eta * p.label;
      } else {
        w[0] = shrink * w[0];
        w[1] = shrink * w[1];
      }
    }

    // ---- support-vector test: yᵢ·f(xᵢ) ≤ 1 + ε ----
    function isSV(p) {
      return p.label * f(p.x, p.y) <= 1 + EPS;
    }
    function countSV() {
      return points.reduce((s, p) => s + (isSV(p) ? 1 : 0), 0);
    }

    // ---- draw ----
    // Class index for colour: +1 → 0 (cyan), -1 → 1 (coral).
    const colOf = (label) => classColor(label === +1 ? 0 : 1);

    function lineForLevel(level) {
      // points where w·x + b = level, clipped to the window. Returns [[x,y],[x,y]] or null.
      const [a, c] = w;
      const xmin = plot.xmin, xmax = plot.xmax, ymin = plot.ymin, ymax = plot.ymax;
      const pts = [];
      // intersect with vertical edges x=xmin, x=xmax  (solve for y, needs c≠0)
      if (Math.abs(c) > 1e-9) {
        for (const xv of [xmin, xmax]) {
          const yv = (level - b - a * xv) / c;
          if (yv >= ymin - 1e-9 && yv <= ymax + 1e-9) pts.push([xv, yv]);
        }
      }
      // intersect with horizontal edges y=ymin, y=ymax  (solve for x, needs a≠0)
      if (Math.abs(a) > 1e-9) {
        for (const yv of [ymin, ymax]) {
          const xv = (level - b - c * yv) / a;
          if (xv >= xmin - 1e-9 && xv <= xmax + 1e-9) pts.push([xv, yv]);
        }
      }
      if (pts.length < 2) return null;
      // pick the two most separated intersection points
      let best = [pts[0], pts[1]], bestD = -1;
      for (let i = 0; i < pts.length; i++)
        for (let j = i + 1; j < pts.length; j++) {
          const d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2;
          if (d > bestD) { bestD = d; best = [pts[i], pts[j]]; }
        }
      return best;
    }

    function draw() {
      plot.clear("#070d1a");

      const n = norm();
      const trained = n > 1e-6;

      // half-plane tint by sign(f) + a brighter band for the street (|f|<1)
      if (trained) {
        const cPos = classColor(0), cNeg = classColor(1);
        plot.shade((x, y) => {
          const v = f(x, y);
          if (Math.abs(v) <= 1) return "rgba(234,238,251,0.06)"; // the street
          return v > 0 ? withAlpha(cPos, 0.10) : withAlpha(cNeg, 0.10);
        }, 60);
      }

      plot.grid(0.5);

      // margin lines (dashed) and decision line (solid)
      if (trained) {
        const m1 = lineForLevel(+1), mm1 = lineForLevel(-1), d0 = lineForLevel(0);
        if (m1) plot.line(m1[0][0], m1[0][1], m1[1][0], m1[1][1], "rgba(234,238,251,0.5)", 1.5, [6, 6]);
        if (mm1) plot.line(mm1[0][0], mm1[0][1], mm1[1][0], mm1[1][1], "rgba(234,238,251,0.5)", 1.5, [6, 6]);
        if (d0) plot.line(d0[0][0], d0[0][1], d0[1][0], d0[1][1], "#eaeefb", 2.5);
      }

      // points; ring the support vectors
      for (const p of points) {
        const sv = trained && isSV(p);
        plot.point(p.x, p.y, 5, colOf(p.label), "rgba(7,13,26,0.85)", 1.5);
        if (sv) {
          const px = plot.sx(p.x), py = plot.sy(p.y);
          plot.ctx.beginPath();
          plot.ctx.arc(px, py, 9, 0, Math.PI * 2);
          plot.ctx.lineWidth = 2.5;
          plot.ctx.strokeStyle = "#ffd166";
          plot.ctx.stroke();
        }
      }
    }

    function refresh() {
      draw();
      const n = norm();
      const trained = n > 1e-6;
      stepEl.textContent = t;
      if (trained) {
        marginEl.textContent = (2 / n).toFixed(3);
        svEl.textContent = countSV();
        normEl.textContent = n.toFixed(3);
        biasEl.textContent = b.toFixed(3);
      } else {
        marginEl.textContent = "—";
        svEl.textContent = "—";
        normEl.textContent = "—";
        biasEl.textContent = "—";
      }

      if (!trained) {
        verdict.innerHTML = points.length < 2
          ? "Add at least one point of each class, then press <b>Train</b>."
          : "Untrained — <code>w</code> is zero so there is no street yet. Press <b>Train</b>.";
        return;
      }

      const sv = countSV();
      const width = 2 / n;
      // count misclassified (a point on the wrong side of the decision line)
      const wrong = points.reduce((s, p) => s + (p.label * f(p.x, p.y) < 0 ? 1 : 0), 0);
      let msg = `Margin width <b>${width.toFixed(3)}</b> (= 2/‖w‖). ` +
        `<b>${sv}</b> support vector${sv === 1 ? "" : "s"} sit on or inside the street — they alone pin the line. `;
      if (wrong > 0) {
        msg += `<b>${wrong}</b> point${wrong === 1 ? " is" : "s are"} on the wrong side; the soft margin (C) tolerates them to keep the street wide.`;
      } else {
        msg += `Every point is correctly classified. Drag C up to harden the margin, or add a point inside the street and re-train.`;
      }
      verdict.innerHTML = msg;
    }

    // ---- training loop: many sub-gradient updates per frame so it settles fast ----
    const UPDATES_PER_FRAME = 300;
    const loop = new Loop(() => {
      for (let k = 0; k < UPDATES_PER_FRAME; k++) pegasosStep();
      refresh();
    });

    function stopLoop() {
      loop.stop();
      trainBtn.textContent = "Train ▸";
    }

    // ---- events ----
    trainBtn.addEventListener("click", () => {
      if (points.length < 2) return;
      loop.toggle();
      trainBtn.textContent = loop.running ? "Pause" : "Train ▸";
    });
    resetBtn.addEventListener("click", () => { stopLoop(); reset(); refresh(); });
    newDataBtn.addEventListener("click", () => { stopLoop(); seedData(); reset(); refresh(); });

    cSlider.input.addEventListener("input", () => {
      C = 10 ** parseFloat(cSlider.input.value);
      // λ changed; if already trained, let the user re-train to see the new street.
      refresh();
    });

    segNeg.addEventListener("click", () => {
      addClass = -1; segNeg.classList.add("active"); segPos.classList.remove("active");
    });
    segPos.addEventListener("click", () => {
      addClass = +1; segPos.classList.add("active"); segNeg.classList.remove("active");
    });

    canvas.addEventListener("pointerdown", (e) => {
      const { x, y } = plot.eventToData(e);
      if (x < plot.xmin || x > plot.xmax || y < plot.ymin || y > plot.ymax) return;
      points.push({ x: clamp(x, plot.xmin, plot.xmax), y: clamp(y, plot.ymin, plot.ymax), label: addClass });
      refresh();
    });

    // ---- init ----
    C = 10 ** parseFloat(cSlider.input.value);
    seedData();
    reset();
    refresh();

    return {
      onEnter() { draw(); },
      onLeave() { stopLoop(); },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: separate two classes with confidence</h3>
    <p>Give a classifier two clouds of labelled points and ask for a line between them, and there are <em>infinitely many</em> answers — any line that doesn't touch a point will do. A <b>Support Vector Machine</b> insists on a particular one: the line with the <b>widest empty street</b> on either side. Intuitively, a fat buffer means new points have to drift a long way before they cross over, so the model generalises better.</p>
  </div>
  <div class="lesson-block">
    <h3>🛣️ What you see on screen</h3>
    <ol class="steps">
      <li><b>The decision line</b> (solid): <code>f(x) = w·x + b = 0</code>. Predict <code>+1</code> on one side, <code>−1</code> on the other.</li>
      <li><b>The two margin lines</b> (dashed): <code>f = +1</code> and <code>f = −1</code>. The gap between them is the street, and its width is <code>2 / ‖w‖</code>.</li>
      <li><b>Support vectors</b> (ringed): the points sitting <em>on</em> or <em>inside</em> the street. These are the only points that hold the line in place.</li>
    </ol>
    <p>Try it: move a point far from the street and re-train — nothing changes. Nudge a support vector and the whole street swings. That is the SVM's defining property.</p>
  </div>
  <div class="lesson-block">
    <h3>📐 The maths: maximise the margin</h3>
    <p>For a label <code>y<sub>i</sub> ∈ {−1, +1}</code>, "correct and outside the street" means <code>y<sub>i</sub>(w·x<sub>i</sub> + b) ≥ 1</code>. The street width is <code>2/‖w‖</code>, so a <em>wider</em> street means a <em>smaller</em> <code>‖w‖</code>. The hard-margin SVM is therefore:</p>
    <div class="formula">minimise &nbsp; ½‖w‖²&nbsp;&nbsp;&nbsp; subject to &nbsp; y<sub>i</sub>(w·x<sub>i</sub> + b) ≥ 1 &nbsp; for all i</div>
    <p>Real data overlaps, so we allow violations with a hinge loss and a strength <code>C</code> — the <b>soft margin</b>:</p>
    <div class="formula">minimise &nbsp; (λ/2)‖w‖² + (1/n) Σ<sub>i</sub> max(0, 1 − y<sub>i</sub>(w·x<sub>i</sub> + b))</div>
    <p>This game trains it with <b>Pegasos</b>, plain stochastic sub-gradient descent. Each step picks one point, uses learning rate <code>η = 1/(λt)</code>, and updates:</p>
    <div class="formula">if y<sub>i</sub>(w·x<sub>i</sub>+b) &lt; 1:&nbsp; w ← (1−ηλ)w + η·y<sub>i</sub>x<sub>i</sub>,&nbsp; b ← b + η·y<sub>i</sub><br>else:&nbsp; w ← (1−ηλ)w</div>
    <p>Points already safely outside the street only shrink <code>w</code> (widen the margin); points that are violating it pull the line toward them. We set <code>λ = 1/(C·n)</code>, so a bigger <code>C</code> means weaker regularisation = a <b>harder</b> margin.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ The catches</h3>
    <ul class="bullets">
      <li><b>C is a trade-off.</b> Large <code>C</code> punishes every violation hard → a narrow street that may hug noisy points (over-fit). Small <code>C</code> tolerates violations → a wide street that ignores outliers but mis-labels more. Slide <code>C</code> and re-train to feel it.</li>
      <li><b>Outliers can dominate.</b> A single point parked deep in the other class becomes a support vector and drags the line. The soft margin softens this, but does not erase it.</li>
      <li><b>Only linear here.</b> This SVM draws a straight line, so it can't separate rings or spirals. The famous <b>kernel trick</b> fixes that: replace dot products <code>x<sub>i</sub>·x<sub>j</sub></code> with a kernel <code>k(x<sub>i</sub>,x<sub>j</sub>)</code> (e.g. RBF), which is equivalent to finding a flat street in a higher-dimensional space — a curved boundary back in ours.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> an SVM chooses the separating line with the widest margin (<code>2/‖w‖</code>), and that line is decided entirely by the few support vectors sitting on the edge of the street — everything else is irrelevant.</p>
  </div>`,
};
