// =====================================================================
//  Gradient Descent — the optimisation engine behind almost every model.
//
//  Drop a ball on a 2-D loss landscape f(x, y) (drawn as a heatmap),
//  pick a learning rate, and watch the update rule
//        p  <-  p - lr * grad(p)
//  feel its way downhill. A gentle elliptical bowl shows how a too-large
//  rate zig-zags and explodes; a bumpy surface shows getting stuck in a
//  local minimum.
// =====================================================================
import {
  el, ghostButton, slider, Loop, Plot,
  heat, clamp,
} from "../shared.js";

export default {
  key: "gradient-descent",
  name: "Gradient Descent",
  icon: "🏔️",
  category: "Foundations",
  tag: "Core",
  blurb: "Roll a ball down a loss landscape — tune the step size and watch it converge or explode.",
  eyebrow: "Foundations · Optimisation",
  title: 'Gradient Descent: <span class="grad">rolling downhill</span> to learn',
  lede:
    "Almost every model on this site learns the same way: define how wrong it is (a <em>loss</em>), then " +
    "repeatedly nudge its parameters in the direction that lowers the loss fastest. That direction is the " +
    "negative <em>gradient</em>. Drop a ball on the landscape, set a learning rate, and watch gradient descent " +
    "feel its way to the bottom.",

  mount(root) {
    // ---- surfaces: each knows f(x,y), grad(x,y), and its min/max over the window ----
    const surfaces = {
      bowl: {
        label: "Bowl (convex)",
        f: (x, y) => 0.12 * x * x + 1.0 * y * y,
        grad: (x, y) => [0.24 * x, 2.0 * y],
        // window is [-3,3]^2 → min 0 at origin, max at a corner
        fmin: 0,
        fmax: 0.12 * 9 + 1.0 * 9, // = 10.08
        start: [-2.4, 2.2],
        note: "A single global minimum at the origin — but it is an <em>ellipse</em>, far steeper along y than x.",
      },
      bumpy: {
        label: "Bumpy (local minima)",
        f: (x, y) => 0.05 * (x * x + y * y) + Math.sin(1.3 * x) * Math.cos(1.3 * y),
        grad: (x, y) => [
          0.10 * x + 1.3 * Math.cos(1.3 * x) * Math.cos(1.3 * y),
          0.10 * y - 1.3 * Math.sin(1.3 * x) * Math.sin(1.3 * y),
        ],
        // |sin*cos| ≤ 1 and 0.05*(x²+y²) ≤ 0.9 over the window
        fmin: -1 - 0.0, // a touch below the true min; good enough for stable colours
        fmax: 0.05 * 18 + 1, // ≈ 1.9
        start: [-1.9, 1.7],
        note: "Several dips. Where the ball lands depends entirely on where it starts.",
      },
    };

    // ---- state ----
    let surfKey = "bowl";
    let surf = surfaces[surfKey];
    let lr = 0.15;
    let p = surf.start.slice();          // current point [x, y]
    let path = [p.slice()];              // visited points
    let iter = 0;
    let status = "ready";                // ready | descending | converged | diverged
    let lastGradMag = Infinity;

    const CONVERGE = 1e-3;               // ‖grad‖ threshold
    const DIVERGE_POS = 30;              // |x| or |y| this far past window = blown up
    const DIVERGE_LOSS = 1e6;

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Heatmap of a loss surface; cool colours are low loss, warm colours high. A path traces the ball rolling downhill." });
    const plot = new Plot(canvas, { xmin: -3, xmax: 3, ymin: -3, ymax: 3, pad: 16 });

    // surface selector
    const segBowl = el("button", { class: "active" }, "Bowl (convex)");
    const segBumpy = el("button", {}, "Bumpy (local minima)");
    const seg = el("div", { class: "seg-group" }, segBowl, segBumpy);

    const stepBtn = el("button", {}, "Step ▸");
    const runBtn = el("button", {}, "Run");
    const resetBtn = ghostButton("Reset");
    const lrSlider = slider({
      min: 0.001, max: 1.2, step: 0.001, value: lr,
      label: "Learning rate η =", format: (v) => v.toFixed(3),
    });

    const iterEl = el("span", { class: "v" }, "0");
    const lossEl = el("span", { class: "v" }, "—");
    const gradEl = el("span", { class: "v" }, "—");
    const lrEl = el("span", { class: "v" }, lr.toFixed(3));
    const statusEl = el("span", { class: "v" }, "ready");

    const verdict = el("p", { class: "verdict" },
      "Click anywhere on the surface to drop the ball, then press <b>Step</b> or <b>Run</b>. Try cranking the learning rate up on the bowl.");
    const noteEl = el("p", { class: "caption", style: { marginTop: "0.6rem" } });

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" },
            "Cool = low loss, warm = high. Click to set the start point. The arrow shows the step direction (−η∇f).")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Surface"),
            el("div", { class: "controls" }, seg),
            noteEl),
          el("div", { class: "panel" },
            el("h3", {}, "Optimiser state"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Iteration"), iterEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Loss f(p)"), lossEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "‖gradient‖"), gradEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Learning rate η"), lrEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Status"), statusEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, stepBtn, runBtn, resetBtn),
            el("div", { class: "controls" }, lrSlider.wrap)),
          el("div", { class: "legend" },
            el("span", { class: "legend-item" },
              el("span", { class: "legend-dot", style: { background: "#1f7ae0" } }), "low loss"),
            el("span", { class: "legend-item" },
              el("span", { class: "legend-dot", style: { background: "#e23b5a" } }), "high loss"),
            el("span", { class: "legend-item" },
              el("span", { class: "legend-dot", style: { background: "#fff" } }), "ball / path")))),
      verdict
    );

    // ---- maths helpers ----
    const norm01 = (v) => clamp((v - surf.fmin) / (surf.fmax - surf.fmin || 1), 0, 1);
    const inWindow = (x, y) => x >= plot.xmin && x <= plot.xmax && y >= plot.ymin && y <= plot.ymax;

    function gradMag(x, y) {
      const [gx, gy] = surf.grad(x, y);
      return Math.hypot(gx, gy);
    }

    // ---- one gradient-descent step:  p <- p - lr * grad(p) ----
    function step() {
      if (status === "converged" || status === "diverged") return;
      const [gx, gy] = surf.grad(p[0], p[1]);
      const gmag = Math.hypot(gx, gy);
      lastGradMag = gmag;

      // converged BEFORE stepping if the slope is already flat
      if (gmag < CONVERGE) { status = "converged"; refresh(); stopLoop(); return; }

      const nx = p[0] - lr * gx;
      const ny = p[1] - lr * gy;
      p = [nx, ny];
      path.push(p.slice());
      iter++;

      // divergence check: shot far outside the window, or loss exploded
      const loss = surf.f(nx, ny);
      if (Math.abs(nx) > DIVERGE_POS || Math.abs(ny) > DIVERGE_POS ||
          !isFinite(loss) || loss > DIVERGE_LOSS) {
        status = "diverged";
        refresh();
        stopLoop();
        return;
      }
      status = "descending";
      // re-check convergence at the new point (and report the post-step slope)
      lastGradMag = gradMag(nx, ny);
      if (lastGradMag < CONVERGE) status = "converged";
      refresh();
      if (status === "converged") stopLoop();
    }

    function resetTo(x, y) {
      p = [x, y];
      path = [p.slice()];
      iter = 0;
      status = "ready";
      lastGradMag = gradMag(x, y);
      refresh();
    }

    // ---- draw ----
    function draw() {
      // heatmap of normalised loss
      plot.shade((x, y) => heat(norm01(surf.f(x, y))), 64);

      // faint grid for spatial reference
      plot.grid(1, "rgba(255,255,255,0.05)");

      // path so far
      if (path.length > 1) {
        plot.path(path, "rgba(255,255,255,0.85)", 2.2);
        // small dots at each visited point to show step sizes
        for (let i = 0; i < path.length; i++) {
          const [x, y] = path[i];
          if (!inWindow(x, y)) continue;
          plot.point(x, y, 2.2, "rgba(255,255,255,0.55)", null, 0);
        }
      }

      // current ball
      const [cx, cy] = p;
      if (inWindow(cx, cy)) {
        // negative-gradient arrow (the next step direction), scaled for visibility
        const [gx, gy] = surf.grad(cx, cy);
        const gmag = Math.hypot(gx, gy);
        if (gmag > 1e-6 && status !== "converged") {
          // show the actual step vector -lr*grad, but cap its drawn length so a
          // huge step is visible as "long" without leaving the canvas
          let dx = -lr * gx, dy = -lr * gy;
          const stepLen = Math.hypot(dx, dy);
          const cap = 1.6; // data units
          if (stepLen > cap) { dx *= cap / stepLen; dy *= cap / stepLen; }
          drawArrow(cx, cy, cx + dx, cy + dy,
            stepLen > cap ? "#ff7a59" : "#ffd166");
        }
        // the ball itself
        plot.point(cx, cy, 6.5, "#ffffff", "#04111f", 2);
      }

      // mark the true global minimum of the bowl (origin) for reference
      if (surfKey === "bowl") {
        const mx = plot.sx(0), my = plot.sy(0);
        plot.ctx.save();
        plot.ctx.strokeStyle = "rgba(255,255,255,0.5)";
        plot.ctx.lineWidth = 1.5;
        plot.ctx.beginPath();
        plot.ctx.moveTo(mx - 6, my); plot.ctx.lineTo(mx + 6, my);
        plot.ctx.moveTo(mx, my - 6); plot.ctx.lineTo(mx, my + 6);
        plot.ctx.stroke();
        plot.ctx.restore();
      }
    }

    function drawArrow(x0, y0, x1, y1, color) {
      const c = plot.ctx;
      const px0 = plot.sx(x0), py0 = plot.sy(y0);
      const px1 = plot.sx(x1), py1 = plot.sy(y1);
      c.save();
      c.strokeStyle = color;
      c.fillStyle = color;
      c.lineWidth = 2.5;
      c.beginPath();
      c.moveTo(px0, py0);
      c.lineTo(px1, py1);
      c.stroke();
      // arrowhead
      const ang = Math.atan2(py1 - py0, px1 - px0);
      const h = 8;
      c.beginPath();
      c.moveTo(px1, py1);
      c.lineTo(px1 - h * Math.cos(ang - 0.4), py1 - h * Math.sin(ang - 0.4));
      c.lineTo(px1 - h * Math.cos(ang + 0.4), py1 - h * Math.sin(ang + 0.4));
      c.closePath();
      c.fill();
      c.restore();
    }

    // ---- readouts + verdict ----
    function refresh() {
      draw();
      const loss = surf.f(p[0], p[1]);
      iterEl.textContent = iter;
      lossEl.textContent = isFinite(loss) ? loss.toFixed(4) : "∞";
      gradEl.textContent = isFinite(lastGradMag) ? lastGradMag.toFixed(4) : "∞";
      lrEl.textContent = lr.toFixed(3);
      statusEl.textContent =
        status === "converged" ? "converged ✓" :
        status === "diverged" ? "DIVERGED ✗" :
        status === "descending" ? "descending" : "ready";

      updateVerdict();
    }

    function updateVerdict() {
      if (status === "diverged") {
        verdict.innerHTML =
          `<b>Diverged.</b> With η = <b>${lr.toFixed(3)}</b> each step overshoots the valley and lands somewhere even ` +
          `steeper, so the ball is flung uphill — loss is exploding to infinity. Lower the learning rate and reset.`;
      } else if (status === "converged") {
        const atGlobal = surfKey === "bowl" || (Math.abs(surf.f(p[0], p[1]) - surf.fmin) < 0.15);
        verdict.innerHTML =
          `<b>Converged</b> after <b>${iter}</b> steps — the gradient is essentially flat (‖∇f‖ < 0.001). ` +
          (surfKey === "bumpy" && !atGlobal
            ? `But notice the loss is <b>${surf.f(p[0], p[1]).toFixed(3)}</b>, not the lowest possible: the ball settled in a <b>local</b> minimum. Drop it elsewhere and you may find a deeper one.`
            : `This is the bottom of the bowl.`);
      } else if (status === "descending") {
        if (surfKey === "bowl" && lr > 1.0) {
          verdict.innerHTML =
            `η = <b>${lr.toFixed(3)}</b> is past the stability limit for the steep y-direction (η &gt; 1 ` +
            `makes the factor |1 − 2η| exceed 1). Each step grows — the ball is about to be flung uphill and <b>diverge</b>.`;
        } else if (surfKey === "bowl" && lr > 0.5) {
          verdict.innerHTML =
            `Watch the path <b>zig-zag</b>: the bowl is steep along y and shallow along x. Above η = 0.5 the y-update ` +
            `<em>overshoots</em> and flips sign each step (it still shrinks while η &lt; 1), bouncing across the steep ` +
            `direction while crawling along the shallow one. Push η past 1 and it diverges.`;
        } else if (lr < 0.02) {
          verdict.innerHTML =
            `Step <b>${iter}</b>: loss <b>${surf.f(p[0], p[1]).toFixed(4)}</b>. With such a tiny η the ball is ` +
            `<b>crawling</b> — correct direction, but it will take ages. Nudge the learning rate up.`;
        } else {
          verdict.innerHTML =
            `Step <b>${iter}</b>: loss falling to <b>${surf.f(p[0], p[1]).toFixed(4)}</b>, ‖∇f‖ = ` +
            `<b>${lastGradMag.toFixed(3)}</b>. Each move is downhill, opposite the gradient. Keep going.`;
        }
      } else {
        verdict.innerHTML =
          `Ball placed at (<b>${p[0].toFixed(2)}</b>, <b>${p[1].toFixed(2)}</b>), loss <b>${surf.f(p[0], p[1]).toFixed(3)}</b>. ` +
          `Press <b>Step</b> for one update, or <b>Run</b> to animate.`;
      }
    }

    // ---- animation loop: a few steps per second ----
    let acc = 0;
    const loop = new Loop((dt) => {
      acc += dt;
      if (acc >= 0.18) { acc = 0; step(); }
    });
    function stopLoop() {
      if (loop.running) { loop.stop(); runBtn.textContent = "Run"; }
    }

    // ---- events ----
    stepBtn.addEventListener("click", () => { stopLoop(); step(); });
    runBtn.addEventListener("click", () => {
      if (status === "converged" || status === "diverged") resetTo(p[0], p[1]);
      loop.toggle();
      runBtn.textContent = loop.running ? "Pause" : "Run";
    });
    resetBtn.addEventListener("click", () => { stopLoop(); resetTo(surf.start[0], surf.start[1]); });

    lrSlider.input.addEventListener("input", () => {
      lr = parseFloat(lrSlider.input.value);
      lrEl.textContent = lr.toFixed(3);
      // changing the rate while sitting at a converged/diverged point re-arms it
      if (status === "converged" || status === "diverged") status = "ready";
      refresh();
    });

    function selectSurface(key) {
      surfKey = key;
      surf = surfaces[key];
      segBowl.classList.toggle("active", key === "bowl");
      segBumpy.classList.toggle("active", key === "bumpy");
      noteEl.innerHTML = surf.note;
      stopLoop();
      resetTo(surf.start[0], surf.start[1]);
    }
    segBowl.addEventListener("click", () => selectSurface("bowl"));
    segBumpy.addEventListener("click", () => selectSurface("bumpy"));

    canvas.addEventListener("pointerdown", (e) => {
      const { x, y } = plot.eventToData(e);
      if (!inWindow(x, y)) return;
      stopLoop();
      resetTo(x, y);
    });

    // ---- init ----
    noteEl.innerHTML = surf.note;
    resetTo(surf.start[0], surf.start[1]);

    return {
      onEnter() { draw(); },
      onLeave() { stopLoop(); },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: get less wrong, one nudge at a time</h3>
    <p>A model has <b>parameters</b> (weights) and a <b>loss</b> — a single number that says how badly it is doing on the data. Learning means finding the parameters that make that number as small as possible. We almost never solve for the minimum in one shot; instead we start somewhere and take small downhill steps. Picture the loss as a landscape with the parameters as the floor coordinates and height as the loss: gradient descent is a ball rolling to the lowest valley it can reach.</p>
  </div>
  <div class="lesson-block">
    <h3>🧭 The gradient points uphill — so we go the other way</h3>
    <p>The <b>gradient</b> ∇f is the vector of partial derivatives. At any point it points in the direction of <em>steepest increase</em> of the loss, and its length is how steep that climb is. To go <em>down</em>, we step in the opposite direction. The orange/amber arrow on the canvas is exactly that step vector, <code>−η∇f</code>.</p>
    <ol class="steps">
      <li><b>Measure the slope.</b> Compute the gradient ∇f at the current point.</li>
      <li><b>Step downhill.</b> Move a little way along <code>−∇f</code>, scaled by the learning rate η.</li>
      <li><b>Repeat</b> until the slope is essentially flat (‖∇f‖ ≈ 0) — you're at a minimum.</li>
    </ol>
  </div>
  <div class="lesson-block">
    <h3>📐 The update rule</h3>
    <p>That whole loop is one tiny equation, applied over and over to the parameter vector θ:</p>
    <div class="formula">θ &larr; θ − η ∇L(θ)</div>
    <p>Here η (eta) is the <b>learning rate</b> — the step size. For the bowl on screen the surface is <code>f = 0.12·x² + y²</code>, so the gradient is <code>∇f = (0.24·x, 2·y)</code> and each step is <code>x ← x − η·0.24·x</code>, <code>y ← y − η·2·y</code>. That is the entire algorithm — no magic.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ Learning rate: too small crawls, too big explodes</h3>
    <ul class="bullets">
      <li><b>η too small</b> → the ball inches forward. It heads the right way but takes forever. Set η near 0.005 on the bowl and watch it crawl.</li>
      <li><b>η too big</b> → it <em>overshoots</em> the valley and lands on the far wall, even higher up. The bowl is an <b>ellipse</b> (steep along y, shallow along x). For the y-direction one step multiplies the offset by <code>1 − 2η</code>: once η &gt; 0.5 that factor goes negative, so the path flips sides and <b>zig-zags</b> across the steep direction (it still shrinks while η &lt; 1). Push η past <b>1.0</b> and <code>|1 − 2η| &gt; 1</code> — every step grows, the loss runs off to infinity. That's <b>divergence</b>.</li>
      <li>The sweet spot is "as large as you can get away with." Real training schedules often <em>shrink</em> η over time to get fast progress early and a stable landing late.</li>
    </ul>
  </div>
  <div class="lesson-block">
    <h3>🕳️ Local vs global minima</h3>
    <p>The bowl is <b>convex</b> — one valley, so wherever you drop the ball it finds the single global minimum. The <b>bumpy</b> surface is not: it has several dips. Gradient descent only ever goes downhill, so it gets trapped in whichever valley it happened to start near, even if a deeper one sits next door. Click different start points and watch it settle at different losses. In practice this is why we use randomness (random initialisation, <em>stochastic</em> gradient descent, momentum) to jiggle past shallow traps.</p>
    <table class="mini-table">
      <thead><tr><th>Symptom</th><th>Cause</th><th>Fix</th></tr></thead>
      <tbody>
        <tr><td>Loss barely moves</td><td>η too small</td><td>increase η</td></tr>
        <tr><td>Loss zig-zags / grows</td><td>η too big</td><td>decrease η</td></tr>
        <tr><td>Settles but not lowest</td><td>local minimum</td><td>restart / add momentum</td></tr>
      </tbody>
    </table>
  </div>
  <div class="lesson-block">
    <h3>🔌 This is the engine inside everything</h3>
    <p>Linear regression, logistic regression, and every neural network on this site learn by exactly this loop: define a loss, compute its gradient with respect to the weights (back-propagation is just the chain rule doing that efficiently), and apply <code>θ ← θ − η∇L</code>. The only differences are the shape of the loss surface and how the gradient is computed — the descent itself is what you just watched.</p>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> Gradient descent repeatedly steps the parameters a little way <em>against</em> the gradient of the loss — small enough not to overshoot, large enough to actually move — and that single rule, <code>θ ← θ − η∇L(θ)</code>, is how nearly every model learns.</p>
  </div>`,
};
