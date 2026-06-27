// =====================================================================
//  Linear Regression — the line of best fit.
//
//  Fit y = w·x + b to a cloud of draggable 2-D points.
//    - The SOLID line is the exact least-squares (OLS) solution, found
//      instantly from the closed-form normal equations.
//    - The optional DASHED line is gradient descent crawling toward the
//      same answer, to show the two routes land on one convex minimum.
//  Click empty space to add a point, drag a point to move it; the fit
//  recomputes live and the residuals show the errors being squared.
// =====================================================================
import {
  el, button, ghostButton, slider, Loop, Plot,
  linearData, clamp, mean, withAlpha,
} from "../shared.js";

export default {
  key: "linear-regression",
  name: "Linear Regression",
  icon: "📈",
  category: "Regression",
  tag: "OLS",
  blurb: "Drag points and watch the best-fit line chase them by minimising squared error.",
  eyebrow: "Regression · Supervised",
  title: 'Linear Regression: the <span class="grad">line of best fit</span>',
  lede:
    "The oldest trick in statistics and still everywhere. Given points, find the straight line that sits as close " +
    "as possible to all of them — by making the total <em>squared</em> vertical error as small as possible. " +
    "Click to add points or drag them around, and watch the line, and its error, respond instantly.",

  mount(root) {
    // ---- state ----
    let points = [];            // { x, y }
    let w = 0, b = 0;           // OLS solution (solid line)
    let mse = 0, r2 = 0;
    let showResiduals = true;
    let showSquares = false;

    // gradient-descent line (dashed) — only active while the loop runs
    let gdOn = false;
    let gw = 0, gb = 0;         // current GD parameters
    let lr = 0.25;              // learning rate
    let gdStep = 0;

    let drag = null;            // index of point being dragged, or null

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Scatter plot of points with a straight line of best fit and vertical residual segments." });
    const plot = new Plot(canvas, { xmin: -1.1, xmax: 1.1, ymin: -1.1, ymax: 1.1, pad: 18 });

    const newDataBtn = ghostButton("New data");
    const clearBtn = ghostButton("Clear");
    const resBtn = button("Hide residuals");
    const sqBtn = ghostButton("Show squared error");

    const gdBtn = button("Animate gradient descent ▸");
    const lrSlider = slider({ min: 0.01, max: 0.6, step: 0.01, value: lr, label: "Learning rate η =", format: (v) => v.toFixed(2) });
    const gdReseedBtn = ghostButton("Re-randomise GD line");

    const wEl = el("span", { class: "v" }, "—");
    const bEl = el("span", { class: "v" }, "—");
    const mseEl = el("span", { class: "v" }, "—");
    const r2El = el("span", { class: "v" }, "—");
    const nEl = el("span", { class: "v" }, "0");

    const gdwEl = el("span", { class: "v" }, "—");
    const gdmseEl = el("span", { class: "v" }, "—");
    const gdStepEl = el("span", { class: "v" }, "0");
    const gdPanel = el("div", { class: "panel", style: { display: "none" } },
      el("h3", {}, "Gradient-descent line"),
      el("div", { class: "kv" }, el("span", { class: "k" }, "ŵ, b̂ (current)"), gdwEl),
      el("div", { class: "kv" }, el("span", { class: "k" }, "MSE (descending)"), gdmseEl),
      el("div", { class: "kv" }, el("span", { class: "k" }, "Steps taken"), gdStepEl),
      el("div", { class: "controls" }, lrSlider.wrap),
      el("div", { class: "controls" }, gdReseedBtn));

    const legend = el("div", { class: "legend" },
      el("span", { class: "legend-item" }, el("span", { class: "legend-line", style: { background: "#5ad2ff" } }), "OLS best fit (exact)"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-line", style: { background: "#ff7a59" } }), "Gradient descent (iterative)"),
      el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { background: "#9aa4c6" } }), "data point"));

    const verdict = el("p", { class: "verdict" },
      "Add or drag points and the cyan line re-solves instantly to balance them.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" }, "Click empty space to add a point. Drag any point to move it — the fit re-solves live.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Best-fit line"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "slope w"), wEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "intercept b"), bEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "MSE (cost)"), mseEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "R²"), r2El),
            el("div", { class: "kv" }, el("span", { class: "k" }, "points n"), nEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, resBtn, sqBtn),
            el("div", { class: "controls" }, gdBtn),
            el("div", { class: "controls" }, newDataBtn, clearBtn)),
          gdPanel,
          legend)),
      verdict
    );

    // ---- maths: closed-form ordinary least squares ----
    // w = Σ((xi-x̄)(yi-ȳ)) / Σ((xi-x̄)²) ;  b = ȳ - w·x̄
    function fitOLS() {
      const n = points.length;
      if (n < 2) { w = 0; b = n === 1 ? points[0].y : 0; mse = 0; r2 = 0; return; }
      const xbar = mean(points.map((p) => p.x));
      const ybar = mean(points.map((p) => p.y));
      let sxy = 0, sxx = 0;
      for (const p of points) {
        const dx = p.x - xbar;
        sxy += dx * (p.y - ybar);
        sxx += dx * dx;
      }
      w = sxx < 1e-12 ? 0 : sxy / sxx;     // all x equal → vertical cloud, slope undefined → flat
      b = ybar - w * xbar;
      mse = costFor(w, b);
      // R² = 1 - SS_res / SS_tot
      let ssRes = 0, ssTot = 0;
      for (const p of points) {
        const yhat = w * p.x + b;
        ssRes += (yhat - p.y) ** 2;
        ssTot += (p.y - ybar) ** 2;
      }
      r2 = ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
    }

    // mean squared error for an arbitrary line — this is the cost J(w,b)
    function costFor(ww, bb) {
      if (!points.length) return 0;
      let s = 0;
      for (const p of points) s += (ww * p.x + bb - p.y) ** 2;
      return s / points.length;
    }

    // one gradient-descent step on J(w,b) = MSE
    //   dJ/dw = (2/n)Σ(ŷi-yi)xi ;  dJ/db = (2/n)Σ(ŷi-yi)
    function gdTick() {
      const n = points.length;
      if (n < 2) return;
      let dw = 0, db = 0;
      for (const p of points) {
        const err = gw * p.x + gb - p.y;
        dw += err * p.x;
        db += err;
      }
      dw = (2 / n) * dw;
      db = (2 / n) * db;
      gw -= lr * dw;
      gb -= lr * db;
      gdStep++;
    }

    function reseedGD() {
      gw = (Math.random() * 2 - 1) * 1.6;     // random-ish slope
      gb = (Math.random() * 2 - 1) * 0.8;     // random-ish intercept
      gdStep = 0;
    }

    // ---- drawing ----
    function lineYAt(x, ww, bb) { return ww * x + bb; }

    function draw() {
      plot.clear("#070d1a");
      plot.grid(0.5);
      plot.axes();

      plot.ctx.save();
      plot.clipToPlot();

      // residuals + optional squares (drawn under the points & line)
      if (showResiduals && points.length >= 2) {
        for (const p of points) {
          const yhat = lineYAt(p.x, w, b);
          plot.line(p.x, p.y, p.x, yhat, withAlpha("#ff7a59", 0.55), 1.5);
          if (showSquares) {
            // a square whose side = |residual| in data units, drawn to the right of the point
            const err = p.y - yhat;                 // signed vertical error
            const sidePx = Math.abs(plot.sy(p.y) - plot.sy(yhat)); // pixel height of the error
            const x0 = plot.sx(p.x);
            const yTop = Math.min(plot.sy(p.y), plot.sy(yhat));
            plot.ctx.fillStyle = withAlpha("#ff7a59", 0.14);
            plot.ctx.strokeStyle = withAlpha("#ff7a59", 0.35);
            plot.ctx.lineWidth = 1;
            // square extends rightward from the residual segment
            plot.ctx.fillRect(x0, yTop, sidePx, sidePx);
            plot.ctx.strokeRect(x0, yTop, sidePx, sidePx);
            void err;
          }
        }
      }

      // gradient-descent dashed line
      if (gdOn && points.length >= 2) {
        plot.line(plot.xmin, lineYAt(plot.xmin, gw, gb), plot.xmax, lineYAt(plot.xmax, gw, gb),
          "#ff7a59", 2.5, [8, 6]);
      }

      // OLS solid line across the window
      if (points.length >= 2) {
        plot.line(plot.xmin, lineYAt(plot.xmin, w, b), plot.xmax, lineYAt(plot.xmax, w, b),
          "#5ad2ff", 2.75);
      } else if (points.length === 1) {
        // with one point a horizontal line through it is the best we can say
        plot.line(plot.xmin, b, plot.xmax, b, withAlpha("#5ad2ff", 0.5), 2, [4, 4]);
      }

      // points (the one being dragged gets a halo)
      points.forEach((p, i) => {
        if (i === drag) {
          plot.ctx.beginPath();
          plot.ctx.arc(plot.sx(p.x), plot.sy(p.y), 10, 0, Math.PI * 2);
          plot.ctx.fillStyle = withAlpha("#5ad2ff", 0.22);
          plot.ctx.fill();
        }
        plot.point(p.x, p.y, 5, "#cfd6f5", "rgba(7,13,26,0.85)", 1.5);
      });

      plot.ctx.restore();
    }

    function refresh() {
      fitOLS();
      draw();
      const n = points.length;
      nEl.textContent = n;
      if (n >= 2) {
        wEl.textContent = w.toFixed(3);
        bEl.textContent = b.toFixed(3);
        mseEl.textContent = mse.toFixed(4);
        r2El.textContent = r2.toFixed(3);
      } else {
        wEl.textContent = "—"; bEl.textContent = n === 1 ? b.toFixed(3) : "—";
        mseEl.textContent = "—"; r2El.textContent = "—";
      }

      if (gdOn) {
        gdwEl.textContent = `${gw.toFixed(2)}, ${gb.toFixed(2)}`;
        gdmseEl.textContent = points.length >= 2 ? costFor(gw, gb).toFixed(4) : "—";
        gdStepEl.textContent = gdStep;
      }

      updateVerdict();
    }

    function updateVerdict() {
      if (points.length < 2) {
        verdict.innerHTML = "Add at least <b>two</b> points — a line needs two anchors before there is anything to fit.";
        return;
      }
      const pct = (clamp(r2, 0, 1) * 100).toFixed(0);
      const tilt = Math.abs(w) < 0.05 ? "sits almost flat" : (w > 0 ? "tilts upward" : "tilts downward");
      let v = `The line ${tilt} to balance the points: <b>y = ${w.toFixed(2)}·x + ${b.toFixed(2)}</b>. ` +
        `It explains <b>R² = ${r2.toFixed(2)}</b> (${pct}%) of the variance in y, with a mean squared error of <b>${mse.toFixed(3)}</b>.`;
      if (gdOn && points.length >= 2) {
        const gap = costFor(gw, gb) - mse;
        if (gap < 1e-5) v += ` Gradient descent has <b>converged</b> onto the exact line.`;
        else v += ` Gradient descent is still <b>${gap.toFixed(4)}</b> above the optimum — watch it slide on.`;
      }
      verdict.innerHTML = v;
    }

    // ---- gradient-descent animation loop ----
    let acc = 0;
    const loop = new Loop((dt) => {
      acc += dt;
      // take a few small steps per frame so convergence is visible but smooth
      if (acc >= 0.016) {
        acc = 0;
        for (let k = 0; k < 3; k++) gdTick();
      }
      draw();
      gdwEl.textContent = `${gw.toFixed(2)}, ${gb.toFixed(2)}`;
      gdmseEl.textContent = points.length >= 2 ? costFor(gw, gb).toFixed(4) : "—";
      gdStepEl.textContent = gdStep;
      updateVerdict();
    });

    // ---- events ----
    resBtn.addEventListener("click", () => {
      showResiduals = !showResiduals;
      resBtn.textContent = showResiduals ? "Hide residuals" : "Show residuals";
      draw();
    });
    sqBtn.addEventListener("click", () => {
      showSquares = !showSquares;
      sqBtn.textContent = showSquares ? "Hide squared error" : "Show squared error";
      draw();
    });
    newDataBtn.addEventListener("click", () => {
      points = linearData(22, 0.7, 0.05, 0.16).map((p) => ({ x: p.x, y: p.y }));
      if (gdOn) reseedGD();
      refresh();
    });
    clearBtn.addEventListener("click", () => {
      points = [];
      if (gdOn) reseedGD();
      refresh();
    });

    gdBtn.addEventListener("click", () => {
      gdOn = !gdOn;
      if (gdOn) {
        reseedGD();
        gdPanel.style.display = "";
        gdBtn.textContent = "Stop gradient descent ◼";
        loop.start();
      } else {
        loop.stop();
        gdPanel.style.display = "none";
        gdBtn.textContent = "Animate gradient descent ▸";
        refresh();
      }
    });
    lrSlider.input.addEventListener("input", () => { lr = parseFloat(lrSlider.input.value); });
    gdReseedBtn.addEventListener("click", () => { reseedGD(); });

    // ---- pointer interaction: add point / drag point ----
    const HIT_PX = 14;  // pixel radius for grabbing an existing point

    function nearestPoint(px, py) {
      let best = -1, bestD = HIT_PX * HIT_PX;
      points.forEach((p, i) => {
        const dx = plot.sx(p.x) - px, dy = plot.sy(p.y) - py;
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      });
      return best;
    }

    canvas.addEventListener("pointerdown", (e) => {
      const { x, y, px, py } = plot.eventToData(e);
      const hit = nearestPoint(px, py);
      if (hit >= 0) {
        drag = hit;
        canvas.setPointerCapture(e.pointerId);
        draw();
      } else {
        // add a point only if inside the data window
        if (x < plot.xmin || x > plot.xmax || y < plot.ymin || y > plot.ymax) return;
        points.push({ x, y });
        refresh();
      }
    });

    canvas.addEventListener("pointermove", (e) => {
      if (drag == null) return;
      const { x, y } = plot.eventToData(e);
      points[drag].x = clamp(x, plot.xmin, plot.xmax);
      points[drag].y = clamp(y, plot.ymin, plot.ymax);
      refresh();
    });

    function endDrag(e) {
      if (drag == null) return;
      drag = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
      refresh();
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // ---- init ----
    points = linearData(22, 0.7, 0.05, 0.16).map((p) => ({ x: p.x, y: p.y }));
    refresh();

    return {
      onEnter() { refresh(); },
      onLeave() {
        loop.stop();
        if (gdOn) {
          gdOn = false;
          gdPanel.style.display = "none";
          gdBtn.textContent = "Animate gradient descent ▸";
        }
      },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: predict a number from a number</h3>
    <p>Linear regression is <b>supervised learning</b> for a continuous output. Each example is an input <code>x</code> paired with a known answer <code>y</code>, and the model is the simplest function you can write: a straight line, <code>ŷ = w·x + b</code>. Once you have <code>w</code> (slope) and <code>b</code> (intercept), you can predict <code>y</code> for any new <code>x</code>. House price from size, blood pressure from dose, sales from ad spend — whenever you expect "more of this means roughly proportionally more of that," this is the first model to reach for.</p>
  </div>
  <div class="lesson-block">
    <h3>📏 Residuals — the errors you can see</h3>
    <p>The orange segments above are the <b>residuals</b>: for each point, the <em>vertical</em> gap between where the line predicts it should be (<code>ŷ<sub>i</sub></code>) and where it actually is (<code>y<sub>i</sub></code>).</p>
    <div class="formula">residual<sub>i</sub> = y<sub>i</sub> − ŷ<sub>i</sub> = y<sub>i</sub> − (w·x<sub>i</sub> + b)</div>
    <p>A good line makes these gaps small overall. But "overall" needs a precise meaning — and that is where squaring comes in.</p>
  </div>
  <div class="lesson-block">
    <h3>➗ Why we square the errors</h3>
    <p>We score a line by its <b>mean squared error</b> — the cost <code>J(w,b)</code>. Square each residual, then average:</p>
    <div class="formula">J(w, b) = MSE = (1/n) Σ<sub>i</sub> ( w·x<sub>i</sub> + b − y<sub>i</sub> )²</div>
    <p>Squaring does three useful things. It makes positive and negative gaps both count (they can't cancel out). It <b>punishes big misses</b> far harder than small ones, so the line refuses to leave any point badly stranded. And critically, it makes <code>J</code> a smooth, <b>convex bowl</b> in <code>(w, b)</code> — one single lowest point, no traps. Tick "Show squared error" to literally see the squares whose total area we are shrinking.</p>
  </div>
  <div class="lesson-block">
    <h3>🧮 Two routes to the same bottom</h3>
    <p>Because the bowl is convex, there is exactly one best line — and two ways to find it:</p>
    <ol class="steps">
      <li><b>Closed form (normal equations).</b> Set the derivatives of <code>J</code> to zero and solve. For one input the answer is exact and instant — that's the cyan line:
        <div class="formula">w = Σ(x<sub>i</sub>−x̄)(y<sub>i</sub>−ȳ) ⁄ Σ(x<sub>i</sub>−x̄)²&nbsp;&nbsp;&nbsp;&nbsp;b = ȳ − w·x̄</div></li>
      <li><b>Gradient descent.</b> Start anywhere and repeatedly step downhill, using the slope of the cost:
        <div class="formula">∂J/∂w = (2/n) Σ(ŷ<sub>i</sub>−y<sub>i</sub>)·x<sub>i</sub>&nbsp;&nbsp;&nbsp;&nbsp;∂J/∂b = (2/n) Σ(ŷ<sub>i</sub>−y<sub>i</sub>)</div>
        Press <b>Animate gradient descent</b> and watch the orange dashed line crawl down the bowl and settle exactly onto the cyan one. Same minimum, reached the slow way — exactly the descent you met in the Gradient Descent game.</li>
    </ol>
    <p>For one or two variables the closed form wins. With millions of features the matrix solve gets expensive, so real systems lean on gradient descent (and its mini-batch cousins) — but they are aiming at the identical optimum.</p>
  </div>
  <div class="lesson-block">
    <h3>📈 R²: how much did the line explain?</h3>
    <p>MSE is in the squared units of <code>y</code>, which is hard to read. <b>R²</b> rescales it to "fraction of the variance the line explains":</p>
    <div class="formula">R² = 1 − ( Σ(y<sub>i</sub>−ŷ<sub>i</sub>)² ⁄ Σ(y<sub>i</sub>−ȳ)² )</div>
    <p>R² = 1 is a perfect fit; R² = 0 means the line did no better than just predicting the average <code>ȳ</code>; it can even go negative for a truly bad line. Drag the points into a tight diagonal and watch R² climb toward 1; scatter them into a blob and watch it collapse toward 0.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ The catches</h3>
    <ul class="bullets">
      <li><b>Outliers dominate.</b> Because errors are <em>squared</em>, one wild point pulls the whole line toward itself. Drag a single point far away and watch the line lurch — squared error is sensitive to outliers (robust losses like absolute error fight this).</li>
      <li><b>It can only draw straight lines.</b> If the true relationship curves, a line underfits no matter how you tune it. You'd add polynomial or other features first.</li>
      <li><b>Correlation isn't cause.</b> A good fit says x and y move together, not that x <em>makes</em> y happen.</li>
      <li><b>Learning rate matters.</b> Crank η too high and gradient descent overshoots and diverges; too low and it crawls. The closed form sidesteps this entirely.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> linear regression picks the slope and intercept that make the total <em>squared</em> vertical error smallest — a convex problem with one exact answer that the normal equations find instantly and gradient descent reaches step by step.</p>
  </div>`,
};
