// =====================================================================
//  PCA — Principal Component Analysis in 2-D.
//
//  Find the orthogonal directions a point cloud varies along the most.
//    1. Mean-centre the points.
//    2. Build the 2x2 covariance matrix C.
//    3. Its eigenvectors are the principal axes; eigenvalues are the
//       variance along each. The larger eigenvalue = PC1.
//    4. "Reduce" by projecting every point onto PC1: 2-D -> 1-D.
//  The user reshapes the cloud (elongation + rotation) and watches the
//  principal axes rotate to follow its grain.
// =====================================================================
import {
  el, button, ghostButton, slider, Plot,
  gaussRandom, clamp, withAlpha,
} from "../shared.js";

export default {
  key: "pca",
  name: "PCA",
  icon: "📐",
  category: "Dimensionality",
  tag: "Unsup",
  blurb: "Discover the axes a cloud of data actually varies along — then drop the rest.",
  eyebrow: "Dimensionality reduction · Unsupervised",
  title: 'PCA: finding the directions that <span class="grad">matter</span>',
  lede:
    "Real data often lives in fewer dimensions than it appears to. PCA finds the directions of greatest <em>variance</em> — " +
    "the principal components — and lets you keep the important ones and throw the rest away. Reshape the cloud and watch " +
    "the principal axes rotate to follow its grain.",

  mount(root) {
    // ---- state ----
    let points = [];          // { x, y } — raw data
    let elong = 2.6;          // ratio a/b (spread/elongation)
    let angle = 0.5;          // rotation in radians
    let projecting = false;   // "project onto PC1" toggle
    const N = 120;

    // PCA results (filled by computePCA)
    let pca = null;           // { mx, my, l1, l2, v1, v2, explained }

    // ---- DOM ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Scatter plot of a correlated point cloud with the principal axes drawn as arrows from the mean." });
    const plot = new Plot(canvas, { xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2, pad: 16 });

    const regenBtn = button("Regenerate");
    const projBtn = ghostButton("Project onto PC1");
    const clearBtn = ghostButton("Clear points");

    const elongSlider = slider({ min: 1, max: 6, step: 0.1, value: elong,
      label: "Spread / elongation (a÷b)", format: (v) => v.toFixed(1) });
    const angleSlider = slider({ min: 0, max: 3.14, step: 0.01, value: angle,
      label: "Rotation angle (rad)", format: (v) => v.toFixed(2) });

    // variance-explained bars
    const bar1 = el("div", { class: "bar-fill", style: { width: "0%" } });
    const bar2 = el("div", { class: "bar-fill warm", style: { width: "0%" } });
    const bar1Val = el("span", { class: "bar-val" }, "—");
    const bar2Val = el("span", { class: "bar-val" }, "—");

    const l1El = el("span", { class: "v" }, "—");
    const l2El = el("span", { class: "v" }, "—");
    const angEl = el("span", { class: "v" }, "—");
    const ratioEl = el("span", { class: "v" }, "—");
    const nEl = el("span", { class: "v" }, "0");

    const verdict = el("p", { class: "verdict" },
      "Reshape the cloud with the sliders and watch the principal axes swing to follow its longest direction. " +
      "Then press <b>Project onto PC1</b> to collapse it to one dimension.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" },
            "Solid cyan arrow = PC1 (most variance), violet arrow = PC2. ✛ marks the mean. Click empty space to add your own points.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Variance explained"),
            el("div", { class: "bars" },
              el("div", { class: "bar-row" },
                el("span", { class: "bl" }, "PC1"),
                el("div", { class: "bar-track" }, bar1),
                bar1Val),
              el("div", { class: "bar-row" },
                el("span", { class: "bl" }, "PC2"),
                el("div", { class: "bar-track" }, bar2),
                bar2Val))),
          el("div", { class: "panel" },
            el("h3", {}, "Principal components"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "λ₁ (var along PC1)"), l1El),
            el("div", { class: "kv" }, el("span", { class: "k" }, "λ₂ (var along PC2)"), l2El),
            el("div", { class: "kv" }, el("span", { class: "k" }, "PC1 angle"), angEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Set rotation / spread"), ratioEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Points"), nEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, regenBtn, projBtn, clearBtn),
            el("div", { class: "controls" }, elongSlider.wrap),
            el("div", { class: "controls" }, angleSlider.wrap)),
          el("div", { class: "legend" },
            el("span", { class: "legend-item" },
              el("span", { class: "legend-line", style: { background: "#5ad2ff" } }), "PC1"),
            el("span", { class: "legend-item" },
              el("span", { class: "legend-line", style: { background: "#b388ff" } }), "PC2"),
            el("span", { class: "legend-item" },
              el("span", { class: "legend-dot", style: { background: "#7c89b8" } }), "data point")))),
      verdict
    );

    // ---- data generation ----
    // Sample (a·N(0,1), b·N(0,1)) then rotate by `angle`. a>b => elongated.
    function regen() {
      const b = 0.34;            // base std dev of the short axis
      const a = b * elong;       // long-axis std dev (a ≥ b for elong ≥ 1)
      const ca = Math.cos(angle), sa = Math.sin(angle);
      points = [];
      for (let i = 0; i < N; i++) {
        const u = gaussRandom(0, a);   // along the long axis (pre-rotation x)
        const v = gaussRandom(0, b);   // along the short axis (pre-rotation y)
        // rotate the (u,v) axes by `angle`
        const x = ca * u - sa * v;
        const y = sa * u + ca * v;
        points.push({ x: clamp(x, -1.18, 1.18), y: clamp(y, -1.18, 1.18) });
      }
    }

    // ---- the actual PCA: closed-form 2x2 eigen-decomposition ----
    function computePCA() {
      const n = points.length;
      if (n < 2) { pca = null; return; }

      // 1. mean
      let mx = 0, my = 0;
      for (const p of points) { mx += p.x; my += p.y; }
      mx /= n; my /= n;

      // 2. covariance matrix C = [[sxx, sxy],[sxy, syy]]  (population, /n)
      let sxx = 0, syy = 0, sxy = 0;
      for (const p of points) {
        const dx = p.x - mx, dy = p.y - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
      }
      sxx /= n; syy /= n; sxy /= n;

      // 3. eigenvalues of a symmetric 2x2: λ = (tr ± √(tr²−4·det)) / 2
      const tr = sxx + syy;
      const det = sxx * syy - sxy * sxy;
      const disc = Math.max(0, tr * tr - 4 * det);   // ≥0 for symmetric matrices
      const root = Math.sqrt(disc);
      const l1 = (tr + root) / 2;   // larger eigenvalue = variance along PC1
      const l2 = (tr - root) / 2;   // smaller = variance along PC2

      // 4. eigenvector for λ1. For C·v = λ·v, a valid direction is (sxy, λ−sxx).
      //    If sxy ≈ 0 the matrix is already diagonal: axes are x and y, and the
      //    larger-variance axis owns PC1.
      let v1x, v1y;
      if (Math.abs(sxy) < 1e-9) {
        if (sxx >= syy) { v1x = 1; v1y = 0; }
        else { v1x = 0; v1y = 1; }
      } else {
        v1x = sxy; v1y = l1 - sxx;
      }
      const norm = Math.hypot(v1x, v1y) || 1;
      v1x /= norm; v1y /= norm;
      // PC2 is orthogonal to PC1 (eigenvectors of a symmetric matrix are ⟂)
      const v2x = -v1y, v2y = v1x;

      const total = l1 + l2;
      const explained = total > 0 ? l1 / total : 0.5;

      pca = { mx, my, sxx, syy, sxy, l1, l2,
        v1: [v1x, v1y], v2: [v2x, v2y], explained };
    }

    // ---- drawing ----
    const VIS = 2.4; // arrows are √λ scaled up so a std-dev is clearly visible

    function draw() {
      plot.clear("#070d1a");
      plot.grid(0.4);
      plot.axes();

      if (!pca) {
        for (const p of points) plot.point(p.x, p.y, 4, "#7c89b8", "rgba(7,13,26,0.8)", 1);
        return;
      }
      const { mx, my, l1, l2, v1, v2 } = pca;

      if (projecting) {
        // PC1 line through the mean, extended across the window
        const ext = 2.6;
        plot.line(mx - v1[0] * ext, my - v1[1] * ext, mx + v1[0] * ext, my + v1[1] * ext,
          withAlpha("#5ad2ff", 0.55), 2, [6, 5]);

        // for each point: drop a perpendicular to its projection on the PC1 line
        for (const p of points) {
          const dx = p.x - mx, dy = p.y - my;
          const t = dx * v1[0] + dy * v1[1];   // scalar coordinate along PC1
          const projX = mx + t * v1[0];
          const projY = my + t * v1[1];
          // original point (dimmed)
          plot.point(p.x, p.y, 3.5, withAlpha("#7c89b8", 0.45), null);
          // faint perpendicular residual segment
          plot.line(p.x, p.y, projX, projY, withAlpha("#ffd166", 0.28), 1);
          // the projected (reduced) point, sitting ON the line
          plot.point(projX, projY, 3.6, "#5ad2ff", "rgba(7,13,26,0.8)", 1);
        }
      } else {
        for (const p of points) plot.point(p.x, p.y, 4, "#7c89b8", "rgba(7,13,26,0.85)", 1);
      }

      // principal-axis arrows from the mean, length ∝ √λ (a standard deviation)
      drawArrow(mx, my, v1, Math.sqrt(l1) * VIS, "#5ad2ff", 3);
      drawArrow(mx, my, v2, Math.sqrt(l2) * VIS, "#b388ff", 2.5);

      // mean marker (✛)
      const cx = plot.sx(mx), cy = plot.sy(my);
      const c = plot.ctx;
      c.save();
      c.strokeStyle = "#fff";
      c.lineWidth = 2;
      c.beginPath();
      c.moveTo(cx - 7, cy); c.lineTo(cx + 7, cy);
      c.moveTo(cx, cy - 7); c.lineTo(cx, cy + 7);
      c.stroke();
      c.restore();
    }

    // arrow in data space from (mx,my) along unit dir `v` for data-length `len`
    function drawArrow(mx, my, v, len, color, width) {
      const tipX = mx + v[0] * len, tipY = my + v[1] * len;
      const c = plot.ctx;
      const x0 = plot.sx(mx), y0 = plot.sy(my);
      const x1 = plot.sx(tipX), y1 = plot.sy(tipY);
      c.save();
      c.strokeStyle = color;
      c.fillStyle = color;
      c.lineWidth = width;
      c.beginPath();
      c.moveTo(x0, y0); c.lineTo(x1, y1);
      c.stroke();
      // arrowhead (in pixel space)
      const ang = Math.atan2(y1 - y0, x1 - x0);
      const h = 9;
      c.beginPath();
      c.moveTo(x1, y1);
      c.lineTo(x1 - h * Math.cos(ang - 0.4), y1 - h * Math.sin(ang - 0.4));
      c.lineTo(x1 - h * Math.cos(ang + 0.4), y1 - h * Math.sin(ang + 0.4));
      c.closePath();
      c.fill();
      c.restore();
    }

    // ---- refresh stats + verdict ----
    function refresh() {
      computePCA();
      draw();

      nEl.textContent = points.length;

      if (!pca) {
        bar1.style.width = "0%"; bar2.style.width = "0%";
        bar1Val.textContent = "—"; bar2Val.textContent = "—";
        l1El.textContent = "—"; l2El.textContent = "—";
        angEl.textContent = "—"; ratioEl.textContent = "—";
        verdict.innerHTML = "Add at least two points (or press <b>Regenerate</b>) so there is a cloud to analyse.";
        return;
      }

      const { l1, l2, v1, explained } = pca;
      const p1 = explained * 100, p2 = (1 - explained) * 100;
      bar1.style.width = p1.toFixed(1) + "%";
      bar2.style.width = p2.toFixed(1) + "%";
      bar1Val.textContent = p1.toFixed(0) + "%";
      bar2Val.textContent = p2.toFixed(0) + "%";

      l1El.textContent = l1.toFixed(4);
      l2El.textContent = l2.toFixed(4);
      const pcAngle = Math.atan2(v1[1], v1[0]);
      angEl.textContent = pcAngle.toFixed(2) + " rad";
      ratioEl.textContent = `${angle.toFixed(2)} rad · ${elong.toFixed(1)}×`;

      if (projecting) {
        const lost = p2.toFixed(0);
        verdict.innerHTML =
          `Each point is now described by <b>one number</b> — its position along PC1. ` +
          `You kept <b>${p1.toFixed(0)}%</b> of the variance and threw away the other <b>${lost}%</b> ` +
          `(the gold residuals are exactly what was lost).`;
      } else if (p1 >= 85) {
        verdict.innerHTML =
          `PC1 captures <b>${p1.toFixed(0)}%</b> of the variance — you could describe each point with <b>one number</b> ` +
          `and lose almost nothing. That is dimensionality reduction.`;
      } else if (p1 >= 60) {
        verdict.innerHTML =
          `PC1 holds <b>${p1.toFixed(0)}%</b> of the variance, PC2 the remaining <b>${p2.toFixed(0)}%</b>. ` +
          `Dropping PC2 loses some real structure — the cloud isn't very elongated.`;
      } else {
        verdict.innerHTML =
          `The two components are nearly balanced (<b>${p1.toFixed(0)}%</b> / <b>${p2.toFixed(0)}%</b>) — ` +
          `the cloud is almost round, so <b>no single direction dominates</b> and reduction would hurt.`;
      }
    }

    // ---- events ----
    regenBtn.addEventListener("click", () => { regen(); refresh(); });
    clearBtn.addEventListener("click", () => { points = []; refresh(); });
    projBtn.addEventListener("click", () => {
      projecting = !projecting;
      projBtn.classList.toggle("active", projecting);
      projBtn.textContent = projecting ? "Show 2-D cloud" : "Project onto PC1";
      refresh();
    });
    elongSlider.input.addEventListener("input", () => {
      elong = parseFloat(elongSlider.input.value);
      regen(); refresh();
    });
    angleSlider.input.addEventListener("input", () => {
      angle = parseFloat(angleSlider.input.value);
      regen(); refresh();
    });

    canvas.addEventListener("pointerdown", (e) => {
      const { x, y } = plot.eventToData(e);
      if (x < plot.xmin || x > plot.xmax || y < plot.ymin || y > plot.ymax) return;
      points.push({ x, y });
      refresh();
    });

    // ---- init ----
    regen();
    refresh();

    return {
      onEnter() { draw(); },
      onLeave() {
        // No Loop runs (the visual is static), so nothing to stop. Reset the
        // projection toggle + its label so the view re-opens in its default state.
        if (projecting) {
          projecting = false;
          projBtn.classList.remove("active");
          projBtn.textContent = "Project onto PC1";
          refresh();
        }
      },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>📐 The job: fewer numbers, same story</h3>
    <p>High-dimensional data is hard to store, plot and reason about — yet much of it is redundant. Pixels in a face photo move together; a customer's "spend" and "visits" rise together. <b>PCA</b> (Principal Component Analysis) finds the directions your data actually varies along and lets you keep only the strong ones. It is unsupervised: no labels, just the shape of the cloud. Used for compression, denoising, and squashing 50-D data down to a 2-D picture you can look at.</p>
  </div>
  <div class="lesson-block">
    <h3>👁️ What you can see above</h3>
    <ol class="steps">
      <li><b>Centre.</b> PCA first shifts the cloud so its mean (the ✛) sits at the origin — variance is measured <em>around the mean</em>.</li>
      <li><b>Find the grain.</b> The cyan arrow (<b>PC1</b>) points along the direction of greatest spread; the violet arrow (<b>PC2</b>) is exactly perpendicular and catches what's left. Each arrow's length is one standard deviation along that axis.</li>
      <li><b>Rank.</b> The bars show how much of the total variance each component holds. Rotate or stretch the cloud and watch the arrows track its longest direction.</li>
      <li><b>Reduce.</b> Hit "Project onto PC1": every point drops perpendicularly onto the PC1 line (the gold stubs are the discarded part). Two numbers became one.</li>
    </ol>
  </div>
  <div class="lesson-block">
    <h3>🧮 The maths: eigenvectors of the covariance matrix</h3>
    <p>"Variance" measures spread, and spread <em>is</em> information. After centring, summarise how the coordinates co-vary with the <b>covariance matrix</b>:</p>
    <div class="formula">C = [ s<sub>xx</sub>&nbsp;&nbsp;s<sub>xy</sub> ; &nbsp;s<sub>xy</sub>&nbsp;&nbsp;s<sub>yy</sub> ]&nbsp;&nbsp; where&nbsp; s<sub>xx</sub>=mean(dx²),&nbsp; s<sub>yy</sub>=mean(dy²),&nbsp; s<sub>xy</sub>=mean(dx·dy)</div>
    <p>The principal directions are the <b>eigenvectors</b> of C, and their <b>eigenvalues λ are the variance along them</b> — that is the whole idea: <code>C·v = λ·v</code>. For a 2×2 there's a closed form (no iteration needed):</p>
    <div class="formula">λ = ( tr ± √(tr² − 4·det) ) / 2,&nbsp;&nbsp; tr = s<sub>xx</sub>+s<sub>yy</sub>,&nbsp; det = s<sub>xx</sub>s<sub>yy</sub> − s<sub>xy</sub>²</div>
    <p>The bigger root is λ₁ (PC1); its eigenvector direction is <code>(s<sub>xy</sub>, λ₁ − s<sub>xx</sub>)</code>, normalised. Rank components by λ, keep the top ones, and report <b>variance explained</b> = λ₁ / (λ₁+λ₂). Reduction is just: <em>rotate onto the PC axes, then drop the low-variance ones</em> — exactly the projection you see.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ Where it bites</h3>
    <ul class="bullets">
      <li><b>Linear only.</b> PCA can only find <em>straight</em> directions. A cloud curled into a spiral or an S-shape fools it — set the cloud nearly round above and watch the bars even out: no direction wins, so reduction would throw away real structure.</li>
      <li><b>Scale-sensitive.</b> Variance depends on units. A feature measured in millimetres will dominate one in metres unless you standardise (z-score) first.</li>
      <li><b>Variance ≠ usefulness.</b> The most-spread direction isn't always the most <em>meaningful</em> one for your task; PCA is blind to labels.</li>
      <li><b>Outliers</b> inflate variance and can yank an axis toward themselves — try clicking one far-off point and watch PC1 swing.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> PCA rotates your data onto the orthogonal directions of maximum variance — the eigenvectors of the covariance matrix — and you keep the few with the biggest eigenvalues, compressing the data while losing as little spread as possible.</p>
  </div>`,
};
