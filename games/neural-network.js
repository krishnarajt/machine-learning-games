// =====================================================================
//  Neural Network — the showpiece.
//
//  A from-scratch multi-layer perceptron (MLP) trained by
//  backpropagation + gradient descent on a non-linear 2-class problem.
//
//  Architecture: 2 inputs -> (one or two) tanh hidden layers -> 1 sigmoid.
//  Forward:  z[l] = W[l]·a[l-1] + b[l];  a[l] = tanh(z) (hidden), sigmoid (out)
//  Loss:     binary cross-entropy
//  Backward: out delta = a_out - y
//            hidden delta[l] = (W[l+1]^T · delta[l+1]) ⊙ (1 - a[l]^2)
//            gradW[l] = delta[l] · a[l-1]^T ;  gradb[l] = delta[l]
//  Update:   W -= lr·gradW ;  b -= lr·gradb   (full batch, many epochs/frame)
//
//  The decision boundary is shaded behind the data and visibly bends as
//  the net trains. A small side canvas draws the live weight diagram.
// =====================================================================
import {
  el, button, ghostButton, slider, Loop, Plot,
  circles, xor, spiral, twoMoons,
  gaussRandom, sigmoid, clamp, mixHex, withAlpha, classColor,
} from "../shared.js";

export default {
  key: "neural-network",
  name: "Neural Network",
  icon: "🧠",
  category: "Classification",
  tag: "Deep",
  blurb: "Train a tiny multi-layer network to learn curved, tangled boundaries by backprop.",
  eyebrow: "Classification · Supervised",
  title: 'Neural Network: <span class="grad">bending</span> the boundary',
  lede:
    "Stack simple neurons in layers, add a dash of non-linearity, and a network can carve boundaries no straight line " +
    "could — circles, spirals, XOR. Watch it learn one from scratch by <em>backpropagation</em>: making a guess, " +
    "measuring the error, and pushing that error back through every weight.",

  mount(root) {
    // ---- palette ----
    const C0 = classColor(0); // cyan  -> class 0
    const C1 = classColor(1); // coral -> class 1

    // ---- state ----
    let dataKey = "circles";
    let data = [];            // { x, y, label } with label in {0,1}
    let lr = 0.15;
    let width = 8;            // hidden layer width
    let depth = 2;            // number of hidden layers (1 or 2)
    let arch = [];            // full layer sizes, e.g. [2,8,8,1]
    let W = [];               // W[l]: rows=out, cols=in  (l = 1..L)
    let b = [];               // b[l]: length = out
    let epoch = 0;
    let loss = 0;
    let acc = 0;
    const EPOCHS_PER_FRAME = 12;

    // ---- DOM: main plot ----
    const canvas = el("canvas", { width: 760, height: 460, role: "img",
      "aria-label": "Decision boundary shaded behind two classes of points; it bends as the network trains." });
    const plot = new Plot(canvas, { xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2, pad: 14 });

    // ---- DOM: network diagram ----
    const netCanvas = el("canvas", { width: 320, height: 220, role: "img",
      "aria-label": "Diagram of the network's layers and weighted connections." });
    const netCtx = netCanvas.getContext("2d");

    // ---- DOM: dataset segmented control ----
    const segSpecs = [
      ["circles", "Circles"],
      ["xor", "XOR"],
      ["spiral", "Spiral"],
      ["moons", "Moons"],
    ];
    const segBtns = segSpecs.map(([k, label]) =>
      el("button", { class: k === dataKey ? "active" : "", onClick: () => setData(k) }, label));
    const segGroup = el("div", { class: "seg-group" }, ...segBtns);

    // ---- DOM: width segmented control ----
    const widthSpecs = [4, 8, 16];
    const widthBtns = widthSpecs.map((w) =>
      el("button", { class: w === width ? "active" : "", onClick: () => setWidth(w) }, String(w)));
    const widthGroup = el("div", { class: "seg-group" }, ...widthBtns);

    // ---- DOM: depth segmented control ----
    const depthSpecs = [1, 2];
    const depthBtns = depthSpecs.map((d) =>
      el("button", { class: d === depth ? "active" : "", onClick: () => setDepth(d) },
        d === 1 ? "1 layer" : "2 layers"));
    const depthGroup = el("div", { class: "seg-group" }, ...depthBtns);

    // ---- DOM: controls ----
    const trainBtn = button("Train ▸");
    const resetBtn = ghostButton("Reset weights");
    const newDataBtn = ghostButton("New data");
    const lrSlider = slider({ min: 0.01, max: 0.6, step: 0.01, value: lr, label: "Learning rate η =", format: (v) => v.toFixed(2) });

    // ---- DOM: stat cards ----
    const epochNum = el("div", { class: "stat-number" }, "0");
    const lossNum = el("div", { class: "stat-number" }, "—");
    const accNum = el("div", { class: "stat-number" }, "—");
    const stats = el("div", { class: "stats" },
      el("div", { class: "stat-card" }, epochNum, el("div", { class: "stat-label" }, "epoch")),
      el("div", { class: "stat-card" }, lossNum, el("div", { class: "stat-label" }, "loss (BCE)")),
      el("div", { class: "stat-card warm" }, accNum, el("div", { class: "stat-label" }, "accuracy")));

    const archEl = el("span", { class: "v" }, "");
    const verdict = el("p", { class: "verdict" },
      "Press <b>Train</b> and watch a straight-line classifier fail while this network bends the boundary to fit. ");

    // ---- layout ----
    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" },
            "Background shade = the network's predicted class probability. Dots = the true labels. " +
            "The boundary is the cyan/coral seam.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Training"),
            stats,
            el("div", { class: "kv" }, el("span", { class: "k" }, "Architecture"), archEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Network"),
            el("div", { class: "canvas-stage pad" }, netCanvas),
            el("p", { class: "caption" },
              "Edge colour = weight sign (cyan +, coral −), thickness/opacity = |weight|.")),
          el("div", { class: "panel" },
            el("h3", {}, "Dataset"),
            el("div", { class: "controls" }, segGroup)),
          el("div", { class: "panel" },
            el("h3", {}, "Model"),
            el("div", { class: "controls" }, el("span", { class: "k" }, "Hidden width"), widthGroup),
            el("div", { class: "controls" }, el("span", { class: "k" }, "Hidden depth"), depthGroup),
            el("div", { class: "controls" }, lrSlider.wrap)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, trainBtn, resetBtn, newDataBtn)),
          el("div", { class: "legend" },
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { background: C0 } }), "Class 0"),
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot", style: { background: C1 } }), "Class 1")))),
      verdict
    );

    // =================================================================
    //  Data
    // =================================================================
    function makeData(k) {
      if (k === "circles") return circles(180, 0.06);
      if (k === "xor") return xor(200, 0.12);
      if (k === "spiral") return spiral(110, 0.04);
      return twoMoons(180, 0.08);
    }

    // =================================================================
    //  Network: generic layered MLP
    // =================================================================
    function buildArch() {
      arch = [2];
      for (let i = 0; i < depth; i++) arch.push(width);
      arch.push(1);
    }

    function initWeights() {
      buildArch();
      W = [null]; // index 0 unused (input layer has no incoming weights)
      b = [null];
      for (let l = 1; l < arch.length; l++) {
        const nIn = arch[l - 1], nOut = arch[l];
        const sd = Math.sqrt(1 / nIn); // He/Xavier-ish: gaussRandom * sqrt(1/fanIn)
        const Wl = [];
        for (let o = 0; o < nOut; o++) {
          const row = new Array(nIn);
          for (let i = 0; i < nIn; i++) row[i] = gaussRandom(0, 1) * sd;
          Wl.push(row);
        }
        W.push(Wl);
        b.push(new Array(nOut).fill(0));
      }
      epoch = 0; loss = 0; acc = 0;
    }

    // forward pass for a single input [x,y]; returns array of activations a[l]
    function forward(input) {
      const a = [input.slice()];
      for (let l = 1; l < arch.length; l++) {
        const Wl = W[l], bl = b[l], prev = a[l - 1];
        const isOut = l === arch.length - 1;
        const out = new Array(Wl.length);
        for (let o = 0; o < Wl.length; o++) {
          const row = Wl[o];
          let z = bl[o];
          for (let i = 0; i < row.length; i++) z += row[i] * prev[i];
          out[o] = isOut ? sigmoid(z) : Math.tanh(z);
        }
        a.push(out);
      }
      return a;
    }

    // quick predict — just the output probability
    function predict(x, y) {
      let prev = [x, y];
      for (let l = 1; l < arch.length; l++) {
        const Wl = W[l], bl = b[l];
        const isOut = l === arch.length - 1;
        const out = new Array(Wl.length);
        for (let o = 0; o < Wl.length; o++) {
          const row = Wl[o];
          let z = bl[o];
          for (let i = 0; i < row.length; i++) z += row[i] * prev[i];
          out[o] = isOut ? sigmoid(z) : Math.tanh(z);
        }
        prev = out;
      }
      return prev[0];
    }

    // one full-batch gradient-descent epoch; returns mean BCE loss
    function trainEpoch() {
      const n = data.length;
      if (!n) { epoch++; return 0; }   // guard: empty dataset -> no update, finite loss
      const L = arch.length - 1;
      // gradient accumulators, same shape as W and b
      const gW = [null], gb = [null];
      for (let l = 1; l <= L; l++) {
        gW.push(W[l].map((row) => new Array(row.length).fill(0)));
        gb.push(new Array(b[l].length).fill(0));
      }

      let totalLoss = 0;
      for (const p of data) {
        const yTrue = p.label;
        const a = forward([p.x, p.y]);
        const aOut = a[L][0];
        // binary cross-entropy (clamped for log safety)
        const yhat = clamp(aOut, 1e-7, 1 - 1e-7);
        totalLoss += -(yTrue * Math.log(yhat) + (1 - yTrue) * Math.log(1 - yhat));

        // ---- backprop ----
        // delta for each layer; delta[l] has length arch[l]
        const delta = new Array(arch.length).fill(null);
        // output layer: sigmoid + BCE => dL/dz = a_out - y
        delta[L] = [aOut - yTrue];
        // hidden layers, going backward
        for (let l = L - 1; l >= 1; l--) {
          const nextW = W[l + 1];      // shape: arch[l+1] x arch[l]
          const nextDelta = delta[l + 1];
          const al = a[l];             // tanh activations of this layer
          const dl = new Array(arch[l]);
          for (let i = 0; i < arch[l]; i++) {
            let s = 0;
            for (let o = 0; o < nextW.length; o++) s += nextW[o][i] * nextDelta[o];
            dl[i] = s * (1 - al[i] * al[i]); // tanh' = 1 - a^2
          }
          delta[l] = dl;
        }
        // accumulate gradients: gradW[l] = delta[l] · a[l-1]^T, gradb[l]=delta[l]
        for (let l = 1; l <= L; l++) {
          const dl = delta[l], prev = a[l - 1];
          const gWl = gW[l], gbl = gb[l];
          for (let o = 0; o < dl.length; o++) {
            const d = dl[o];
            const gRow = gWl[o];
            for (let i = 0; i < prev.length; i++) gRow[i] += d * prev[i];
            gbl[o] += d;
          }
        }
      }

      // gradient-descent update (mean gradient over the batch)
      const scale = lr / n;
      for (let l = 1; l <= L; l++) {
        const Wl = W[l], bl = b[l], gWl = gW[l], gbl = gb[l];
        for (let o = 0; o < Wl.length; o++) {
          const row = Wl[o], gRow = gWl[o];
          for (let i = 0; i < row.length; i++) row[i] -= scale * gRow[i];
          bl[o] -= scale * gbl[o];
        }
      }

      epoch++;
      return totalLoss / n;
    }

    function computeAccuracy() {
      let correct = 0;
      for (const p of data) {
        const pred = predict(p.x, p.y) >= 0.5 ? 1 : 0;
        if (pred === p.label) correct++;
      }
      return data.length ? correct / data.length : 0;
    }

    // =================================================================
    //  Drawing — main decision boundary
    // =================================================================
    function draw() {
      plot.clear("#070d1a");
      // decision-boundary backdrop: blend cyan<->coral by sigmoid output
      plot.shade((x, y) => {
        const p = predict(x, y);          // probability of class 1
        const col = mixHex(C0, C1, clamp(p, 0, 1));
        return withAlpha(col, 0.5);
      }, 56);
      plot.grid(0.6, "rgba(255,255,255,0.05)");

      // data points by TRUE label
      for (const p of data) {
        plot.point(p.x, p.y, 4.5, p.label === 1 ? C1 : C0, "rgba(7,13,26,0.85)", 1.3);
      }
      drawNet();
    }

    // =================================================================
    //  Drawing — network diagram
    // =================================================================
    function drawNet() {
      const w = netCanvas.width, h = netCanvas.height;
      netCtx.clearRect(0, 0, w, h);
      netCtx.fillStyle = "#070d1a";
      netCtx.fillRect(0, 0, w, h);

      const padX = 26, padY = 22;
      const nLayers = arch.length;
      const colX = (l) => padX + (l / (nLayers - 1)) * (w - 2 * padX);
      const nodeY = (l, i) => {
        const cnt = arch[l];
        if (cnt === 1) return h / 2;
        return padY + (i / (cnt - 1)) * (h - 2 * padY);
      };

      // edges first (colour = sign, opacity/width = magnitude)
      for (let l = 1; l < nLayers; l++) {
        const Wl = W[l];
        // normalise by the largest |weight| in this layer for stable visuals
        let maxAbs = 1e-6;
        for (const row of Wl) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
        for (let o = 0; o < Wl.length; o++) {
          const row = Wl[o];
          const x2 = colX(l), y2 = nodeY(l, o);
          for (let i = 0; i < row.length; i++) {
            const wv = row[i];
            const mag = Math.abs(wv) / maxAbs;        // 0..1
            const x1 = colX(l - 1), y1 = nodeY(l - 1, i);
            netCtx.strokeStyle = withAlpha(wv >= 0 ? C0 : C1, 0.15 + 0.7 * mag);
            netCtx.lineWidth = 0.4 + 2.6 * mag;
            netCtx.beginPath();
            netCtx.moveTo(x1, y1);
            netCtx.lineTo(x2, y2);
            netCtx.stroke();
          }
        }
      }
      // nodes
      for (let l = 0; l < nLayers; l++) {
        for (let i = 0; i < arch[l]; i++) {
          const x = colX(l), y = nodeY(l, i);
          netCtx.beginPath();
          netCtx.arc(x, y, l === 0 || l === nLayers - 1 ? 6 : 5, 0, Math.PI * 2);
          netCtx.fillStyle = l === nLayers - 1 ? "#ffd166" : "#0e1830";
          netCtx.strokeStyle = "rgba(234,238,251,0.55)";
          netCtx.lineWidth = 1.4;
          netCtx.fill();
          netCtx.stroke();
        }
      }
      // layer labels
      netCtx.fillStyle = "rgba(234,238,251,0.5)";
      netCtx.font = "10px 'JetBrains Mono', monospace";
      netCtx.textAlign = "center";
      netCtx.fillText("in", colX(0), h - 6);
      netCtx.fillText("out", colX(nLayers - 1), h - 6);
      netCtx.textAlign = "left";
    }

    // =================================================================
    //  Stats / verdict
    // =================================================================
    function refresh() {
      epochNum.textContent = epoch;
      lossNum.textContent = epoch ? loss.toFixed(3) : "—";
      accNum.textContent = epoch ? Math.round(acc * 100) + "%" : "—";
      archEl.textContent = "[ " + arch.join(", ") + " ]";

      if (!epoch) {
        verdict.innerHTML = "Press <b>Train</b> and watch a straight-line classifier fail while this network bends " +
          "the boundary to fit. " + datasetHint();
      } else if (acc >= 0.97) {
        verdict.innerHTML = `Solved it: <b>${Math.round(acc * 100)}%</b> accuracy after <b>${epoch}</b> epochs. ` +
          `The boundary curves to wrap each class — something a single straight line could never do.`;
      } else if (acc >= 0.85) {
        verdict.innerHTML = `Epoch <b>${epoch}</b>: loss <b>${loss.toFixed(3)}</b>, accuracy <b>${Math.round(acc * 100)}%</b> ` +
          `and still tightening. The seam is bending into shape — keep training.`;
      } else {
        verdict.innerHTML = `Epoch <b>${epoch}</b>: loss <b>${loss.toFixed(3)}</b>, accuracy <b>${Math.round(acc * 100)}%</b>. ` +
          `Early days — the gradients are still pushing the weights around. ` + datasetHint();
      }
    }

    function datasetHint() {
      if (dataKey === "spiral") return "Spirals are the hardest here — they need a wider/deeper net and patience.";
      if (dataKey === "xor") return "XOR is the classic proof that one straight line cannot separate these classes.";
      return "";
    }

    // =================================================================
    //  Loop
    // =================================================================
    const loop = new Loop(() => {
      for (let i = 0; i < EPOCHS_PER_FRAME; i++) loss = trainEpoch();
      acc = computeAccuracy();
      draw();
      refresh();
    });

    function setTrainLabel() { trainBtn.textContent = loop.running ? "Pause ⏸" : "Train ▸"; }

    // =================================================================
    //  Setters / events
    // =================================================================
    function setData(k) {
      dataKey = k;
      segBtns.forEach((btn, i) => btn.classList.toggle("active", segSpecs[i][0] === k));
      data = makeData(k);
      hardReset();
    }
    function setWidth(w) {
      width = w;
      widthBtns.forEach((btn, i) => btn.classList.toggle("active", widthSpecs[i] === w));
      hardReset();
    }
    function setDepth(d) {
      depth = d;
      depthBtns.forEach((btn, i) => btn.classList.toggle("active", depthSpecs[i] === d));
      hardReset();
    }
    function hardReset() {
      loop.stop(); setTrainLabel();
      initWeights();
      acc = computeAccuracy();
      draw();
      refresh();
    }

    trainBtn.addEventListener("click", () => { loop.toggle(); setTrainLabel(); });
    resetBtn.addEventListener("click", () => { hardReset(); });
    newDataBtn.addEventListener("click", () => { data = makeData(dataKey); hardReset(); });
    lrSlider.input.addEventListener("input", () => { lr = parseFloat(lrSlider.input.value); });

    // ---- init ----
    data = makeData(dataKey);
    initWeights();
    acc = computeAccuracy();
    draw();
    refresh();

    return {
      onEnter() { draw(); refresh(); },
      onLeave() { loop.stop(); setTrainLabel(); },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🧩 The job: boundaries a straight line can't draw</h3>
    <p>Some classes can't be split by a straight line. The two rings of <b>Circles</b>, the diagonal quadrants of <b>XOR</b>, the interlocking <b>Spirals</b> — a single line (a logistic-regression boundary) will always get a big chunk wrong. A neural network learns a <em>curved</em> boundary by stacking many simple units, and it discovers that curve entirely from the labelled data.</p>
  </div>
  <div class="lesson-block">
    <h3>⚙️ A neuron = weighted sum + a squish</h3>
    <p>One neuron takes its inputs, multiplies each by a weight, adds a bias, then passes the total through a non-linear <b>activation</b>:</p>
    <div class="formula">z = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + b &nbsp;&nbsp;&nbsp; a = tanh(z)</div>
    <p>The output neuron uses a <b>sigmoid</b> instead, squashing its total into a 0–1 probability of class 1.</p>
  </div>
  <div class="lesson-block">
    <h3>🔑 Why the non-linearity is everything</h3>
    <p>Here's the trick most people miss. If every layer were purely linear (no tanh), the whole stack would collapse: a linear function of a linear function is still just <em>one</em> linear function. You'd be right back to a single straight line — exactly the <b>Logistic Regression</b> game, no matter how many layers you piled on. The <code>tanh</code> bend in each hidden unit is what lets layers <em>compose</em> simple bends into genuinely curved, multi-region boundaries.</p>
  </div>
  <div class="lesson-block">
    <h3>➡️ Forward pass: making a guess</h3>
    <p>Push an input through the layers. With activation vectors a, weight matrices W and bias vectors b:</p>
    <div class="formula">a<sup>(0)</sup> = [x, y]<br>z<sup>(l)</sup> = W<sup>(l)</sup>·a<sup>(l−1)</sup> + b<sup>(l)</sup><br>a<sup>(l)</sup> = tanh(z<sup>(l)</sup>)&nbsp;&nbsp;(hidden), &nbsp; sigmoid(z<sup>(l)</sup>)&nbsp;(output)</div>
    <p>The final number is the predicted probability that the point is class 1. We score the guess with <b>binary cross-entropy</b>:</p>
    <div class="formula">L = −[ y·log(ŷ) + (1−y)·log(1−ŷ) ]</div>
  </div>
  <div class="lesson-block">
    <h3>⬅️ Backpropagation: pushing the error back</h3>
    <p>Backprop is just the <b>chain rule</b> applied layer by layer, from the output toward the input. It answers: "how much did each weight contribute to the error?" Define a per-neuron error signal <code>δ</code>. For sigmoid-output + cross-entropy the output one is beautifully simple, and each hidden layer pulls its error from the layer in front of it:</p>
    <div class="formula">δ<sup>(out)</sup> = a<sub>out</sub> − y<br>δ<sup>(l)</sup> = (W<sup>(l+1)T</sup>·δ<sup>(l+1)</sup>) ⊙ (1 − a<sup>(l)2</sup>)</div>
    <p>That last term, <code>1 − a²</code>, is the derivative of tanh. Once every layer has its <code>δ</code>, the gradients are tiny products:</p>
    <div class="formula">∂L/∂W<sup>(l)</sup> = δ<sup>(l)</sup>·a<sup>(l−1)T</sup> &nbsp;&nbsp;&nbsp; ∂L/∂b<sup>(l)</sup> = δ<sup>(l)</sup></div>
  </div>
  <div class="lesson-block">
    <h3>📉 Gradient descent: the actual learning</h3>
    <p>Now nudge every weight a small step <em>downhill</em> on the loss — exactly the rule from the <b>Gradient Descent</b> game, just applied to thousands of weights at once:</p>
    <div class="formula">W ← W − η · ∂L/∂W &nbsp;&nbsp;&nbsp; b ← b − η · ∂L/∂b</div>
    <p>This game runs many epochs per animation frame, so you can watch the cyan/coral seam bend and tighten in real time. The <b>η</b> (learning-rate) slider sets the step size: too small and it crawls, too large and the loss thrashes or diverges.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ Failure modes</h3>
    <ul class="bullets">
      <li><b>Too little capacity.</b> A 4-wide, 1-layer net often can't fit the spiral. Add width/depth to give it more bends to work with.</li>
      <li><b>Too much capacity.</b> A huge net can memorise noise — overfitting. With real data you'd hold out a validation set; here the boundary just gets wigglier than the true shape.</li>
      <li><b>Bad learning rate.</b> Crank η up and watch the loss bounce; drop it too low and training stalls.</li>
      <li><b>Random restarts matter.</b> Reset re-rolls the weights — sometimes it lands in a better basin than others. Same lesson as k-Means: gradient methods find <em>local</em> optima.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> a neural network stacks weighted-sum-plus-squish neurons so layers can compose simple bends into complex curved boundaries; backpropagation is the chain rule sending error gradients backward, and gradient descent uses them to learn the weights — the foundation of all deep learning.</p>
  </div>`,
};
