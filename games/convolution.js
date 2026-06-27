// =====================================================================
//  Convolution — the building block of convolutional neural networks.
//
//  An image is just a grid of numbers (intensities 0..255). Convolution
//  slides a tiny 3x3 grid of weights — a KERNEL — across the image, and
//  at every position multiplies-and-sums the overlap to produce one
//  output pixel. Stack those outputs and you get a "feature map".
//
//  This game generates a synthetic grayscale image (offline, no assets),
//  lets you pick or hand-edit a kernel, and shows the input + output
//  side by side. An "Animate scan" toggle sweeps the 3x3 window across
//  the image so you can watch the slide-multiply-sum happen.
// =====================================================================
import { el, ghostButton, Loop, clamp, gaussRandom } from "../shared.js";

// ---- logical image size (small so convolution is cheap) ----
const N = 80;          // image is N x N
const DISP = 360;      // display canvas size in pixels

// ---- kernel presets (3x3, row-major 9-arrays) ----
// Each preset carries its own divisor + offset + a plain-English description.
const PRESETS = {
  identity: {
    label: "Identity",
    k: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    div: 1, offset: 0,
    detects: "nothing — passes the image straight through, unchanged.",
  },
  box: {
    label: "Box blur",
    k: [1, 1, 1, 1, 1, 1, 1, 1, 1],
    div: 9, offset: 0,
    detects: "smooths by averaging each pixel with its 8 neighbours.",
  },
  gauss: {
    label: "Gaussian blur",
    k: [1, 2, 1, 2, 4, 2, 1, 2, 1],
    div: 16, offset: 0,
    detects: "a softer, weighted blur — centre counts most, corners least.",
  },
  sharpen: {
    label: "Sharpen",
    k: [0, -1, 0, -1, 5, -1, 0, -1, 0],
    div: 1, offset: 0,
    detects: "boosts the centre against its neighbours, exaggerating edges.",
  },
  laplacian: {
    label: "Edge (Laplacian)",
    k: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
    div: 1, offset: 0,
    detects: "every edge in every direction — bright where intensity changes.",
  },
  sobelx: {
    label: "Sobel-X",
    k: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    div: 1, offset: 128,
    detects: "vertical edges — strong response where intensity changes left-to-right.",
  },
  sobely: {
    label: "Sobel-Y",
    k: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    div: 1, offset: 128,
    detects: "horizontal edges — strong response where intensity changes top-to-bottom.",
  },
  emboss: {
    label: "Emboss",
    k: [-2, -1, 0, -1, 1, 1, 0, 1, 2],
    div: 1, offset: 128,
    detects: "a 3-D 'stamped metal' look — lights one side of every edge, darkens the other.",
  },
};

export default {
  key: "convolution",
  name: "Convolution",
  icon: "👁️",
  category: "Computer vision",
  tag: "CNN",
  blurb: "Slide little kernels over an image and watch edges, blurs and textures pop out.",
  eyebrow: "Computer vision · CNNs",
  title: 'Convolution: how a computer <span class="grad">sees edges</span>',
  lede:
    "Before a neural network can recognise a cat, it has to turn raw pixels into features. The workhorse is " +
    "<em>convolution</em>: slide a tiny grid of numbers — a kernel — across the image, multiplying and summing as you go. " +
    "Different kernels detect different things. Pick one, or edit it, and watch what the computer sees.",

  mount(root) {
    // ---- state ----
    let img = new Float64Array(N * N);   // source intensities 0..255 (row-major)
    let out = new Float64Array(N * N);   // convolved output 0..255
    let kernel = PRESETS.sobelx.k.slice();
    let div = PRESETS.sobelx.div;
    let offset = PRESETS.sobelx.offset;
    let presetKey = "sobelx";
    let scanI = 0, scanJ = 0;            // animated scan window centre
    let scanning = false;

    const at = (a, i, j) => a[i * N + j];
    const set = (a, i, j, v) => { a[i * N + j] = v; };

    // ---- build the synthetic source image procedurally (offline) ----
    function buildImage() {
      // Draw shapes on an offscreen canvas, then read intensities back.
      const off = document.createElement("canvas");
      off.width = N; off.height = N;
      const c = off.getContext("2d");

      // smooth horizontal gradient background (left dark -> right light)
      const grad = c.createLinearGradient(0, 0, N, 0);
      grad.addColorStop(0, "#202020");
      grad.addColorStop(1, "#c8c8c8");
      c.fillStyle = grad;
      c.fillRect(0, 0, N, N);

      // a filled circle (clean curved edges)
      c.fillStyle = "#f4f4f4";
      c.beginPath();
      c.arc(N * 0.30, N * 0.34, N * 0.16, 0, Math.PI * 2);
      c.fill();

      // a dark rectangle (clean straight edges)
      c.fillStyle = "#0a0a0a";
      c.fillRect(N * 0.55, N * 0.12, N * 0.30, N * 0.26);

      // a thick diagonal stripe (slanted edges)
      c.save();
      c.strokeStyle = "#ededed";
      c.lineWidth = N * 0.07;
      c.beginPath();
      c.moveTo(N * 0.10, N * 0.95);
      c.lineTo(N * 0.95, N * 0.55);
      c.stroke();
      c.restore();

      // read pixels back; grayscale = average of RGB
      let imgData;
      try {
        imgData = c.getImageData(0, 0, N, N).data;
      } catch (err) {
        imgData = null; // canvas read can fail in odd sandboxes — fall back below
      }
      const buf = new Float64Array(N * N);
      if (imgData) {
        for (let p = 0; p < N * N; p++) {
          const r = imgData[p * 4], g = imgData[p * 4 + 1], b = imgData[p * 4 + 2];
          buf[p] = (r + g + b) / 3;
        }
      } else {
        // pure-JS fallback: a gradient so the game still works
        for (let i = 0; i < N; i++)
          for (let j = 0; j < N; j++)
            buf[i * N + j] = (j / (N - 1)) * 255;
      }

      // sprinkle a little texture/noise so blur has something to chew on
      for (let p = 0; p < N * N; p++) {
        buf[p] = clamp(buf[p] + gaussRandom(0, 6), 0, 255);
      }
      img = buf;
    }

    // ---- the actual 2-D convolution ----
    // out[i][j] = Σ_{u,v in -1..1} K[u+1][v+1] · in[i+u][j+v]
    // edges handled by clamping the source index to image bounds.
    function convolve() {
      const safeDiv = div === 0 ? 1 : div;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          let acc = 0;
          for (let u = -1; u <= 1; u++) {
            const si = clamp(i + u, 0, N - 1);
            for (let v = -1; v <= 1; v++) {
              const sj = clamp(j + v, 0, N - 1);
              acc += kernel[(u + 1) * 3 + (v + 1)] * at(img, si, sj);
            }
          }
          set(out, i, j, clamp(acc / safeDiv + offset, 0, 255));
        }
      }
    }

    // ---- DOM: two canvases ----
    const inCanvas = el("canvas", { width: DISP, height: DISP, class: "conv-canvas",
      role: "img", "aria-label": "Synthetic input image: a circle, a rectangle and a diagonal stripe over a gradient." });
    const outCanvas = el("canvas", { width: DISP, height: DISP, class: "conv-canvas",
      role: "img", "aria-label": "The output feature map after convolution." });
    const inCtx = inCanvas.getContext("2d");
    const outCtx = outCanvas.getContext("2d");
    inCtx.imageSmoothingEnabled = false;
    outCtx.imageSmoothingEnabled = false;

    // offscreen N x N image buffers we blit (scaled up) onto the display canvases
    const inGlyph = document.createElement("canvas"); inGlyph.width = N; inGlyph.height = N;
    const outGlyph = document.createElement("canvas"); outGlyph.width = N; outGlyph.height = N;
    const inGCtx = inGlyph.getContext("2d");
    const outGCtx = outGlyph.getContext("2d");
    const inImageData = inGCtx.createImageData(N, N);
    const outImageData = outGCtx.createImageData(N, N);

    // ---- preset segmented control ----
    const segGroup = el("div", { class: "seg-group" });
    const segButtons = {};
    for (const [key, p] of Object.entries(PRESETS)) {
      const b = el("button", { class: key === presetKey ? "active" : "" }, p.label);
      b.addEventListener("click", () => applyPreset(key));
      segButtons[key] = b;
      segGroup.append(b);
    }

    // ---- editable 3x3 number grid ----
    const cellInputs = [];
    const grid = el("div", { class: "conv-grid" });
    for (let idx = 0; idx < 9; idx++) {
      const inp = el("input", { type: "number", step: "any", value: kernel[idx], class: "conv-cell" });
      inp.addEventListener("input", () => {
        const val = parseFloat(inp.value);
        kernel[idx] = isNaN(val) ? 0 : val;
        presetKey = "custom";
        markCustom();
        recompute();
      });
      cellInputs.push(inp);
      grid.append(inp);
    }

    // ---- divisor input ----
    const divInput = el("input", { type: "number", step: "any", value: div, class: "conv-cell", style: { width: "5.5em" } });
    divInput.addEventListener("input", () => {
      const v = parseFloat(divInput.value);
      div = isNaN(v) ? 1 : v;
      presetKey = "custom"; markCustom(); recompute();
    });
    const offsetInput = el("input", { type: "number", step: "any", value: offset, class: "conv-cell", style: { width: "5.5em" } });
    offsetInput.addEventListener("input", () => {
      const v = parseFloat(offsetInput.value);
      offset = isNaN(v) ? 0 : v;
      presetKey = "custom"; markCustom(); recompute();
    });

    // ---- buttons ----
    const scanBtn = el("button", {}, "Animate scan ▸");
    const newImgBtn = ghostButton("New image");

    // ---- readouts ----
    const nameEl = el("span", { class: "v" }, PRESETS.sobelx.label);
    const divEl = el("span", { class: "v" }, "1");
    const offEl = el("span", { class: "v" }, "+128");
    const verdict = el("p", { class: "verdict" });

    // ---- layout ----
    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" },
            el("div", { class: "conv-pair" },
              el("div", { class: "conv-col" },
                el("div", { class: "conv-tag" }, "Input image"),
                inCanvas),
              el("div", { class: "conv-col" },
                el("div", { class: "conv-tag" }, "Output (feature map)"),
                outCanvas))),
          el("p", { class: "caption" },
            "An image is just a grid of numbers. The kernel slides over it, multiply-and-sum at every spot. " +
            "Hit ", el("b", {}, "Animate scan"), " to watch the 3×3 window sweep and reveal the output pixel by pixel.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Current kernel"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Preset"), nameEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Divisor"), divEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Offset"), offEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Pick a kernel"),
            segGroup),
          el("div", { class: "panel" },
            el("h3", {}, "Edit the 3×3 weights"),
            grid,
            el("div", { class: "controls" },
              el("label", { class: "toggle" }, "÷ divisor ", divInput),
              el("label", { class: "toggle" }, "+ offset ", offsetInput)),
            el("p", { class: "caption" }, "Type any numbers — the output recomputes instantly.")),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, scanBtn, newImgBtn))),
      ),
      verdict
    );

    // ---- rendering ----
    function blitGray(imageData, gctx, glyph, dctx, src) {
      const d = imageData.data;
      for (let p = 0; p < N * N; p++) {
        const g = src[p] | 0;
        d[p * 4] = g; d[p * 4 + 1] = g; d[p * 4 + 2] = g; d[p * 4 + 3] = 255;
      }
      gctx.putImageData(imageData, 0, 0);
      dctx.imageSmoothingEnabled = false;
      dctx.clearRect(0, 0, DISP, DISP);
      dctx.drawImage(glyph, 0, 0, N, N, 0, 0, DISP, DISP);
    }

    function drawScanOverlay() {
      // highlight the current 3x3 window on the input, and the target pixel on output
      const cell = DISP / N;
      // input window (3x3 around scanI,scanJ)
      inCtx.save();
      inCtx.strokeStyle = "#ffd166";
      inCtx.lineWidth = 2;
      inCtx.strokeRect((scanJ - 1) * cell, (scanI - 1) * cell, cell * 3, cell * 3);
      inCtx.fillStyle = "rgba(255,209,102,0.18)";
      inCtx.fillRect((scanJ - 1) * cell, (scanI - 1) * cell, cell * 3, cell * 3);
      inCtx.restore();
      // output target pixel
      outCtx.save();
      outCtx.strokeStyle = "#5ad2ff";
      outCtx.lineWidth = 2;
      outCtx.strokeRect(scanJ * cell, scanI * cell, cell, cell);
      outCtx.restore();
    }

    function renderInput() { blitGray(inImageData, inGCtx, inGlyph, inCtx, img); }

    function renderOutput(upTo) {
      // when scanning, only reveal output up to the scan position (raster order)
      if (upTo == null) {
        blitGray(outImageData, outGCtx, outGlyph, outCtx, out);
      } else {
        const d = outImageData.data;
        for (let p = 0; p < N * N; p++) {
          const revealed = p <= upTo;
          const g = revealed ? (out[p] | 0) : 14; // unrevealed = dark
          d[p * 4] = g; d[p * 4 + 1] = g; d[p * 4 + 2] = g; d[p * 4 + 3] = 255;
        }
        outGCtx.putImageData(outImageData, 0, 0);
        outCtx.imageSmoothingEnabled = false;
        outCtx.clearRect(0, 0, DISP, DISP);
        outCtx.drawImage(outGlyph, 0, 0, N, N, 0, 0, DISP, DISP);
      }
    }

    function recompute() {
      convolve();
      renderInput();
      if (scanning) {
        renderOutput(scanI * N + scanJ);
        drawScanOverlay();
      } else {
        renderOutput(null);
      }
      updateReadout();
    }

    // ---- readout / verdict ----
    function fmtOffset(o) { return o > 0 ? `+${o}` : `${o}`; }

    function updateReadout() {
      const p = PRESETS[presetKey];
      nameEl.textContent = p ? p.label : "Custom";
      divEl.textContent = String(div);
      offEl.textContent = fmtOffset(offset);
      const detects = p ? p.detects : "whatever pattern your hand-tuned weights respond to.";
      verdict.innerHTML =
        `The <b>${p ? p.label : "custom"}</b> kernel detects ${detects} ` +
        `Each output pixel is the sum of nine input pixels weighted by the grid` +
        (div !== 1 ? `, divided by ${div}` : "") +
        (offset !== 0 ? `, then shifted by ${fmtOffset(offset)}` : "") + `.`;
    }

    // ---- preset / custom switching ----
    function applyPreset(key) {
      const p = PRESETS[key];
      if (!p) return;
      presetKey = key;
      kernel = p.k.slice();
      div = p.div;
      offset = p.offset;
      // reflect into inputs
      for (let i = 0; i < 9; i++) cellInputs[i].value = kernel[i];
      divInput.value = div;
      offsetInput.value = offset;
      // highlight active seg button
      for (const [k, b] of Object.entries(segButtons)) b.classList.toggle("active", k === key);
      recompute();
    }

    function markCustom() {
      for (const b of Object.values(segButtons)) b.classList.remove("active");
    }

    // ---- animated scan loop ----
    let acc = 0;
    const loop = new Loop((dt) => {
      acc += dt;
      // advance ~ a few pixels per frame so the sweep is visible but not glacial
      while (acc >= 0.012) {
        acc -= 0.012;
        scanJ++;
        if (scanJ >= N) { scanJ = 0; scanI++; }
        if (scanI >= N) { scanI = 0; } // loop the sweep forever
      }
      renderInput();
      renderOutput(scanI * N + scanJ);
      drawScanOverlay();
    });

    function startScan() {
      scanning = true;
      scanI = 0; scanJ = 0; acc = 0;
      loop.start();
      scanBtn.textContent = "Stop scan ◼";
    }
    function stopScan() {
      scanning = false;
      loop.stop();
      scanBtn.textContent = "Animate scan ▸";
      renderInput();
      renderOutput(null); // show the full finished output
    }

    scanBtn.addEventListener("click", () => { scanning ? stopScan() : startScan(); });
    newImgBtn.addEventListener("click", () => { buildImage(); recompute(); });

    // ---- init ----
    buildImage();
    recompute();

    return {
      onEnter() {
        // redraw on entry; stay paused unless the user starts the scan
        recompute();
      },
      onLeave() {
        // MUST stop the loop and reset the button label
        loop.stop();
        scanning = false;
        scanBtn.textContent = "Animate scan ▸";
        renderOutput(null);
      },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: turning pixels into features</h3>
    <p>A neural network can't do much with three million raw pixel values. It needs <b>features</b> — "there's an edge here", "a corner there", "a patch of fur". Convolution is the operation that extracts them. It's the <code>C</code> in <b>CNN</b> (Convolutional Neural Network) and it powers nearly every modern vision system: photo tagging, self-driving perception, medical imaging.</p>
    <p>The starting point is the realisation on screen above: <b>an image is just a grid of numbers</b>. Each pixel is an intensity from 0 (black) to 255 (white).</p>
  </div>
  <div class="lesson-block">
    <h3>🔁 What you can see happening</h3>
    <ol class="steps">
      <li><b>Take a tiny kernel</b> — here a 3×3 grid of weights (nine numbers).</li>
      <li><b>Place it over the image</b> so its centre sits on one pixel. Press <em>Animate scan</em> to watch the yellow window slide.</li>
      <li><b>Multiply and sum.</b> Multiply each kernel weight by the pixel underneath it and add up all nine products. That single number becomes one pixel of the output.</li>
      <li><b>Slide to the next position</b> and repeat across the whole image. The result is a new image — a <b>feature map</b>.</li>
    </ol>
    <p>Different kernels read out different things: <b>blur</b> kernels average neighbours (the weights are all positive and sum to one); <b>edge</b> kernels subtract neighbours from the centre, so they're near zero on flat regions and large where intensity jumps; <b>Sobel</b> kernels are lopsided left-vs-right (or top-vs-bottom), so they measure the <em>direction</em> of the gradient.</p>
  </div>
  <div class="lesson-block">
    <h3>📐 The maths</h3>
    <p>Discrete 2-D convolution (technically cross-correlation, the convention CNNs use) of an image <code>I</code> with a kernel <code>K</code>:</p>
    <div class="formula">out[i][j] = Σ<sub>u=−1..1</sub> Σ<sub>v=−1..1</sub> K[u+1][v+1] · I[i+u][j+v]</div>
    <p>For blur kernels we divide the sum by the total of the weights so brightness is preserved; for edge and emboss kernels the sum can be negative, so we add an offset of <b>128</b> (mid-grey) and clamp the result to 0..255. At the borders, where the window hangs off the edge, this game <b>clamps</b> the source index to the nearest valid pixel.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ The catches</h3>
    <ul class="bullets">
      <li><b>Normalisation matters.</b> If a blur kernel's weights don't sum to one, the whole image gets brighter or darker. Try setting the divisor wrong and watch it wash out.</li>
      <li><b>Edges go negative.</b> Edge kernels produce negative numbers where intensity falls; without the +128 offset everything below zero would just clamp to black and you'd lose half the signal.</li>
      <li><b>Border handling is a choice.</b> Clamping (used here), zero-padding, or wrapping each give slightly different results at the rim — a real source of bugs.</li>
      <li><b>One 3×3 kernel sees almost nothing.</b> It only looks at a 3-pixel neighbourhood. Real networks stack many layers so the <em>receptive field</em> grows.</li>
    </ul>
  </div>
  <div class="lesson-block">
    <h3>🧠 The key point: in a CNN, kernels are <em>learned</em></h3>
    <p>Here you hand-pick or hand-tune the nine numbers. In a real convolutional neural network <b>nobody chooses them</b> — the kernel weights are parameters, learned by gradient descent (exactly the training loop you can play with in the <b>Neural Network</b> game). The network discovers, on its own, that a Sobel-like edge detector is useful in the first layer.</p>
    <p>Two ideas make this cheap and powerful:</p>
    <ul class="bullets">
      <li><b>Weight sharing.</b> The same nine numbers are reused at every position, so a layer has only a handful of parameters instead of one per pixel — and a feature learned in one corner works everywhere.</li>
      <li><b>Locality, then depth.</b> Each kernel sees a small patch, but stacking conv layers composes them: layer 1 finds <em>edges</em> → layer 2 combines edges into <em>textures and corners</em> → deeper layers into <em>object parts</em> → finally whole <em>objects</em> (a cat).</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> convolution slides a tiny weight grid over an image and multiply-sums at every spot to build a feature map — and the magic of CNNs is that those weights are <em>learned</em>, so the network teaches itself which edges and textures to look for.</p>
  </div>`,
};
