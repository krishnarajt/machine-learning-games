// =====================================================================
//  shared.js — the toolkit every game is built on.
//
//  Four things live here:
//    1. el()      — a tiny hyperscript helper for building DOM in JS.
//    2. Loop      — a start/stop wrapper around requestAnimationFrame.
//    3. Plot      — a 2-D canvas with data<->pixel mapping, mouse hit
//                   testing, a grid, axes and drawing helpers. This is the
//                   workhorse behind almost every game.
//    4. helpers   — maths, randomness, colours and dataset generators.
//
//  Coordinate convention for Plot: you give it a data-space window
//  (xmin..xmax, ymin..ymax). Data y grows UP the screen (normal maths axes),
//  which is the opposite of raw canvas pixels — Plot handles the flip.
// =====================================================================

/* --------------------------------------------------------------- *
 * 1. DOM — el(tag, attrs?, ...children)
 *    attrs special keys: class, html (innerHTML), text (textContent),
 *    style (object), onClick/onInput/... (event listeners). Children may be
 *    nodes, strings, arrays, or null/false (skipped).
 * --------------------------------------------------------------- */
export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs && typeof attrs === "object" && attrs.nodeType === undefined && !Array.isArray(attrs)) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k === "text") node.textContent = v;
      else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
      else if (k === "dataset" && typeof v === "object") Object.assign(node.dataset, v);
      else if (/^on[A-Z]/.test(k) && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
      else if (v === true) node.setAttribute(k, "");
      else node.setAttribute(k, v);
    }
  } else if (attrs != null) {
    // called as el(tag, child) — treat second arg as a child
    children.unshift(attrs);
  }
  for (const kid of children.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

/* Convenience builders for the common control widgets so every game's
 * controls look identical without copy-pasting markup. */
export function button(label, attrs = {}) {
  return el("button", { ...attrs }, label);
}
export function ghostButton(label, attrs = {}) {
  return el("button", { class: "ghost", ...attrs }, label);
}
// A labelled range slider. Returns { wrap, input, value } where `value` is the
// <span> that you update with the formatted number.
export function slider({ min, max, step, value, label, format = (v) => v }) {
  const input = el("input", { type: "range", min, max, step, value });
  const valueEl = el("span", { class: "ctl-val" }, format(value));
  input.addEventListener("input", () => (valueEl.textContent = format(parseFloat(input.value))));
  const wrap = el("label", { class: "toggle" }, label ? `${label} ` : "", valueEl, input);
  return { wrap, input, value: valueEl };
}

/* --------------------------------------------------------------- *
 * 2. Loop — requestAnimationFrame with start()/stop().
 *    fn receives (dt, t) where dt is seconds since the last frame.
 * --------------------------------------------------------------- */
export class Loop {
  constructor(fn) {
    this.fn = fn;
    this.running = false;
    this._raf = 0;
    this._last = 0;
  }
  start() {
    if (this.running) return;
    this.running = true;
    this._last = 0;
    const step = (t) => {
      if (!this.running) return;
      const dt = this._last ? Math.min(0.05, (t - this._last) / 1000) : 0;
      this._last = t;
      this.fn(dt, t);
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }
  stop() {
    this.running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
  }
  toggle() {
    this.running ? this.stop() : this.start();
  }
}

export const REDUCED_MOTION =
  typeof window !== "undefined" &&
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* --------------------------------------------------------------- *
 * 3. Plot — a 2-D canvas mapped to a data-space window.
 * --------------------------------------------------------------- */
export class Plot {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.xmin = opts.xmin ?? -1;
    this.xmax = opts.xmax ?? 1;
    this.ymin = opts.ymin ?? -1;
    this.ymax = opts.ymax ?? 1;
    const pad = opts.pad ?? 30;
    this.padL = opts.padL ?? pad;
    this.padR = opts.padR ?? pad;
    this.padT = opts.padT ?? pad;
    this.padB = opts.padB ?? pad;
  }
  get W() { return this.canvas.width; }
  get H() { return this.canvas.height; }
  get plotW() { return this.W - this.padL - this.padR; }
  get plotH() { return this.H - this.padT - this.padB; }

  setWindow({ xmin, xmax, ymin, ymax }) {
    if (xmin !== undefined) this.xmin = xmin;
    if (xmax !== undefined) this.xmax = xmax;
    if (ymin !== undefined) this.ymin = ymin;
    if (ymax !== undefined) this.ymax = ymax;
  }

  // data -> pixel
  sx(x) { return this.padL + ((x - this.xmin) / (this.xmax - this.xmin)) * this.plotW; }
  sy(y) { return this.padT + (1 - (y - this.ymin) / (this.ymax - this.ymin)) * this.plotH; }
  // pixel -> data
  ix(px) { return this.xmin + ((px - this.padL) / this.plotW) * (this.xmax - this.xmin); }
  iy(py) { return this.ymin + (1 - (py - this.padT) / this.plotH) * (this.ymax - this.ymin); }
  // scale a data-space length to pixels (x axis)
  scaleX(dx) { return (dx / (this.xmax - this.xmin)) * this.plotW; }

  // mouse/touch event -> data coords (accounts for CSS scaling of the canvas)
  eventToData(evt) {
    const rect = this.canvas.getBoundingClientRect();
    const src = evt.touches && evt.touches[0] ? evt.touches[0] : evt;
    const px = (src.clientX - rect.left) * (this.canvas.width / rect.width);
    const py = (src.clientY - rect.top) * (this.canvas.height / rect.height);
    return { x: this.ix(px), y: this.iy(py), px, py };
  }

  clear(fill = "#070d1a") {
    this.ctx.clearRect(0, 0, this.W, this.H);
    if (fill) {
      this.ctx.fillStyle = fill;
      this.ctx.fillRect(0, 0, this.W, this.H);
    }
  }

  grid(step = 0.5, color = "rgba(255,255,255,0.06)") {
    const c = this.ctx;
    c.strokeStyle = color;
    c.lineWidth = 1;
    c.beginPath();
    const x0 = Math.ceil(this.xmin / step) * step;
    for (let x = x0; x <= this.xmax + 1e-9; x += step) {
      c.moveTo(this.sx(x), this.padT);
      c.lineTo(this.sx(x), this.padT + this.plotH);
    }
    const y0 = Math.ceil(this.ymin / step) * step;
    for (let y = y0; y <= this.ymax + 1e-9; y += step) {
      c.moveTo(this.padL, this.sy(y));
      c.lineTo(this.padL + this.plotW, this.sy(y));
    }
    c.stroke();
  }

  // x/y axes through the data origin (0,0) if it is in view
  axes(color = "rgba(255,255,255,0.22)") {
    const c = this.ctx;
    c.strokeStyle = color;
    c.lineWidth = 1.25;
    c.beginPath();
    if (this.ymin <= 0 && this.ymax >= 0) {
      c.moveTo(this.padL, this.sy(0));
      c.lineTo(this.padL + this.plotW, this.sy(0));
    }
    if (this.xmin <= 0 && this.xmax >= 0) {
      c.moveTo(this.sx(0), this.padT);
      c.lineTo(this.sx(0), this.padT + this.plotH);
    }
    c.stroke();
  }

  // Fill the plot area with a decision/scalar field. `fn(x, y)` is evaluated
  // at the centre of each cell and must return a CSS colour (or null to skip).
  shade(fn, res = 44) {
    const c = this.ctx;
    const cols = res;
    const rows = Math.max(2, Math.round(res * (this.plotH / this.plotW)));
    const cw = this.plotW / cols;
    const ch = this.plotH / rows;
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        const px = this.padL + (i + 0.5) * cw;
        const py = this.padT + (j + 0.5) * ch;
        const color = fn(this.ix(px), this.iy(py));
        if (!color) continue;
        c.fillStyle = color;
        c.fillRect(this.padL + i * cw, this.padT + j * ch, cw + 1, ch + 1);
      }
    }
  }

  // a filled data-space point
  point(x, y, r = 5, fill = "#5ad2ff", stroke = "rgba(7,13,26,0.85)", strokeW = 1.5) {
    const c = this.ctx;
    c.beginPath();
    c.arc(this.sx(x), this.sy(y), r, 0, Math.PI * 2);
    c.fillStyle = fill;
    c.fill();
    if (stroke) {
      c.lineWidth = strokeW;
      c.strokeStyle = stroke;
      c.stroke();
    }
  }

  // a line between two data points
  line(x0, y0, x1, y1, color = "rgba(255,255,255,0.9)", width = 2, dash = null) {
    const c = this.ctx;
    c.save();
    c.strokeStyle = color;
    c.lineWidth = width;
    if (dash) c.setLineDash(dash);
    c.beginPath();
    c.moveTo(this.sx(x0), this.sy(y0));
    c.lineTo(this.sx(x1), this.sy(y1));
    c.stroke();
    c.restore();
  }

  // a polyline through an array of [x,y] data points
  path(pts, color = "rgba(255,255,255,0.9)", width = 2, dash = null) {
    if (pts.length < 2) return;
    const c = this.ctx;
    c.save();
    c.strokeStyle = color;
    c.lineWidth = width;
    if (dash) c.setLineDash(dash);
    c.beginPath();
    pts.forEach(([x, y], i) => (i ? c.lineTo(this.sx(x), this.sy(y)) : c.moveTo(this.sx(x), this.sy(y))));
    c.stroke();
    c.restore();
  }

  // clip subsequent drawing to the plot rectangle (call inside save/restore)
  clipToPlot() {
    const c = this.ctx;
    c.beginPath();
    c.rect(this.padL, this.padT, this.plotW, this.plotH);
    c.clip();
  }

  label(text, px, py, color = "rgba(234,238,251,0.7)", align = "left", font = "12px 'JetBrains Mono', monospace") {
    const c = this.ctx;
    c.fillStyle = color;
    c.font = font;
    c.textAlign = align;
    c.fillText(text, px, py);
    c.textAlign = "left";
  }
}

/* --------------------------------------------------------------- *
 * 4a. Maths helpers
 * --------------------------------------------------------------- */
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const dist2 = (ax, ay, bx, by) => (ax - bx) ** 2 + (ay - by) ** 2;
export const dist = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);
export const sum = (a) => a.reduce((s, v) => s + v, 0);
export const mean = (a) => (a.length ? sum(a) / a.length : 0);
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
export const range = (n) => Array.from({ length: n }, (_, i) => i);
export const round = (v, d = 2) => {
  const p = 10 ** d;
  return Math.round(v * p) / p;
};

/* --------------------------------------------------------------- *
 * 4b. Randomness (browser Math.random is fine here)
 * --------------------------------------------------------------- */
export const randRange = (a, b) => a + Math.random() * (b - a);
export const randInt = (n) => Math.floor(Math.random() * n);
export const pick = (arr) => arr[randInt(arr.length)];
// standard normal via Box–Muller
export function gaussRandom(mean = 0, sd = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
export function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = randInt(i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/* --------------------------------------------------------------- *
 * 4c. Colours
 * --------------------------------------------------------------- */
// class palette — cyan, coral, violet, amber, teal (distinct + on-brand)
export const CLASS_COLORS = ["#5ad2ff", "#ff7a59", "#b388ff", "#ffd166", "#43e0c8"];
export const classColor = (i) => CLASS_COLORS[((i % CLASS_COLORS.length) + CLASS_COLORS.length) % CLASS_COLORS.length];

export function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const n = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return { r: parseInt(n.slice(0, 2), 16), g: parseInt(n.slice(2, 4), 16), b: parseInt(n.slice(4, 6), 16) };
}
export function withAlpha(hex, a) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}
export function mixHex(hexA, hexB, t) {
  const A = hexToRgb(hexA), B = hexToRgb(hexB);
  const r = Math.round(lerp(A.r, B.r, t));
  const g = Math.round(lerp(A.g, B.g, t));
  const b = Math.round(lerp(A.b, B.b, t));
  return `rgb(${r},${g},${b})`;
}
// perceptual-ish cool->warm ramp for scalar heatmaps (t in 0..1)
const HEAT_STOPS = ["#1a2b6b", "#1f7ae0", "#43e0c8", "#ffd166", "#ff7a59", "#e23b5a"];
export function heat(t) {
  t = clamp(t, 0, 1);
  const s = t * (HEAT_STOPS.length - 1);
  const i = Math.floor(s);
  if (i >= HEAT_STOPS.length - 1) return HEAT_STOPS[HEAT_STOPS.length - 1];
  return mixHex(HEAT_STOPS[i], HEAT_STOPS[i + 1], s - i);
}

/* --------------------------------------------------------------- *
 * 4d. Dataset generators
 *     Points are { x, y } objects; labelled sets add `label` (0,1,2,...).
 *     Synthetic shapes are scaled to sit roughly inside [-1, 1] on both axes.
 * --------------------------------------------------------------- */
export function gauss(n, cx, cy, sd = 0.12, label = 0) {
  return range(n).map(() => ({ x: gaussRandom(cx, sd), y: gaussRandom(cy, sd), label }));
}

// k labelled Gaussian blobs from an array of [cx, cy] centres
export function blobs(centers, nPer = 40, sd = 0.12) {
  return centers.flatMap(([cx, cy], i) => gauss(nPer, cx, cy, sd, i));
}

// two interleaving half-moons (classic non-linear 2-class set), label 0/1
export function twoMoons(n = 120, noise = 0.08) {
  const out = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = Math.PI * (i / half);
    out.push({ x: Math.cos(t), y: Math.sin(t), label: 0 });
  }
  for (let i = 0; i < n - half; i++) {
    const t = Math.PI * (i / (n - half));
    out.push({ x: 1 - Math.cos(t), y: -Math.sin(t) + 0.35, label: 1 });
  }
  // centre, scale to ~[-1,1], add noise
  return out.map((p) => ({
    x: (p.x - 0.5) * 0.9 + gaussRandom(0, noise),
    y: (p.y - 0.18) * 0.9 + gaussRandom(0, noise),
    label: p.label,
  }));
}

// concentric circles: inner = class 0, outer ring = class 1
export function circles(n = 120, noise = 0.06) {
  const out = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const a = randRange(0, Math.PI * 2);
    const r = 0.32 + gaussRandom(0, noise);
    out.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 0 });
  }
  for (let i = 0; i < n - half; i++) {
    const a = randRange(0, Math.PI * 2);
    const r = 0.8 + gaussRandom(0, noise);
    out.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 1 });
  }
  return out;
}

// XOR pattern: opposite quadrants share a class
export function xor(n = 120, noise = 0.12) {
  return range(n).map(() => {
    const x = randRange(-0.85, 0.85);
    const y = randRange(-0.85, 0.85);
    return { x: x + gaussRandom(0, noise), y: y + gaussRandom(0, noise), label: x * y > 0 ? 0 : 1 };
  });
}

// two-arm spiral (each arm a class)
export function spiral(nPer = 70, noise = 0.04) {
  const out = [];
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < nPer; i++) {
      const r = (i / nPer) * 0.95;
      const t = (i / nPer) * 3.2 + c * Math.PI;
      out.push({ x: r * Math.cos(t) + gaussRandom(0, noise), y: r * Math.sin(t) + gaussRandom(0, noise), label: c });
    }
  }
  return out;
}

// noisy points along a line y = slope*x + intercept (for regression)
export function linearData(n = 30, slope = 0.7, intercept = 0.05, noise = 0.18, xmin = -0.9, xmax = 0.9) {
  return range(n).map(() => {
    const x = randRange(xmin, xmax);
    return { x, y: slope * x + intercept + gaussRandom(0, noise) };
  });
}
