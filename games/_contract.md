# Game module contract

Every game is one file in `games/` that **default-exports a single object**.
`main.js` builds the page chrome (sidebar entry, view header, lesson container)
and the game only fills in its interactive area.

```js
import { el, slider, ghostButton, Loop, Plot, /* …whatever you need… */ } from "../shared.js";

export default {
  key: "kmeans",            // unique slug; also the URL hash (#kmeans)
  name: "k-Means",          // sidebar + tile label
  icon: "⭕",               // one emoji
  category: "Clustering",   // groups the sidebar; reuse exact existing strings
  tag: "Unsup",             // tiny pill in the sidebar (≤5 chars), optional
  blurb: "One sentence for the dashboard tile.",
  eyebrow: "Clustering · Unsupervised",   // small label above the title
  title: 'k-Means: <span class="grad">finding groups</span>',  // innerHTML of the <h1>
  lede: "1–3 sentence intro. innerHTML allowed (<em>, <b>, <code>).",

  // Build the interactive UI INTO `root` (an empty <div>). Wire all logic here.
  // Return a lifecycle handle so animation loops pause off-screen.
  mount(root) {
    // ...create DOM with el(), append to root, attach listeners...
    return {
      onEnter() {},   // called each time the view is shown (resume/redraw)
      onLeave() {},   // called when navigating away (MUST stop any Loop)
    };
  },

  lesson: `<div class="lesson-block">…</div> …`,  // HTML string, see below
};
```

## Hard rules

1. **Import only from `../shared.js`** and use the browser canvas API. Do not add
   libraries, fetch anything, or touch `localStorage`/network. The site is offline-static.
2. **Self-contained.** Build *all* your DOM inside `root`. Never `getElementById`
   for things outside your subtree, and never assume IDs are global — other games
   exist on the same page. Keep references to the elements you create.
3. **Pause off-screen.** If you use a `Loop`, you MUST stop it in `onLeave()`.
   Don't auto-start heavy animation on mount; let the user press play, or start it
   in `onEnter()`.
4. **No top-level side effects** beyond defining the export. Nothing runs until
   `mount()` is called.
5. Respect `REDUCED_MOTION` from shared.js for purely decorative animation.

## What `shared.js` gives you

- `el(tag, attrs?, ...children)` — hyperscript. attrs: `class`, `html`, `text`,
  `style` (object), `dataset` (object), `onClick`/`onInput`/… listeners. Children
  may be nodes, strings, arrays, or null.
- `button(label, attrs?)`, `ghostButton(label, attrs?)`.
- `slider({min,max,step,value,label,format})` → `{ wrap, input, value }`. Append
  `wrap`; read `input.value`; the `value` span auto-updates its text.
- `Loop(fn)` → `.start()/.stop()/.toggle()`, `.running`. `fn(dt, t)` where `dt` is
  seconds since the last frame (clamped ≤ 0.05).
- `Plot(canvas, {xmin,xmax,ymin,ymax,pad|padL,padR,padT,padB})` — 2-D canvas:
  `sx(x)/sy(y)` data→pixel, `ix/iy` pixel→data, `eventToData(evt)`→`{x,y,px,py}`
  (handles CSS scaling — use for mouse), `clear(fill)`, `grid(step,color)`,
  `axes(color)`, `shade(fn,res)` (fill background by `fn(x,y)→cssColor|null`,
  great for decision boundaries), `point(x,y,r,fill,stroke,strokeW)`,
  `line(x0,y0,x1,y1,color,width,dash)`, `path([[x,y]…],color,width,dash)`,
  `clipToPlot()`, `label(text,px,py,color,align,font)`. `plot.ctx` is the raw 2-D context.
- maths: `clamp, lerp, dist2, dist, sum, mean, sigmoid, range(n), round(v,d)`.
- random: `randRange(a,b), randInt(n), pick(arr), gaussRandom(mean,sd), shuffle(arr)`.
- colours: `CLASS_COLORS`, `classColor(i)`, `hexToRgb`, `withAlpha(hex,a)`,
  `mixHex(a,b,t)`, `heat(t)` (0..1 cool→warm ramp).
- datasets (return `{x,y,label?}`, scaled to ~[-1,1]): `gauss(n,cx,cy,sd,label)`,
  `blobs(centers,nPer,sd)`, `twoMoons(n,noise)`, `circles(n,noise)`, `xor(n,noise)`,
  `spiral(nPer,noise)`, `linearData(n,slope,intercept,noise,xmin,xmax)`.

## Canvas sizing

Create the canvas with explicit `width`/`height` attributes (internal resolution,
e.g. 760×460). CSS stretches it to 100% width — `Plot.eventToData` already corrects
for that, so always map mouse coords through it. Put the canvas inside
`<div class="canvas-stage">`.

## Recommended view layout (matches the CSS)

```
<div class="game-grid">
  <div>
    <div class="canvas-stage"><canvas …></div>
    <p class="caption">hint about interaction</p>
  </div>
  <div class="game-side">
    <div class="panel"><h3>…</h3> …kv rows…</div>
    <div class="panel"><h3>Controls</h3> <div class="controls">…</div></div>
    <div class="legend">…</div>
  </div>
</div>
<p class="verdict">live, plain-English readout of what just happened</p>
```

Useful CSS classes (don't invent new ones): `game-grid`, `game-side`, `panel`,
`canvas-stage`, `controls`, `toggle`, `seg-group` (+ `.active` on the selected
button), `stats`/`stats.two`/`stat-card`/`stat-number`/`stat-label`, `kv`(`.k`/`.v`),
`bars`/`bar-row`(`.bl`)/`bar-track`/`bar-fill`(+`.warm`)/`bar-val`, `tokens`/`token`
(+`.spam`/`.ham`), `legend`/`legend-item`/`legend-dot`(+`.sq`)/`legend-line`,
`pill`(+`.cyan`/`.teal`), `verdict`, `caption`, `grad`/`grad-warm` (gradient text).

## Lesson format (the `lesson` string)

Plain HTML string injected after the game. Use these blocks (already styled):
`<div class="lesson-block"><h3>…</h3><p>…</p></div>`, `<ol class="steps"><li>…`,
`<ul class="bullets"><li>…`, `<div class="formula">…</div>` (monospace, supports
`<sub>/<sup>`), `<div class="callout">`/`<div class="callout key">`,
`<table class="mini-table">`. Explain: what problem it solves, the steps you can
see on screen, the key maths/intuition, the failure modes. Friendly but accurate.
Finish with a one-line `callout key` summary.

## Lifecycle gotchas

- `mount()` runs once (first time the view opens). Do initial draw there.
- `onEnter()` runs every open after that — re-draw, optionally resume.
- `onLeave()` must `loop.stop()`. Reset any play/pause button label too.
