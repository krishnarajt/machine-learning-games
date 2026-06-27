// =====================================================================
//  Q-Learning — a reinforcement-learning agent in a grid world.
//
//  Tabular Q-learning. The agent lives in an editable grid, takes
//  actions (up/down/left/right), collects rewards, and updates a
//  table Q[row][col][action] with the temporal-difference rule:
//
//     Q(s,a) ← Q(s,a) + α[ r + γ·max_a' Q(s',a') − Q(s,a) ]
//
//  We draw the grid ourselves on a plain canvas (cells, not a Plot):
//  each learnable cell is tinted by its state value V(s)=max_a Q(s,a)
//  and carries a small arrow for the greedy best action. Value visibly
//  spreads backward from the goal over many episodes.
// =====================================================================
import { el, button, ghostButton, slider, Loop, clamp, heat, randInt } from "../shared.js";

export default {
  key: "q-learning",
  name: "Q-Learning",
  icon: "🤖",
  category: "Reinforcement",
  tag: "RL",
  blurb: "Train an agent to cross a grid world using nothing but rewards and penalties.",
  eyebrow: "Reinforcement learning",
  title: 'Q-Learning: an agent that learns by <span class="grad">trial and reward</span>',
  lede:
    "No examples, no labels — just rewards. A Q-learning agent wanders a grid, bumps into hazards, stumbles onto the goal, " +
    "and slowly learns which moves pay off. Watch the value of each square light up and the best-action arrows organise " +
    "into a route as it plays episode after episode.",

  mount(root) {
    // ---- world geometry ----
    const COLS = 9, ROWS = 7;
    const CW = 760, CH = Math.round((760 * ROWS) / COLS); // ~591

    // cell types
    const EMPTY = 0, WALL = 1, HAZARD = 2, GOAL = 3;
    // actions: 0=up 1=down 2=left 3=right
    const DR = [-1, 1, 0, 0];
    const DC = [0, 0, -1, 1];
    const NA = 4;

    // ---- hyperparameters ----
    let epsilon = 0.3;       // exploration probability
    let alpha = 0.5;         // learning rate
    let gamma = 0.95;        // discount factor
    let autoDecay = true;    // decay epsilon toward floor over episodes
    const EPS_FLOOR = 0.05;
    let stepsPerFrame = 12;  // env steps per animation frame
    const MAX_STEPS = 80;

    // ---- world & learning state ----
    let grid;        // grid[r][c] -> cell type
    let start = { r: ROWS - 2, c: 1 };
    let goal = { r: 1, c: COLS - 2 };
    let Q;           // Q[r][c][a]
    let agent = { r: start.r, c: start.c };
    let episode = 0;
    let stepInEp = 0;
    let lastReward = 0;
    let epReward = 0;        // accumulating reward of the in-progress episode
    let lastEpReward = 0;    // total reward of the most recently finished episode
    const recent = [];       // recent episode outcomes (1 = reached goal, 0 = not)
    const RECENT_N = 30;

    // ---- paint tool ----
    let paintTool = "wall"; // wall | hazard | goal | start | erase
    let painting = false;

    // ---- DOM ----
    const canvas = el("canvas", {
      width: CW, height: CH, role: "img",
      "aria-label": "Grid world coloured by learned state value, with arrows showing the agent's best action in each cell.",
    });
    const ctx = canvas.getContext("2d");

    const trainBtn = button("Train ▸");
    const resetQBtn = ghostButton("Reset Q");
    const resetWorldBtn = ghostButton("Reset world");

    const epsSlider = slider({ min: 0, max: 1, step: 0.01, value: epsilon, label: "ε explore", format: (v) => v.toFixed(2) });
    const alphaSlider = slider({ min: 0.01, max: 1, step: 0.01, value: alpha, label: "α learn", format: (v) => v.toFixed(2) });
    const gammaSlider = slider({ min: 0.5, max: 0.999, step: 0.001, value: gamma, label: "γ discount", format: (v) => v.toFixed(3) });
    const speedSlider = slider({ min: 1, max: 60, step: 1, value: stepsPerFrame, label: "speed", format: (v) => `${v}×` });

    const decayChk = el("input", { type: "checkbox", checked: autoDecay });
    const decayWrap = el("label", { class: "toggle" }, decayChk, " auto-decay ε");

    // paint segmented control
    const toolDefs = [
      ["wall", "Wall"], ["hazard", "Hazard"], ["goal", "Goal"], ["start", "Start"], ["erase", "Erase"],
    ];
    const toolButtons = {};
    const segGroup = el("div", { class: "seg-group" });
    for (const [k, label] of toolDefs) {
      const b = el("button", { class: k === paintTool ? "active" : "" }, label);
      b.addEventListener("click", () => {
        paintTool = k;
        for (const kk of Object.keys(toolButtons)) toolButtons[kk].classList.toggle("active", kk === k);
      });
      toolButtons[k] = b;
      segGroup.append(b);
    }

    // stats
    const epEl = el("span", { class: "v" }, "0");
    const epsEl = el("span", { class: "v" }, epsilon.toFixed(2));
    const abEl = el("span", { class: "v" }, `${alpha.toFixed(2)} / ${gamma.toFixed(3)}`);
    const lastREl = el("span", { class: "v" }, "—");
    const successEl = el("span", { class: "v" }, "—");

    const verdict = el("p", { class: "verdict" },
      "Press " ,el("b", {}, "Train"), " and watch the squares light up. Early on the agent flails about; value then spreads backward from the green goal and the arrows snap into a route.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" },
            "Pick a paint tool and click (or drag) cells to edit the world. Editing resets what the agent has learned. The 🤖 token is the agent.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Learning state"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Episode"), epEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "ε exploration"), epsEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "α / γ"), abEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Last-episode reward"), lastREl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Success rate (last 30)"), successEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, trainBtn, resetQBtn, resetWorldBtn),
            el("div", { class: "controls" }, epsSlider.wrap, decayWrap),
            el("div", { class: "controls" }, alphaSlider.wrap, gammaSlider.wrap),
            el("div", { class: "controls" }, speedSlider.wrap)),
          el("div", { class: "panel" },
            el("h3", {}, "Paint the world"),
            el("div", { class: "controls" }, segGroup)),
          el("div", { class: "legend" },
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: heat(0) } }), "low value"),
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: heat(1) } }), "high value"),
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: "#1db954" } }), "goal +1"),
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: "#e23b5a" } }), "hazard −1"),
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: "#3a3f55" } }), "wall")))),
      verdict
    );

    // ---- world setup ----
    function blankGrid() {
      return Array.from({ length: ROWS }, () => Array.from({ length: COLS }, () => EMPTY));
    }

    function defaultWorld() {
      grid = blankGrid();
      // border-free interior with a few walls forming a corridor + some hazards
      const walls = [
        [1, 3], [2, 3], [3, 3], [4, 3],
        [2, 5], [3, 5], [4, 5], [5, 5],
        [1, 6],
      ];
      for (const [r, c] of walls) grid[r][c] = WALL;
      const hazards = [
        [4, 4], [3, 6], [5, 2], [2, 7],
      ];
      for (const [r, c] of hazards) grid[r][c] = HAZARD;
      start = { r: ROWS - 2, c: 1 };
      goal = { r: 1, c: COLS - 2 };
      grid[goal.r][goal.c] = GOAL;
      grid[start.r][start.c] = EMPTY;
    }

    function freshQ() {
      Q = Array.from({ length: ROWS }, () =>
        Array.from({ length: COLS }, () => new Float64Array(NA)));
    }

    function resetAgent() {
      agent = { r: start.r, c: start.c };
      stepInEp = 0;
      epReward = 0;
    }

    function isTerminal(r, c) {
      return grid[r][c] === GOAL || grid[r][c] === HAZARD;
    }
    function passable(r, c) {
      return r >= 0 && r < ROWS && c >= 0 && c < COLS && grid[r][c] !== WALL;
    }

    function argmaxQ(r, c) {
      const q = Q[r][c];
      let best = 0, bv = q[0];
      for (let a = 1; a < NA; a++) if (q[a] > bv) { bv = q[a]; best = a; }
      return best;
    }
    function maxQ(r, c) {
      const q = Q[r][c];
      let m = q[0];
      for (let a = 1; a < NA; a++) if (q[a] > m) m = q[a];
      return m;
    }

    // one environment step + Q update. Returns true if the episode ended.
    function envStep() {
      const r = agent.r, c = agent.c;
      // ε-greedy action selection
      let a;
      if (Math.random() < epsilon) a = randInt(NA);
      else a = argmaxQ(r, c);

      // attempt the move; walls / off-grid = stay in place
      let nr = r + DR[a], nc = c + DC[a];
      if (!passable(nr, nc)) { nr = r; nc = c; }

      // reward
      let reward;
      const type = grid[nr][nc];
      if (type === GOAL) reward = 1;
      else if (type === HAZARD) reward = -1;
      else reward = -0.01; // living cost per step

      const terminalNext = (type === GOAL || type === HAZARD);

      // Q-learning update: target = r + γ·max_a' Q(s',a'); if terminal, just r
      const target = terminalNext ? reward : reward + gamma * maxQ(nr, nc);
      Q[r][c][a] += alpha * (target - Q[r][c][a]);

      agent.r = nr; agent.c = nc;
      stepInEp++;
      epReward += reward;
      lastReward = reward;

      let ended = false, reachedGoal = false;
      if (terminalNext) { ended = true; reachedGoal = (type === GOAL); }
      else if (stepInEp >= MAX_STEPS) { ended = true; reachedGoal = false; }

      if (ended) finishEpisode(reachedGoal);
      return ended;
    }

    function finishEpisode(reachedGoal) {
      episode++;
      lastEpReward = epReward;
      recent.push(reachedGoal ? 1 : 0);
      if (recent.length > RECENT_N) recent.shift();
      if (autoDecay) {
        // exponential decay of ε toward the floor
        epsilon = EPS_FLOOR + (epsilon - EPS_FLOOR) * 0.985;
        epsSlider.input.value = epsilon;
        epsSlider.value.textContent = epsilon.toFixed(2);
      }
      resetAgent();
    }

    // ---- value-range helper for the heat map ----
    function valueRange() {
      let lo = Infinity, hi = -Infinity;
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          if (grid[r][c] === WALL || isTerminal(r, c)) continue;
          const v = maxQ(r, c);
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
      if (!isFinite(lo)) { lo = 0; hi = 0; }
      return { lo, hi };
    }

    // ---- drawing ----
    function cellRect(r, c) {
      const w = CW / COLS, h = CH / ROWS;
      return { x: c * w, y: r * h, w, h };
    }

    function drawArrow(cx, cy, a, size, color) {
      // unit direction for action a
      const dx = DC[a], dy = DR[a];
      const hx = cx + dx * size, hy = cy + dy * size;
      const tx = cx - dx * size, ty = cy - dy * size;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(hx, hy);
      ctx.stroke();
      // arrowhead
      const ah = size * 0.55;
      // perpendicular
      const px = -dy, py = dx;
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(hx - dx * ah + px * ah * 0.6, hy - dy * ah + py * ah * 0.6);
      ctx.lineTo(hx - dx * ah - px * ah * 0.6, hy - dy * ah - py * ah * 0.6);
      ctx.closePath();
      ctx.fill();
    }

    function draw() {
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = "#070d1a";
      ctx.fillRect(0, 0, CW, CH);

      const { lo, hi } = valueRange();
      const span = hi - lo;

      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          const { x, y, w, h } = cellRect(r, c);
          const type = grid[r][c];

          if (type === WALL) {
            ctx.fillStyle = "#3a3f55";
            ctx.fillRect(x, y, w, h);
          } else if (type === HAZARD) {
            ctx.fillStyle = "#e23b5a";
            ctx.fillRect(x, y, w, h);
            ctx.fillStyle = "rgba(7,13,26,0.9)";
            ctx.font = `${Math.round(h * 0.42)}px sans-serif`;
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("☠", x + w / 2, y + h / 2 + 1);
          } else if (type === GOAL) {
            ctx.fillStyle = "#1db954";
            ctx.fillRect(x, y, w, h);
            ctx.fillStyle = "rgba(7,13,26,0.92)";
            ctx.font = `${Math.round(h * 0.42)}px sans-serif`;
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("★", x + w / 2, y + h / 2 + 1);
          } else {
            // learnable cell: tint by normalised state value
            const v = maxQ(r, c);
            const t = span > 1e-9 ? clamp((v - lo) / span, 0, 1) : 0.0;
            ctx.fillStyle = heat(t);
            ctx.fillRect(x, y, w, h);
            // greedy best-action arrow (only once the cell has learned something)
            const q = Q[r][c];
            const learned = q[0] !== 0 || q[1] !== 0 || q[2] !== 0 || q[3] !== 0;
            if (learned) {
              drawArrow(x + w / 2, y + h / 2, argmaxQ(r, c), Math.min(w, h) * 0.22, "rgba(7,13,26,0.78)");
            }
          }

          // start marker
          if (r === start.r && c === start.c) {
            ctx.strokeStyle = "#ffd166";
            ctx.lineWidth = 3;
            ctx.strokeRect(x + 2.5, y + 2.5, w - 5, h - 5);
          }
        }
      }

      // grid lines
      ctx.strokeStyle = "rgba(255,255,255,0.10)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let c = 0; c <= COLS; c++) { ctx.moveTo((c * CW) / COLS, 0); ctx.lineTo((c * CW) / COLS, CH); }
      for (let r = 0; r <= ROWS; r++) { ctx.moveTo(0, (r * CH) / ROWS); ctx.lineTo(CW, (r * CH) / ROWS); }
      ctx.stroke();

      // agent token
      const ar = cellRect(agent.r, agent.c);
      ctx.font = `${Math.round(ar.h * 0.6)}px sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = "#fff";
      ctx.fillText("🤖", ar.x + ar.w / 2, ar.y + ar.h / 2 + 1);
    }

    function refreshStats() {
      epEl.textContent = episode;
      epsEl.textContent = epsilon.toFixed(2);
      abEl.textContent = `${alpha.toFixed(2)} / ${gamma.toFixed(3)}`;
      lastREl.textContent = episode > 0 ? lastEpReward.toFixed(2) : "—";
      if (recent.length) {
        const rate = (recent.reduce((s, v) => s + v, 0) / recent.length) * 100;
        successEl.textContent = `${rate.toFixed(0)}%`;
      } else successEl.textContent = "—";
    }

    function updateVerdict() {
      const { lo, hi } = valueRange();
      if (episode === 0) {
        verdict.innerHTML = "Episode 0: the table is all zeros, so every arrow is a tie and the agent moves at random. It needs to <i>stumble</i> onto the goal at least once before any value exists to learn from.";
        return;
      }
      const rate = recent.length ? (recent.reduce((s, v) => s + v, 0) / recent.length) * 100 : 0;
      if (hi - lo < 0.02) {
        verdict.innerHTML = `Episode <b>${episode}</b>: still mostly exploring (ε = <b>${epsilon.toFixed(2)}</b>). Value hasn't propagated far yet — the agent has to reach the green ★ to seed it.`;
      } else if (rate < 55) {
        verdict.innerHTML = `Episode <b>${episode}</b>: value is spreading <b>backward</b> from the goal — squares near ★ glow warm and their arrows already point the right way. Success rate so far: <b>${rate.toFixed(0)}%</b>.`;
      } else {
        verdict.innerHTML = `Episode <b>${episode}</b>: the arrows now form a route from the start to the goal and the agent reaches ★ about <b>${rate.toFixed(0)}%</b> of recent episodes. As ε falls it exploits this learned policy.`;
      }
    }

    function refresh() { draw(); refreshStats(); updateVerdict(); }

    // ---- animation loop: run several env steps per frame ----
    const loop = new Loop(() => {
      for (let i = 0; i < stepsPerFrame; i++) envStep();
      refresh();
    });

    // ---- events: controls ----
    trainBtn.addEventListener("click", () => {
      loop.toggle();
      trainBtn.textContent = loop.running ? "Pause ❚❚" : "Train ▸";
    });
    resetQBtn.addEventListener("click", () => {
      freshQ(); resetAgent();
      episode = 0; lastEpReward = 0; recent.length = 0;
      refresh();
    });
    resetWorldBtn.addEventListener("click", () => {
      loop.stop(); trainBtn.textContent = "Train ▸";
      defaultWorld(); freshQ(); resetAgent();
      episode = 0; lastEpReward = 0; recent.length = 0;
      refresh();
    });

    epsSlider.input.addEventListener("input", () => {
      epsilon = parseFloat(epsSlider.input.value);
      refreshStats();
    });
    alphaSlider.input.addEventListener("input", () => { alpha = parseFloat(alphaSlider.input.value); refreshStats(); });
    gammaSlider.input.addEventListener("input", () => { gamma = parseFloat(gammaSlider.input.value); refreshStats(); });
    speedSlider.input.addEventListener("input", () => { stepsPerFrame = parseInt(speedSlider.input.value, 10); });
    decayChk.addEventListener("change", () => { autoDecay = decayChk.checked; });

    // ---- events: painting cells ----
    function eventToCell(e) {
      const rect = canvas.getBoundingClientRect();
      let c = Math.floor(((e.clientX - rect.left) / rect.width) * COLS);
      let r = Math.floor(((e.clientY - rect.top) / rect.height) * ROWS);
      c = clamp(c, 0, COLS - 1);
      r = clamp(r, 0, ROWS - 1);
      return { r, c };
    }

    function paintCell(r, c) {
      const cur = grid[r][c];
      let changed = false;
      const isStart = (r === start.r && c === start.c);
      switch (paintTool) {
        case "wall":
          if (cur !== GOAL && !isStart && cur !== WALL) { grid[r][c] = WALL; changed = true; }
          break;
        case "hazard":
          if (cur !== GOAL && !isStart && cur !== HAZARD) { grid[r][c] = HAZARD; changed = true; }
          break;
        case "goal":
          if (!isStart) {
            grid[goal.r][goal.c] = EMPTY;     // clear old goal (only one allowed)
            goal = { r, c };
            grid[r][c] = GOAL;
            changed = true;
          }
          break;
        case "start":
          if (cur !== GOAL) {
            // only one start; new start cell becomes empty terrain
            if (grid[r][c] === WALL || grid[r][c] === HAZARD) grid[r][c] = EMPTY;
            start = { r, c };
            changed = true;
          }
          break;
        case "erase":
          if (cur === WALL || cur === HAZARD) { grid[r][c] = EMPTY; changed = true; }
          break;
      }
      if (changed) {
        // editing the world invalidates what was learned
        freshQ(); resetAgent();
        episode = 0; lastEpReward = 0; recent.length = 0;
      }
      return changed;
    }

    canvas.addEventListener("pointerdown", (e) => {
      painting = true;
      if (canvas.setPointerCapture && e.pointerId != null) {
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
      }
      const { r, c } = eventToCell(e);
      paintCell(r, c);
      refresh();
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!painting) return;
      const { r, c } = eventToCell(e);
      // goal/start are single-cell, do not drag-spam them
      if (paintTool === "goal" || paintTool === "start") return;
      paintCell(r, c);
      refresh();
    });
    const endPaint = (e) => {
      painting = false;
      if (canvas.releasePointerCapture && e && e.pointerId != null) {
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      }
    };
    canvas.addEventListener("pointerup", endPaint);
    canvas.addEventListener("pointercancel", endPaint);

    // ---- init ----
    defaultWorld();
    freshQ();
    resetAgent();
    refresh();

    return {
      onEnter() { draw(); refreshStats(); },
      onLeave() { loop.stop(); trainBtn.textContent = "Train ▸"; },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: learning to act from rewards alone</h3>
    <p><b>Reinforcement learning</b> is a different beast from the supervised games on this site. There is no dataset of correct answers. Instead an <em>agent</em> sits in an <em>environment</em>, takes <em>actions</em>, and receives a scalar <em>reward</em>. Its only goal: act so as to collect the most reward over the long run. Here the world is a grid; the agent must find the green goal (+1) while avoiding hazards (−1), paying a tiny living cost (−0.01) for every step so dawdling is discouraged.</p>
  </div>
  <div class="lesson-block">
    <h3>🔁 The RL loop you can watch</h3>
    <ol class="steps">
      <li><b>Observe state</b> <code>s</code> — which cell the agent is in.</li>
      <li><b>Choose an action</b> <code>a</code> — up/down/left/right, picked <em>ε-greedily</em> (see below).</li>
      <li><b>Act &amp; observe</b> the reward <code>r</code> and the next cell <code>s'</code>. A wall or edge bounces the agent back in place (and still costs a step).</li>
      <li><b>Learn</b> — nudge <code>Q(s,a)</code> toward the reward plus the best value reachable from <code>s'</code>.</li>
      <li><b>Repeat</b> until the agent lands on the goal or a hazard (episode ends), or it runs out of steps; then it teleports back to the start and tries again.</li>
    </ol>
    <p>Each tinted square shows its <b>value</b> <code>V(s)=max<sub>a</sub> Q(s,a)</code> on a cool→warm heat scale, and its arrow is the current greedy best action. Watch warmth — and correct arrows — creep <em>backward</em> from the goal.</p>
  </div>
  <div class="lesson-block">
    <h3>💡 What a Q-value means</h3>
    <p><code>Q(s,a)</code> is the agent's estimate of the <b>total future (discounted) reward</b> it will earn if it takes action <code>a</code> in state <code>s</code> and then behaves greedily forever after. Big Q = "this move leads somewhere good." The policy is simply: in each state, take the action with the largest Q.</p>
    <p>"Discounted" means a reward <code>k</code> steps away is worth <code>γ<sup>k</sup></code> of an immediate one. With γ near 1 the agent is <b>far-sighted</b> (it will happily take a long path to a big payoff); with small γ it grabs whatever is close and ignores the future.</p>
  </div>
  <div class="lesson-block">
    <h3>🧮 The temporal-difference update</h3>
    <p>This single line is the whole learning rule. After every step it pulls <code>Q(s,a)</code> a fraction <code>α</code> of the way toward a better estimate:</p>
    <div class="formula">Q(s,a) ← Q(s,a) + α [ r + γ · max<sub>a'</sub> Q(s',a') − Q(s,a) ]</div>
    <p>The bracket is the <b>TD error</b>: the gap between what we thought (<code>Q(s,a)</code>) and a fresher one-step estimate (the reward we just got plus the best we believe we can do from <code>s'</code>). If <code>s'</code> is terminal there is no future, so the target is just <code>r</code>. Because the goal hands out a real +1, that value flows into its neighbours, then their neighbours — this is why value <b>propagates backward</b> one ring per successful episode. <code>α</code> sets how fast (small = slow but stable, large = fast but jittery).</p>
  </div>
  <div class="lesson-block">
    <h3>⚖️ Exploration vs exploitation: ε-greedy</h3>
    <p>If the agent always took its current best action it might never discover a better route — or, before it has ever reached the goal, it would loop forever because every action looks equally worthless. So with probability <code>ε</code> it acts <b>randomly</b> (explore) and otherwise takes the greedy best (exploit). Early on you want high ε to find the goal; later you want low ε to actually follow the route you learned. Turn on <b>auto-decay ε</b> and watch it fall toward 0.05 as episodes pile up.</p>
  </div>
  <div class="lesson-block">
    <h3>⚠️ Failure modes</h3>
    <ul class="bullets">
      <li><b>Never finding the goal.</b> With ε too low and Q all zeros, the agent can wander aimlessly and never get the first +1 to learn from. Crank ε up, or shorten the path.</li>
      <li><b>α too high.</b> Big learning rates make Q bounce around and the arrows flicker instead of settling.</li>
      <li><b>γ too low.</b> A short-sighted agent won't "see" a distant goal through the −0.01 living costs and may give up.</li>
      <li><b>Tabular only.</b> We store one number per (cell, action). Real worlds have astronomically many states, so modern agents replace the table with a neural network — that's <em>Deep Q-Networks</em>, the same idea that learned to play Atari and underlies game-playing agents.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> Q-learning fills in a table of "how good is this move?" by repeatedly nudging each entry toward <code>reward + γ · (best value next door)</code>, so the goal's payoff seeps backward across the grid until the greedy arrows spell out a route.</p>
  </div>`,
};
