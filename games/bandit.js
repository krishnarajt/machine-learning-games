// =====================================================================
//  Multi-Armed Bandit — explore vs exploit, the one-state RL problem.
//
//  K slot machines, each with a hidden Bernoulli win-probability p_a.
//  A "pull" of arm a returns reward 1 with prob p_a, else 0. The agent
//  must decide, every pull, whether to EXPLOIT the arm that looks best
//  so far or EXPLORE another to learn more. We pit three classic
//  strategies against the same hidden world:
//
//    ε-greedy  : argmax empirical mean, but a random arm with prob ε.
//    UCB1      : argmax [ mean_a + c·sqrt( ln t / n_a ) ] — optimism.
//    Thompson  : sample θ_a ~ Beta(α_a, β_a), pick argmax θ_a.
//
//  The scoreboard is REGRET: how much reward we threw away versus an
//  oracle that always pulls the best arm. We draw the per-arm bars and
//  the regret-over-time curve on one plain canvas; stats go in panels.
// =====================================================================
import {
  el, button, ghostButton, slider, Loop,
  clamp, classColor, withAlpha, range, randRange, randInt, gaussRandom,
} from "../shared.js";

export default {
  key: "bandit",
  name: "Multi-Armed Bandit",
  icon: "🎰",
  category: "Reinforcement",
  tag: "RL",
  blurb: "Pull slot machines with hidden odds and learn which strategy finds the best one fastest.",
  eyebrow: "Reinforcement learning",
  title: 'Multi-Armed Bandit: <span class="grad">explore</span> vs <span class="grad">exploit</span>',
  lede:
    "You face a row of slot machines, each with unknown payout odds. Every pull is a chance to <em>exploit</em> the arm you think is best — " +
    "or <em>explore</em> a different one to learn more. That dilemma is the heart of reinforcement learning. " +
    "Pit ε-greedy, UCB and Thompson sampling against each other and watch the regret add up.",

  mount(root) {
    // ---- world geometry ----
    const K = 5;                 // number of arms
    const CW = 760, CH = 460;
    const ctxFont = "'JetBrains Mono', monospace";

    // ---- hyperparameters ----
    let strategy = "egreedy";    // egreedy | ucb | thompson
    let epsilon = 0.1;           // ε-greedy exploration probability
    let ucbC = Math.SQRT2;       // UCB exploration constant (~1.414)
    let pullsPerSec = 12;        // auto-run speed

    // ---- learning / world state ----
    let trueP = [];              // hidden Bernoulli probs, one per arm
    let bestArm = 0;             // index of the genuinely best arm
    let bestP = 0;               // its win probability (= p*)

    let n = [];                  // pulls per arm
    let succ = [];               // successes per arm
    let t = 0;                   // total pulls so far
    let totalReward = 0;
    let cumRegret = 0;
    let lastArm = -1;            // arm picked on the most recent pull
    let lastReward = 0;
    const regretHist = [];       // [t, cumRegret] sampled points for the curve

    // ---- DOM: canvas ----
    const canvas = el("canvas", {
      width: CW, height: CH, role: "img",
      "aria-label": "Bar chart of the estimated win-rate of each slot-machine arm, with the true rate marked, plus a cumulative-regret curve over time.",
    });
    const ctx = canvas.getContext("2d");

    // ---- DOM: controls ----
    const pullBtn = button("Pull once");
    const runBtn = button("Auto-run ▸");
    const resetStatsBtn = ghostButton("Reset stats");
    const resetWorldBtn = ghostButton("New machines");

    const epsSlider = slider({ min: 0.01, max: 0.5, step: 0.01, value: epsilon, label: "ε explore", format: (v) => v.toFixed(2) });
    const cSlider = slider({ min: 0, max: 3, step: 0.05, value: ucbC, label: "c (UCB)", format: (v) => v.toFixed(2) });
    const speedSlider = slider({ min: 1, max: 60, step: 1, value: pullsPerSec, label: "speed", format: (v) => `${v}/s` });

    // strategy segmented control
    const stratDefs = [
      ["egreedy", "ε-greedy"],
      ["ucb", "UCB1"],
      ["thompson", "Thompson"],
    ];
    const stratButtons = {};
    const segGroup = el("div", { class: "seg-group" });
    for (const [k, label] of stratDefs) {
      const b = el("button", { class: k === strategy ? "active" : "" }, label);
      b.addEventListener("click", () => {
        if (strategy === k) return;
        strategy = k;
        for (const kk of Object.keys(stratButtons)) stratButtons[kk].classList.toggle("active", kk === k);
        // changing strategy resets stats so the comparison is fair (probs kept)
        loop.stop(); runBtn.textContent = "Auto-run ▸";
        resetStats();
        refresh();
      });
      stratButtons[k] = b;
      segGroup.append(b);
    }

    // ---- DOM: stats ----
    const pullsEl = el("span", { class: "v" }, "0");
    const rewardEl = el("span", { class: "v" }, "0");
    const avgEl = el("span", { class: "v" }, "—");
    const regretEl = el("span", { class: "v" }, "0.00");
    const foundEl = el("span", { class: "v" }, "—");

    const legend = el("div", { class: "legend" });
    const verdict = el("p", { class: "verdict" },
      "Pick a strategy and press ", el("b", {}, "Pull once"), " or ", el("b", {}, "Auto-run"),
      ". Each tall bar is an arm's estimated win-rate; the faint notch is its hidden true rate. Watch the regret curve bend flat as the agent locks onto the best machine.");

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "canvas-stage" }, canvas),
          el("p", { class: "caption" },
            "Top: each arm's estimated win-rate (bar) vs its hidden true rate (notch); the highlighted bar was just pulled. Bottom: cumulative regret — reward lost versus an oracle that always plays the best arm.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel" },
            el("h3", {}, "Scoreboard"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Total pulls"), pullsEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Total reward"), rewardEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Average reward"), avgEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Cumulative regret"), regretEl),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Best arm found?"), foundEl)),
          el("div", { class: "panel" },
            el("h3", {}, "Strategy"),
            el("div", { class: "controls" }, segGroup),
            el("div", { class: "controls" }, epsSlider.wrap),
            el("div", { class: "controls" }, cSlider.wrap)),
          el("div", { class: "panel" },
            el("h3", {}, "Controls"),
            el("div", { class: "controls" }, pullBtn, runBtn),
            el("div", { class: "controls" }, resetStatsBtn, resetWorldBtn),
            el("div", { class: "controls" }, speedSlider.wrap)),
          legend)),
      verdict
    );

    // ---- world setup ----
    function newWorld() {
      trueP = range(K).map(() => randRange(0.1, 0.9));
      bestP = -1; bestArm = 0;
      for (let a = 0; a < K; a++) if (trueP[a] > bestP) { bestP = trueP[a]; bestArm = a; }
      resetStats();
    }

    function resetStats() {
      n = range(K).map(() => 0);
      succ = range(K).map(() => 0);
      t = 0;
      totalReward = 0;
      cumRegret = 0;
      lastArm = -1;
      lastReward = 0;
      regretHist.length = 0;
      regretHist.push([0, 0]);
    }

    // empirical mean reward of an arm (0 if never pulled)
    function emp(a) { return n[a] ? succ[a] / n[a] : 0; }

    // current believed-best arm (argmax empirical mean; ties -> lowest index)
    function argmaxEmp() {
      let best = 0, bv = -Infinity;
      for (let a = 0; a < K; a++) {
        const m = emp(a);
        if (m > bv) { bv = m; best = a; }
      }
      return best;
    }

    // ---- Beta sampler for Thompson, via two Gamma samples ----
    // Gamma(k,1) sampler — Marsaglia & Tsang for k>=1, with a boost for k<1.
    function gammaSample(k) {
      if (k < 1) {
        // boost: Gamma(k) = Gamma(k+1) * U^(1/k)
        const u = Math.random();
        return gammaSample(k + 1) * Math.pow(u || 1e-12, 1 / k);
      }
      const d = k - 1 / 3;
      const c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do {
          x = gaussRandom(0, 1);
          v = 1 + c * x;
        } while (v <= 0);
        v = v * v * v;
        const u = Math.random();
        const x2 = x * x;
        if (u < 1 - 0.0331 * x2 * x2) return d * v;
        if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) return d * v;
      }
    }
    // Beta(a,b) = X / (X + Y), X~Gamma(a), Y~Gamma(b)
    function betaSample(a, b) {
      const x = gammaSample(a);
      const y = gammaSample(b);
      const s = x + y;
      return s > 0 ? x / s : 0.5;
    }
    // Beta posterior params for an arm: α = 1 + successes, β = 1 + failures
    function alphaOf(a) { return 1 + succ[a]; }
    function betaOf(a) { return 1 + (n[a] - succ[a]); }
    // Beta pdf (unnormalised shape is enough for drawing the density curve)
    function betaPdf(x, a, b) {
      if (x <= 0 || x >= 1) return 0;
      // log B(a,b) via lgamma for numerical stability
      const lnB = lgamma(a) + lgamma(b) - lgamma(a + b);
      const lp = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lnB;
      return Math.exp(lp);
    }
    // Lanczos approximation of ln Γ(z)
    function lgamma(z) {
      const g = 7;
      const c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
      ];
      if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
      z -= 1;
      let a = c[0];
      const tt = z + g + 0.5;
      for (let i = 1; i < g + 2; i++) a += c[i] / (z + i);
      return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(tt) - tt + Math.log(a);
    }

    // ---- action selection: which arm does the current strategy pull? ----
    function chooseArm() {
      if (strategy === "egreedy") {
        if (Math.random() < epsilon) return randInt(K);
        return argmaxEmp();
      }
      if (strategy === "ucb") {
        // pull any never-tried arm first (treat its bonus as +∞)
        for (let a = 0; a < K; a++) if (n[a] === 0) return a;
        let best = 0, bv = -Infinity;
        const lnt = Math.log(t);
        for (let a = 0; a < K; a++) {
          const bonus = ucbC * Math.sqrt(lnt / n[a]);
          const score = emp(a) + bonus;
          if (score > bv) { bv = score; best = a; }
        }
        return best;
      }
      // thompson
      let best = 0, bv = -Infinity;
      for (let a = 0; a < K; a++) {
        const theta = betaSample(alphaOf(a), betaOf(a));
        if (theta > bv) { bv = theta; best = a; }
      }
      return best;
    }

    // ---- one pull of the chosen arm ----
    function pull() {
      const a = chooseArm();
      const reward = Math.random() < trueP[a] ? 1 : 0;
      n[a]++;
      succ[a] += reward;
      t++;
      totalReward += reward;
      // regret of THIS pull = p* − p_a (expected reward gap to the oracle)
      cumRegret += bestP - trueP[a];
      lastArm = a;
      lastReward = reward;
      // keep the regret curve from growing unbounded in memory
      if (regretHist.length === 0 || t - regretHist[regretHist.length - 1][0] >= Math.max(1, Math.floor(t / 600) + 1)) {
        regretHist.push([t, cumRegret]);
      } else {
        regretHist[regretHist.length - 1] = [t, cumRegret];
      }
    }

    // ---- drawing ----
    // layout: arms occupy the top region, regret curve the bottom region.
    const PAD = 16;
    const TOP_H = 250;                       // height of the arm-bars region
    const BOT_Y = TOP_H + 28;                // top of the regret region
    const BOT_H = CH - BOT_Y - PAD;

    function drawArms() {
      const x0 = PAD + 32;                   // leave room for the left axis labels
      const x1 = CW - PAD;
      const w = x1 - x0;
      const slot = w / K;
      const barW = slot * 0.56;
      const baseY = PAD + TOP_H;             // y=0 (empty) baseline
      const topY = PAD + 18;                 // y=1 (certain win) line
      const h = baseY - topY;
      const believed = t > 0 ? argmaxEmp() : -1;

      // y-axis gridlines at 0, 0.25, 0.5, 0.75, 1
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      ctx.font = `11px ${ctxFont}`;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let g = 0; g <= 4; g++) {
        const frac = g / 4;
        const y = baseY - frac * h;
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        ctx.fillStyle = "rgba(234,238,251,0.45)";
        ctx.fillText(frac.toFixed(2), x0 - 6, y);
      }

      for (let a = 0; a < K; a++) {
        const cx = x0 + slot * (a + 0.5);
        const col = classColor(a);
        const mean = emp(a);
        const bx = cx - barW / 2;

        // thompson: sketch the Beta posterior density behind the bar
        if (strategy === "thompson") {
          const A = alphaOf(a), B = betaOf(a);
          // find a peak value to normalise the density height to the column
          let peak = 1e-9;
          const SAMP = 48;
          const dens = [];
          for (let i = 0; i <= SAMP; i++) {
            const p = i / SAMP;
            const d = betaPdf(p, A, B);
            dens.push(d);
            if (d > peak) peak = d;
          }
          ctx.beginPath();
          for (let i = 0; i <= SAMP; i++) {
            const p = i / SAMP;
            const yy = baseY - (dens[i] / peak) * h * 0.9;
            const xx = cx - barW / 2 + p * barW;
            if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
          }
          ctx.lineTo(cx + barW / 2, baseY);
          ctx.lineTo(cx - barW / 2, baseY);
          ctx.closePath();
          ctx.fillStyle = withAlpha(col, 0.16);
          ctx.fill();
          ctx.strokeStyle = withAlpha(col, 0.6);
          ctx.lineWidth = 1.5;
          ctx.stroke();
        } else {
          // empirical-mean bar
          const barH = mean * h;
          ctx.fillStyle = a === lastArm ? col : withAlpha(col, 0.55);
          ctx.fillRect(bx, baseY - barH, barW, barH);
          if (a === lastArm) {
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.strokeRect(bx, baseY - barH, barW, barH);
          }
        }

        // true probability notch (faint dashed line across the column)
        const ty = baseY - trueP[a] * h;
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(bx - 4, ty); ctx.lineTo(cx + barW / 2 + 4, ty);
        ctx.stroke();
        ctx.restore();

        // crown on the genuinely best arm
        if (a === bestArm) {
          ctx.font = `16px sans-serif`;
          ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
          ctx.fillText("👑", cx, topY - 2);
        }

        // arm label + pull count below the baseline
        ctx.fillStyle = a === lastArm ? "#fff" : "rgba(234,238,251,0.75)";
        ctx.font = `12px ${ctxFont}`;
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText(`#${a + 1}`, cx, baseY + 6);
        ctx.fillStyle = "rgba(234,238,251,0.55)";
        ctx.font = `11px ${ctxFont}`;
        ctx.fillText(`n=${n[a]}`, cx, baseY + 22);

        // a small ring under the believed-best (argmax-mean) arm
        if (a === believed) {
          ctx.strokeStyle = withAlpha(col, 0.9);
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cx, baseY + 44, 5, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // region caption
      ctx.fillStyle = "rgba(234,238,251,0.5)";
      ctx.font = `11px ${ctxFont}`;
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(strategy === "thompson" ? "Beta posterior per arm  ·  --- true rate  ·  👑 best  ·  ○ believed-best"
        : "estimated win-rate  ·  --- true rate  ·  👑 best  ·  ○ believed-best", x0, PAD);
    }

    function drawRegret() {
      const x0 = PAD + 40;
      const x1 = CW - PAD;
      const y0 = BOT_Y;
      const y1 = BOT_Y + BOT_H;
      const w = x1 - x0, h = y1 - y0;

      // frame
      ctx.strokeStyle = "rgba(255,255,255,0.10)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, w, h);

      const maxT = Math.max(10, t);
      const maxR = Math.max(1, cumRegret * 1.1);

      // linear-regret reference line (constant-rate exploration baseline):
      // a fixed per-pull gap would grow as a straight diagonal.
      ctx.strokeStyle = "rgba(255,255,255,0.14)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x0, y1);
      ctx.lineTo(x1, y1 - clamp(cumRegret / maxR, 0, 1) * h);
      ctx.stroke();
      ctx.setLineDash([]);

      // the actual regret curve
      if (regretHist.length >= 2) {
        ctx.strokeStyle = "#ff7a59";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        regretHist.forEach(([tt, rr], i) => {
          const px = x0 + (tt / maxT) * w;
          const py = y1 - clamp(rr / maxR, 0, 1) * h;
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // axis labels
      ctx.fillStyle = "rgba(234,238,251,0.55)";
      ctx.font = `11px ${ctxFont}`;
      ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText("regret", x0, y0 - 4);
      ctx.textAlign = "right"; ctx.textBaseline = "top";
      ctx.fillText(`${maxR.toFixed(1)}`, x0 - 4, y0);
      ctx.fillText(`0`, x0 - 4, y1 - 12);
      ctx.textAlign = "right"; ctx.textBaseline = "top";
      ctx.fillText(`t = ${t}`, x1, y1 + 2);
      ctx.textAlign = "left";
      ctx.fillText("dashed = linear growth (no learning)", x0, y1 + 2);
    }

    function draw() {
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = "#070d1a";
      ctx.fillRect(0, 0, CW, CH);
      drawArms();
      drawRegret();
    }

    function refreshStats() {
      pullsEl.textContent = t;
      rewardEl.textContent = totalReward;
      avgEl.textContent = t ? (totalReward / t).toFixed(3) : "—";
      regretEl.textContent = cumRegret.toFixed(2);
      if (t === 0) { foundEl.textContent = "—"; return; }
      const believed = argmaxEmp();
      foundEl.textContent = believed === bestArm ? `yes ✓ (arm #${bestArm + 1})` : `no — likes #${believed + 1}`;
    }

    function refreshLegend() {
      legend.replaceChildren();
      for (let a = 0; a < K; a++) {
        const tag = a === bestArm ? " 👑" : "";
        legend.append(el("span", { class: "legend-item" },
          el("span", { class: "legend-dot", style: { background: classColor(a) } }),
          `Arm #${a + 1} · p=${trueP[a].toFixed(2)}${tag}`));
      }
    }

    function updateVerdict() {
      if (t === 0) {
        verdict.innerHTML = "Nothing pulled yet. The agent has <b>no idea</b> which of the five machines pays best — every estimate is zero. Start pulling to find out.";
        return;
      }
      const avg = totalReward / t;
      const believed = argmaxEmp();
      const onBest = believed === bestArm;
      const regretPerPull = cumRegret / t;
      const stratName = strategy === "egreedy" ? "ε-greedy" : strategy === "ucb" ? "UCB1" : "Thompson sampling";
      if (!onBest) {
        verdict.innerHTML = `<b>${stratName}</b> after <b>${t}</b> pulls currently believes arm <b>#${believed + 1}</b> is best, but the true winner is arm <b>#${bestArm + 1}</b> (p=${bestP.toFixed(2)}). ` +
          `Average reward <b>${avg.toFixed(3)}</b>, regret climbing at <b>${regretPerPull.toFixed(3)}</b>/pull. It needs more evidence.`;
      } else {
        verdict.innerHTML = `<b>${stratName}</b> has the right arm: it favours <b>#${bestArm + 1}</b> (p=${bestP.toFixed(2)}). ` +
          `Average reward <b>${avg.toFixed(3)}</b> (oracle ceiling ${bestP.toFixed(2)}). Regret per pull has fallen to <b>${regretPerPull.toFixed(3)}</b> — the curve is bending flat as it stops wasting pulls on losers.`;
      }
    }

    function refresh() {
      draw();
      refreshStats();
      refreshLegend();
      updateVerdict();
    }

    // ---- animation loop: several pulls per second, accumulator-paced ----
    let acc = 0;
    const loop = new Loop((dt) => {
      acc += dt;
      const interval = 1 / pullsPerSec;
      let did = false;
      // catch up but cap to avoid runaway bursts on a slow frame
      let budget = 8;
      while (acc >= interval && budget-- > 0) {
        acc -= interval;
        pull();
        did = true;
      }
      if (did) refresh(); else draw();
    });

    // ---- events ----
    pullBtn.addEventListener("click", () => {
      if (loop.running) { loop.stop(); runBtn.textContent = "Auto-run ▸"; }
      pull();
      refresh();
    });
    runBtn.addEventListener("click", () => {
      loop.toggle();
      runBtn.textContent = loop.running ? "Pause ❚❚" : "Auto-run ▸";
    });
    resetStatsBtn.addEventListener("click", () => {
      loop.stop(); runBtn.textContent = "Auto-run ▸";
      resetStats();
      refresh();
    });
    resetWorldBtn.addEventListener("click", () => {
      loop.stop(); runBtn.textContent = "Auto-run ▸";
      newWorld();
      refresh();
    });

    epsSlider.input.addEventListener("input", () => { epsilon = parseFloat(epsSlider.input.value); });
    cSlider.input.addEventListener("input", () => { ucbC = parseFloat(cSlider.input.value); });
    speedSlider.input.addEventListener("input", () => { pullsPerSec = parseInt(speedSlider.input.value, 10); });

    // ---- init ----
    newWorld();
    refresh();

    return {
      onEnter() { draw(); refreshStats(); },
      onLeave() { loop.stop(); runBtn.textContent = "Auto-run ▸"; },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>🎯 The job: the explore / exploit dilemma</h3>
    <p>A <b>multi-armed bandit</b> is the cleanest possible reinforcement-learning problem. You face <em>K</em> slot machines ("arms"), each paying out with a hidden probability you don't know. On every pull you face one decision: <b>exploit</b> the arm that has looked best so far, or <b>explore</b> a different arm to gather more information about it. Exploit too soon and you may get stuck on a mediocre machine; explore too much and you waste pulls on losers. Every adaptive system — A/B tests, ad selection, recommended articles, clinical treatment assignment — lives on this knife-edge.</p>
  </div>
  <div class="lesson-block">
    <h3>🔁 What you can see on screen</h3>
    <ol class="steps">
      <li><b>The world.</b> Five arms, each with a hidden true win-rate <code>p<sub>a</sub></code> drawn at random. A pull returns reward 1 with probability <code>p<sub>a</sub></code>, else 0 — a Bernoulli trial. The 👑 marks the genuinely best arm (you, the observer, get to see it; the agent does not).</li>
      <li><b>The bars.</b> Each bar is the arm's <em>estimated</em> win-rate (successes ÷ pulls). The faint dashed notch is its <em>true</em> rate — the bar should creep toward it as <code>n<sub>a</sub></code> grows. The highlighted bar is the arm just pulled; the small ring marks the arm the agent currently believes is best.</li>
      <li><b>The regret curve.</b> The orange line at the bottom is cumulative regret over time. A good strategy makes it <em>bend flat</em>; a bad one keeps climbing in a straight line.</li>
    </ol>
  </div>
  <div class="lesson-block">
    <h3>😵 Why pure greedy gets stuck</h3>
    <p>The naive idea — "always pull whichever arm has the highest empirical mean so far" — fails badly. Suppose a genuinely bad arm gets lucky on its first pull (reward 1, so estimate 1.0). Pure greedy will pull it forever and <em>never sample the others enough</em> to discover they are better. With no built-in exploration, a single lucky fluke can trap the agent on a loser. Every strategy below is a different cure for this.</p>
  </div>
  <div class="lesson-block">
    <h3>🎲 ε-greedy: a dash of forced randomness</h3>
    <p>The simplest fix. With probability <code>ε</code> pull a <b>uniformly random</b> arm (explore); otherwise pull the empirical best (exploit).</p>
    <div class="formula">a<sub>t</sub> = argmax<sub>a</sub> &mu;&#770;<sub>a</sub> &nbsp;with prob 1&minus;&epsilon;,&nbsp; random arm with prob &epsilon;</div>
    <p>It's robust and trivial to code, but the exploration never stops: even once it knows the best arm, it keeps throwing away an <code>ε</code> fraction of pulls on random arms, so its regret grows <b>linearly</b> forever (the dashed reference line). Decaying <code>ε</code> over time fixes that, at the cost of a tuning knob.</p>
  </div>
  <div class="lesson-block">
    <h3>🔭 UCB1: optimism under uncertainty</h3>
    <p>Instead of acting on the mean, act on an <em>optimistic upper bound</em> for each arm. Add a confidence bonus that is large when an arm has been pulled rarely and shrinks as evidence accumulates:</p>
    <div class="formula">a<sub>t</sub> = argmax<sub>a</sub> [ &mu;&#770;<sub>a</sub> + c &middot; &radic;( ln t / n<sub>a</sub> ) ]</div>
    <p>The bonus term goes to infinity for an unpulled arm (so every arm is tried once), and decays like <code>1/&radic;n<sub>a</sub></code> as the arm is sampled. "Optimism in the face of uncertainty": an arm is worth pulling either because it <em>looks</em> good or because we're <em>unsure</em> about it. There's no randomness at all — it is fully deterministic — and its regret grows only <b>logarithmically</b>, which is provably near-optimal. The constant <code>c</code> tunes how aggressively it explores.</p>
  </div>
  <div class="lesson-block">
    <h3>🧪 Thompson sampling: act on a sample from your beliefs</h3>
    <p>The Bayesian approach. Keep a posterior distribution over each arm's true rate. For a Bernoulli arm with <code>s</code> successes and <code>f</code> failures, that posterior is a <b>Beta</b> distribution:</p>
    <div class="formula">p<sub>a</sub> ~ Beta(&alpha;<sub>a</sub>, &beta;<sub>a</sub>),&nbsp; &alpha;<sub>a</sub> = 1 + successes,&nbsp; &beta;<sub>a</sub> = 1 + failures</div>
    <p>To choose, draw one sample <code>&theta;<sub>a</sub></code> from each arm's Beta and pull the arm with the largest sample — then update that arm's posterior with the result. Switch the strategy to Thompson above and the bars become the actual posterior densities: rarely-pulled arms have wide, flat curves (high uncertainty → frequently sampled high → explored), while well-known arms have tall narrow spikes. This "probability matching" explores automatically, needs no tuning knob, and is often the strongest of the three in practice. We sample Beta here from two Gamma draws (Marsaglia–Tsang).</p>
  </div>
  <div class="lesson-block">
    <h3>📉 Regret: the scoreboard</h3>
    <p>You can't score a bandit by raw reward — that depends on how generous the hidden machines happen to be. The honest yardstick is <b>regret</b>: how much reward you lost versus an oracle that always pulls the best arm <code>p* = max<sub>a</sub> p<sub>a</sub></code>.</p>
    <div class="formula">regret<sub>t</sub> = t &middot; p* &minus; &Sigma; (rewards collected)</div>
    <p>A learning strategy makes the regret curve <b>concave</b> — it climbs fast while exploring, then bends toward flat once it commits to the best arm. A non-learning strategy (or constant-ε) keeps adding regret at a steady rate, so its curve is a straight diagonal. Flatter is better.</p>
  </div>
  <div class="lesson-block">
    <h3>🔗 A bandit is a one-state RL problem</h3>
    <p>This is reinforcement learning stripped to its core. Unlike <b>Q-Learning</b>, where the agent moves between states and an action changes where you end up, a bandit has <em>no state transitions</em>: every pull faces the identical situation, and the only thing you learn is the value of each action. That makes it the perfect place to study explore-vs-exploit in isolation — the same dilemma <b>Q-Learning</b> handles with its ε-greedy action choice, just without the complication of a changing world.</p>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> a bandit must balance trying new arms against milking the best-known one; ε-greedy explores at random, UCB explores by being optimistic about uncertain arms, Thompson explores by sampling from its Bayesian beliefs — and you grade them all by how flat they bend the regret curve.</p>
  </div>`,
};
