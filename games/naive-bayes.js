// =====================================================================
//  Naive Bayes — a spam filter doing maths.
//
//  Train multinomial Naive Bayes on a tiny hardcoded corpus of spam/ham
//  messages, then classify whatever the user types — live, word by word.
//
//  Training:  count word occurrences per class; priors from message counts;
//             P(w|c) = (count + 1) / (total_words_in_c + V)   [Laplace +1]
//  Classify:  score_c = log P(c) + Σ log P(w|c)  (work in LOG space)
//             P(spam|msg) = 1 / (1 + exp(score_ham − score_spam))  [softmax of 2]
// =====================================================================
import { el, ghostButton, clamp } from "../shared.js";

// ---- training corpus (hardcoded) ------------------------------------
const SPAM = [
  "win free money now",
  "claim your prize winner",
  "cheap meds discount online",
  "urgent verify your account click link",
  "you won a free gift card",
  "limited offer act now buy cheap",
  "congratulations you are a lucky winner",
  "double your income work from home",
  "free trial click here to claim",
  "lowest price guaranteed order now",
  "exclusive deal money back winner",
  "your loan has been approved click now",
];
const HAM = [
  "are we still on for lunch",
  "the meeting moved to 3pm tomorrow",
  "can you review my draft please",
  "thanks for dinner last night",
  "i will send the report by friday",
  "see you at the office in the morning",
  "happy birthday hope you have a great day",
  "let me know if you need anything",
  "the project deadline is next week",
  "could you call me back later today",
  "great work on the presentation team",
  "lets grab coffee after the standup",
];

export default {
  key: "naive-bayes",
  name: "Naive Bayes",
  icon: "📨",
  category: "Classification",
  tag: "Bayes",
  blurb: "Type a message and watch Bayes' rule weigh each word as spam or not.",
  eyebrow: "Classification · Supervised",
  title: 'Naive Bayes: a <span class="grad">spam filter</span> doing maths',
  lede:
    "This is how classic spam filters think. Every word carries evidence — <em>free</em> and <em>winner</em> lean spam, " +
    "<em>meeting</em> and <em>lunch</em> lean legit — and Naive Bayes multiplies all that evidence together using Bayes' rule. " +
    "Type a message and watch the probability swing word by word.",

  mount(root) {
    // ---- 1. TRAIN the model (runs once on mount) --------------------
    const tokenize = (s) =>
      s.toLowerCase().split(/[^a-z]+/).filter((w) => w.length > 0);

    function train(spamDocs, hamDocs) {
      const counts = { spam: new Map(), ham: new Map() };
      const totalWords = { spam: 0, ham: 0 };
      const vocab = new Set();

      const ingest = (docs, cls) => {
        for (const doc of docs) {
          for (const w of tokenize(doc)) {
            counts[cls].set(w, (counts[cls].get(w) || 0) + 1);
            totalWords[cls] += 1;
            vocab.add(w);
          }
        }
      };
      ingest(spamDocs, "spam");
      ingest(hamDocs, "ham");

      const V = vocab.size;
      const nDocs = spamDocs.length + hamDocs.length;
      const logPrior = {
        spam: Math.log(spamDocs.length / nDocs),
        ham: Math.log(hamDocs.length / nDocs),
      };

      // Laplace add-one: P(w|c) = (count_c(w)+1) / (total_c + V)
      const logLik = (w, cls) =>
        Math.log(((counts[cls].get(w) || 0) + 1) / (totalWords[cls] + V));

      return { counts, totalWords, vocab, V, nDocs, logPrior, logLik, spamN: spamDocs.length, hamN: hamDocs.length };
    }

    const model = train(SPAM, HAM);

    // log-likelihood ratio of a single word: log P(w|spam) − log P(w|ham)
    const wordLLR = (w) => model.logLik(w, "spam") - model.logLik(w, "ham");

    // ---- 2. DOM -----------------------------------------------------
    const textarea = el("textarea", {
      rows: 3,
      spellcheck: "false",
      placeholder: "Type a message… e.g. 'win a free prize now'",
      style: {
        width: "100%",
        resize: "vertical",
        background: "rgba(8,12,26,0.7)",
        border: "1px solid var(--line-strong)",
        color: "var(--text)",
        borderRadius: "var(--r-sm)",
        padding: "0.7rem 0.85rem",
        font: "0.95rem 'JetBrains Mono', monospace",
        lineHeight: "1.5",
      },
    });

    // big P(spam) readout + probability bar
    const probPct = el("div", {
      style: { font: "700 2.4rem 'Space Grotesk', sans-serif", lineHeight: "1", letterSpacing: "-0.02em" },
    }, "50%");
    const probSub = el("div", { class: "stat-label" }, "P(spam | message)");
    const probFill = el("div", { class: "bar-fill warm", style: { width: "50%" } });
    const probBar = el("div", { class: "bar-track", style: { height: "18px", marginTop: "0.7rem" } }, probFill);

    const tokensWrap = el("div", { class: "tokens" });
    const tokensEmpty = el("p", { class: "caption" }, "Tokens appear here, tinted by whether each word leans spam or ham.");

    const barsWrap = el("div", { class: "bars" });

    const kvPrior = el("span", { class: "v" }, "—");
    const kvEvidence = el("span", { class: "v" }, "—");
    const kvPost = el("span", { class: "v" }, "—");
    const kvDecision = el("span", { class: "v" }, "—");

    const verdict = el("p", { class: "verdict" },
      "Type a message above (or hit an example) and watch Bayes' rule weigh each word.");

    // preset example buttons
    const presets = [
      ["🚩 free money now", "win free money now"],
      ["🚩 prize winner", "congratulations you are a lucky winner claim your prize"],
      ["🚩 cheap meds", "cheap meds discount click here to order now"],
      ["✅ lunch?", "are we still on for lunch today"],
      ["✅ meeting", "the meeting moved to 3pm can you review my draft"],
      ["✅ thanks", "thanks for dinner last night see you tomorrow"],
    ];
    const presetBtns = presets.map(([label, text]) =>
      ghostButton(label, { onClick: () => { textarea.value = text; classify(); textarea.focus(); } }));

    root.append(
      el("div", { class: "game-grid" },
        el("div", {},
          el("div", { class: "panel" },
            el("h3", {}, "Your message"),
            textarea,
            el("div", { class: "controls", style: { marginTop: "0.8rem" } }, ...presetBtns)),
          el("div", { class: "panel", style: { marginTop: "0.9rem" } },
            el("h3", {}, "Tokens — evidence per word"),
            tokensWrap,
            tokensEmpty),
          el("p", { class: "caption" },
            "Each chip is one word. Coral = leans spam, blue = leans ham, grey = unseen / neutral.")),
        el("div", { class: "game-side" },
          el("div", { class: "panel", style: { textAlign: "center" } },
            el("h3", { style: { textAlign: "left" } }, "Probability it's spam"),
            probPct, probSub, probBar),
          el("div", { class: "panel" },
            el("h3", {}, "Bayes' rule, step by step"),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Prior P(spam)"), kvPrior),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Evidence (log-odds)"), kvEvidence),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Posterior P(spam)"), kvPost),
            el("div", { class: "kv" }, el("span", { class: "k" }, "Decision"), kvDecision)),
          el("div", { class: "panel" },
            el("h3", {}, "Top contributing words"),
            barsWrap,
            el("p", { class: "caption" }, "Bars show log P(w|spam) ⁄ P(w|ham). Right = spammy, left = hammy.")),
          el("div", { class: "legend" },
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: "#ff7a59" } }), "leans spam"),
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: "#5ad2ff" } }), "leans ham"),
            el("span", { class: "legend-item" }, el("span", { class: "legend-dot sq", style: { background: "#7c89b8" } }), "unseen / neutral")))),
      verdict
    );

    // ---- 3. CLASSIFY (the live inference) ---------------------------
    function classify() {
      const raw = textarea.value;
      const words = tokenize(raw);

      // tokens panel
      tokensWrap.replaceChildren();
      tokensEmpty.style.display = words.length ? "none" : "";
      for (const w of words) {
        const llr = wordLLR(w);
        const known = model.vocab.has(w);
        // class chip by sign of the log-likelihood-ratio
        let cls = "";
        if (known && Math.abs(llr) > 1e-6) cls = llr > 0 ? "spam" : "ham";
        tokensWrap.append(el("span", { class: "token" + (cls ? " " + cls : "") }, w));
      }

      // scores in LOG space:  score_c = log P(c) + Σ log P(w|c)
      // Out-of-vocab words are INCLUDED via the +1 smoothing term: for an
      // unseen word, count=0 in both classes, so each contributes
      // log(1/(total_c+V)) to that class. Its log-ratio is small but nonzero
      // (because the two classes have different totals), so OOV words barely
      // move the verdict — exactly what we want.
      let scoreSpam = model.logPrior.spam;
      let scoreHam = model.logPrior.ham;
      for (const w of words) {
        scoreSpam += model.logLik(w, "spam");
        scoreHam += model.logLik(w, "ham");
      }

      // posterior via softmax over the two scores (numerically stable form)
      const logOdds = scoreSpam - scoreHam;          // Σ evidence + prior log-odds
      const pSpam = 1 / (1 + Math.exp(-logOdds));     // = 1/(1+exp(score_ham−score_spam))
      const pct = Math.round(pSpam * 100);

      // big readout + bar
      probPct.textContent = pct + "%";
      probFill.style.width = clamp(pct, 0, 100) + "%";

      // kv rows
      const priorSpam = model.spamN / model.nDocs;
      const evidenceOnly = logOdds - (model.logPrior.spam - model.logPrior.ham); // Σ log-ratio of words
      kvPrior.textContent = priorSpam.toFixed(2);
      kvEvidence.textContent = (words.length ? (evidenceOnly >= 0 ? "+" : "") + evidenceOnly.toFixed(2) : "0.00");
      kvPost.textContent = pSpam.toFixed(3);
      const decision = pSpam >= 0.5 ? "SPAM 🚩" : "HAM ✅";
      kvDecision.textContent = decision;

      // top contributing words (largest |LLR|, known words only), de-duped
      const seen = new Set();
      const uniq = [];
      for (const w of words) {
        if (model.vocab.has(w) && !seen.has(w)) { seen.add(w); uniq.push(w); }
      }
      const ranked = uniq
        .map((w) => ({ w, llr: wordLLR(w) }))
        .filter((d) => Math.abs(d.llr) > 1e-6)
        .sort((a, b) => Math.abs(b.llr) - Math.abs(a.llr))
        .slice(0, 6);

      barsWrap.replaceChildren();
      if (!ranked.length) {
        barsWrap.append(el("p", { class: "caption" },
          words.length ? "No training words recognised yet — only the prior is talking." : "Waiting for input…"));
      } else {
        const maxAbs = Math.max(...ranked.map((d) => Math.abs(d.llr)), 1e-6);
        for (const { w, llr } of ranked) {
          const frac = Math.abs(llr) / maxAbs;             // 0..1 bar length
          const spammy = llr > 0;
          barsWrap.append(
            el("div", { class: "bar-row" },
              el("span", { class: "bl" }, w),
              el("div", { class: "bar-track" },
                el("div", { class: "bar-fill" + (spammy ? " warm" : ""), style: { width: (frac * 100).toFixed(0) + "%" } })),
              el("span", { class: "bar-val" }, (llr >= 0 ? "+" : "") + llr.toFixed(2))));
        }
      }

      // plain-English verdict
      if (!words.length) {
        verdict.innerHTML = "Empty message — with no words, only the <b>prior</b> speaks: " +
          `P(spam) = ${priorSpam.toFixed(2)} (the corpus is half spam, half ham).`;
      } else {
        const spamLeaning = ranked.filter((d) => d.llr > 0).map((d) => d.w);
        const hamLeaning = ranked.filter((d) => d.llr < 0).map((d) => d.w);
        const fmt = (arr) => arr.slice(0, 2).map((w) => `<code>${w}</code>`).join(" and ");
        let drivers = "";
        if (pSpam >= 0.5 && spamLeaning.length) drivers = `${fmt(spamLeaning)} pushed this`;
        else if (pSpam < 0.5 && hamLeaning.length) drivers = `${fmt(hamLeaning)} pulled this`;
        else if (spamLeaning.length || hamLeaning.length) drivers = `the evidence pushed this`;
        else drivers = "with no recognised words the prior pushed this";
        verdict.innerHTML =
          `${drivers} <b>${pct}%</b> ${pSpam >= 0.5 ? "spam" : "ham"} → flagged <b>${pSpam >= 0.5 ? "SPAM" : "HAM"}</b>. ` +
          `Evidence summed to <b>${(evidenceOnly >= 0 ? "+" : "") + evidenceOnly.toFixed(2)}</b> log-odds on top of the prior.`;
      }
    }

    textarea.addEventListener("input", classify);

    // ---- init -------------------------------------------------------
    textarea.value = "win a free prize now";
    classify();

    return {
      onEnter() { classify(); },
      onLeave() { /* no Loop running; nothing to stop */ },
    };
  },

  lesson: `
  <div class="lesson-block">
    <h3>📨 The job: is this message spam?</h3>
    <p>Naive Bayes is a <b>supervised classifier</b>. Show it a pile of messages each labelled <em>spam</em> or <em>ham</em> (legit), and it learns which words are suspicious. Then for a brand-new message it answers a probability question: <b>given these words, how likely is this spam?</b> It's the classic spam-filter engine — fast, tiny, and shockingly hard to beat as a text baseline.</p>
  </div>
  <div class="lesson-block">
    <h3>🧮 Bayes' rule — flip the question around</h3>
    <p>We can't directly measure "probability of spam given the words." But from the training data we <em>can</em> measure the reverse: how often each word shows up in spam vs ham. Bayes' rule flips one into the other:</p>
    <div class="formula">P(class | words) = P(words | class) · P(class) ⁄ P(words)</div>
    <p>In words: <b>posterior ∝ likelihood × prior</b>. The <b>prior</b> P(class) is how common each class is before reading anything (here ~50/50). The <b>likelihood</b> P(words|class) is how spam-like or ham-like the actual words are. The denominator P(words) is the same for both classes, so to <em>compare</em> spam vs ham we can ignore it.</p>
  </div>
  <div class="lesson-block">
    <h3>😇 The "naive" assumption</h3>
    <p>P(words | class) for a whole sentence is hard — words interact. So Naive Bayes makes a bold, technically-wrong simplification: it assumes every word is <b>independent of the others given the class</b>. Then the joint likelihood is just a product:</p>
    <div class="formula">P(w<sub>1</sub>,w<sub>2</sub>,…,w<sub>n</sub> | c) = ∏<sub>i</sub> P(w<sub>i</sub> | c)</div>
    <p>Real language isn't independent ("new york", "click here") — but the approximation is cheap and the <em>decision</em> usually still lands on the right side. That's the famous "wrong but works."</p>
    <p>Each per-word probability is estimated by counting, with <b>Laplace (add-one) smoothing</b>:</p>
    <div class="formula">P(w | c) = ( count<sub>c</sub>(w) + 1 ) ⁄ ( total words in c + V )</div>
    <p>where V is the vocabulary size. The <b>+1</b> matters: without it, a single never-before-seen word would give P(w|c)=0 and zero out the entire product — one odd word would veto everything. Add-one keeps every probability positive.</p>
  </div>
  <div class="lesson-block">
    <h3>➕ Why we work in log space</h3>
    <p>Multiplying dozens of tiny probabilities (each ≪ 1) underflows to <code>0</code> in floating point — the whole answer collapses. Taking a logarithm turns the fragile <em>product</em> into a sturdy <em>sum of evidence</em>:</p>
    <div class="formula">score<sub>c</sub> = log P(c) + Σ<sub>i</sub> log P(w<sub>i</sub> | c)</div>
    <p>Each word just <b>adds</b> a little evidence to each class's running total — that's the "Evidence (log-odds)" readout above. To turn the two scores back into a probability we use the softmax of two numbers, which is exactly the sigmoid of their difference:</p>
    <div class="formula">P(spam | msg) = 1 ⁄ ( 1 + e<sup>(score<sub>ham</sub> − score<sub>spam</sub>)</sup> )</div>
    <p>Each token chip above shows the sign of one word's log-ratio <code>log P(w|spam) ⁄ P(w|ham)</code>: positive = coral (spammy), negative = blue (hammy). Out-of-vocab words still get the +1 smoothing term, so they nudge almost nothing.</p>
  </div>
  <div class="lesson-block">
    <h3>⚖️ The decision threshold &amp; failure modes</h3>
    <ul class="bullets">
      <li><b>Threshold.</b> We flag spam when P(spam) ≥ 0.5. A real filter raises the bar (say 0.9) to avoid junking real mail — trading a few missed spams for far fewer false alarms.</li>
      <li><b>The independence lie bites on correlated words.</b> Repeating "free free free" triple-counts the same signal, so confidence can be overstated.</li>
      <li><b>Garbage in.</b> If your training corpus is skewed or tiny, the priors and likelihoods are skewed too — the model only knows the words it has seen.</li>
      <li><b>Probabilities are over-confident.</b> Because of the naive assumption, the number often slams toward 0% or 100%. Trust the <em>ranking/decision</em> more than the exact percentage.</li>
    </ul>
  </div>
  <div class="callout key">
    <p><b>In one line:</b> Naive Bayes counts how spam-like each word is, pretends the words are independent so it can just <em>add up</em> their log-evidence on top of a prior, and flips that total into P(spam) with a sigmoid — a fast, dependable text-classification baseline.</p>
  </div>`,
};
