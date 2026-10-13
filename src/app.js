/**
 * app.js — page controller: training host (worker or main thread), rendering and interaction.
 */
(function () {
  "use strict";
  const NQ = self.NQ, P = NQ.potentials, L = P.L;
  const $ = id => document.getElementById(id);
  const DRAW_MAX = 12, DRAW_N = 241, N_CHOICES = [1, 2, 3, 4];

  // ------------------------------------------------------------------ state
  const state = {
    potential: { kind: "preset", id: "harmonic" },
    customSamples: null,
    nStates: 3,
    seed: 1,
    view: "psi",
    running: true,
    drawing: false,
    snap: null,
    gen: 0,
    dirty: true,
  };

  // ------------------------------------------------------------------ theme
  let C = {};
  function readTheme() {
    const s = getComputedStyle(document.documentElement);
    const v = k => s.getPropertyValue(k).trim();
    C = {
      ink: v("--ink"), muted: v("--muted"), line: v("--line"), grid: v("--grid"), well: v("--well"),
      panel: v("--panel"), chip: v("--chip"), accent: v("--accent"), good: v("--good"),
      states: [v("--s0"), v("--s1"), v("--s2"), v("--s3")], mono: v("--f-mono"), body: v("--f-body"),
    };
    state.dirty = true;
  }

  // ------------------------------------------------------------------ training host
  // Prefers a Web Worker; falls back to the main thread if workers are unavailable
  // (e.g. the page was opened from file://) or the worker stays silent.
  const host = {
    mode: null, worker: null, local: null, cfg: null, heard: false, watchdog: 0,

    start(cfg) {
      this.cfg = cfg;
      state.gen++;
      state.snap = null;
      state.running = true;
      state.dirty = true;
      if (this.mode !== "local") {
        if (!this.worker) this.worker = this.spawn();
        if (this.worker) {
          this.mode = "worker";
          this.worker.postMessage({ type: "init", gen: state.gen, cfg });
          if (!this.heard) {
            clearTimeout(this.watchdog);
            this.watchdog = setTimeout(() => { if (!this.heard) this.fallback(); }, 4000);
          }
          return;
        }
      }
      this.mode = "local";
      this.local = new NQ.Trainer(cfg);
    },

    spawn() {
      try {
        const w = new Worker("src/worker.js");
        w.onmessage = e => this.onMessage(e.data);
        w.onerror = e => { if (e.preventDefault) e.preventDefault(); this.fallback(); };
        return w;
      } catch (err) {
        return null;
      }
    },

    fallback() {
      if (this.worker) { this.worker.terminate(); this.worker = null; }
      this.mode = "local";
      if (this.cfg) this.start(this.cfg);
    },

    onMessage(m) {
      this.heard = true;
      if (m.gen !== state.gen) return; // stale run
      if (m.type === "snapshot") { state.snap = m.snap; state.dirty = true; }
      else if (m.type === "error") showError(m.message);
    },

    pause() {
      state.running = false;
      if (this.mode === "worker") this.worker.postMessage({ type: "pause" });
    },

    resume() {
      state.running = true;
      if (this.mode === "worker") this.worker.postMessage({ type: "resume" });
      else if (this.local && this.local.done) this.local.extend();
    },

    /** Called every animation frame. Only does work in main-thread mode. */
    frame() {
      if (this.mode !== "local" || !this.local) return;
      if (state.running && !this.local.done) this.local.run(12);
      if (state.running || !state.snap) { state.snap = this.local.snapshot(); state.dirty = true; }
    },
  };

  function startTraining() {
    const potential = state.potential.kind === "samples"
      ? { kind: "samples", samples: Array.from(state.customSamples) }
      : state.potential;
    host.start({ potential, nStates: state.nStates, seed: state.seed });
    updateToggle();
  }

  // ------------------------------------------------------------------ potential helpers
  function currentV() {
    return state.potential.kind === "samples" ? P.fromSamples(state.customSamples) : P.PRESETS.find(p => p.id === state.potential.id).V;
  }

  function samplesFrom(V) {
    return Float64Array.from({ length: DRAW_N }, (_, i) => Math.min(DRAW_MAX, V(-L + 2 * L * i / (DRAW_N - 1))));
  }

  function smooth(a, passes = 3) {
    let s = Float64Array.from(a);
    for (let p = 0; p < passes; p++) {
      const t = Float64Array.from(s);
      for (let i = 1; i < s.length - 1; i++) t[i] = 0.25 * s[i - 1] + 0.5 * s[i] + 0.25 * s[i + 1];
      s = t;
    }
    return s;
  }

  // ------------------------------------------------------------------ canvas basics
  function setup(cv) {
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    return { ctx, W, H };
  }

  function niceTicks(lo, hi, count) {
    const raw = (hi - lo) / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  const fmtTick = v => (Math.abs(v) >= 1000 ? (v / 1000) + "k" : String(Math.round(v * 1000) / 1000));

  // ------------------------------------------------------------------ main energy diagram
  const MAIN_PAD = { l: 46, r: 96, t: 14, b: 30 };
  let mainView = null; // last transform, used by draw mode

  function mainRange(snap, Vgrid) {
    if (state.drawing) return { y0: -0.6, y1: DRAW_MAX + 0.6 };
    let vmin = Infinity;
    for (const v of Vgrid) vmin = Math.min(vmin, v);
    const E = snap ? snap.exactEnergies : [];
    if (!E.length) return { y0: vmin - 0.5, y1: vmin + 6 };
    const n = E.length, top = E[n - 1];
    const spacing = n > 1 ? (top - E[0]) / (n - 1) : Math.max(0.4, E[0] - vmin);
    return { y0: vmin - 0.18 * spacing - 0.05, y1: top + 1.05 * spacing };
  }

  function drawMain() {
    const cv = $("cv-main");
    const { ctx, W, H } = setup(cv);
    if (!W || !H) return;
    const pad = MAIN_PAD, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const x0 = -L, x1 = L;
    const V = currentV();
    const NX = 480;
    const xs = Float64Array.from({ length: NX }, (_, i) => x0 + (x1 - x0) * i / (NX - 1));
    const Vg = xs.map(V);
    const snap = state.snap;
    const { y0, y1 } = mainRange(snap, Vg);
    const X = x => pad.l + (x - x0) / (x1 - x0) * pw;
    const Y = y => pad.t + (1 - (y - y0) / (y1 - y0)) * ph;
    mainView = { pad, pw, ph, x0, x1, y0, y1 };

    // grid and axes
    ctx.font = `11px ${C.mono}`;
    ctx.strokeStyle = C.line; ctx.fillStyle = C.muted; ctx.lineWidth = 1;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of niceTicks(y0, y1, 6)) {
      const yy = Math.round(Y(v)) + 0.5;
      ctx.globalAlpha = 0.55; ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(fmtTick(v), pad.l - 6, yy);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const v of niceTicks(x0, x1, 8)) {
      const xx = Math.round(X(v)) + 0.5;
      ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.moveTo(xx, pad.t); ctx.lineTo(xx, pad.t + ph); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(fmtTick(v), xx, pad.t + ph + 7);
    }
    ctx.save(); ctx.translate(12, pad.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("energy", 0, 0); ctx.restore();
    ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText("position x", pad.l + pw - 6, pad.t + ph - 5);

    ctx.save();
    ctx.beginPath(); ctx.rect(pad.l, pad.t, pw, ph); ctx.clip();

    // potential
    ctx.beginPath();
    ctx.moveTo(X(x0), pad.t + ph + 2);
    for (let i = 0; i < NX; i++) ctx.lineTo(X(xs[i]), Y(Vg[i]));
    ctx.lineTo(X(x1), pad.t + ph + 2);
    ctx.closePath();
    ctx.fillStyle = C.well; ctx.fill();
    ctx.beginPath();
    for (let i = 0; i < NX; i++) i ? ctx.lineTo(X(xs[i]), Y(Vg[i])) : ctx.moveTo(X(xs[i]), Y(Vg[i]));
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.stroke();

    const labels = [];
    if (snap && snap.states.length) {
      const E = snap.exactEnergies, n = snap.states.length;
      const spacing = n > 1 ? (E[n - 1] - E[0]) / (n - 1) : Math.max(0.4, E[0] - Math.min(...Vg));
      const amp = 0.42 * spacing;
      const sx = snap.x;
      snap.states.forEach((st, k) => {
        const col = C.states[k % 4];
        const val = a => (state.view === "psi" ? a : a * a);
        let peak = 0;
        for (const r of st.ref) peak = Math.max(peak, Math.abs(val(r)));
        const sc = peak ? amp / peak : 0;

        // exact level, drawn only over classically allowed regions (V < E)
        ctx.setLineDash([5, 4]); ctx.strokeStyle = C.ink; ctx.globalAlpha = 0.45; ctx.lineWidth = 1.2;
        ctx.beginPath();
        let open = false;
        for (let i = 0; i < NX; i++) {
          const inside = Vg[i] < st.Eexact;
          if (inside && !open) { ctx.moveTo(X(xs[i]), Y(st.Eexact)); open = true; }
          else if (!inside && open) { ctx.lineTo(X(xs[i]), Y(st.Eexact)); open = false; }
        }
        if (open) ctx.lineTo(X(x1), Y(st.Eexact));
        ctx.stroke();

        // exact wavefunction
        ctx.beginPath();
        for (let i = 0; i < sx.length; i++) {
          const yy = Y(st.Eexact + sc * val(st.ref[i]));
          i ? ctx.lineTo(X(sx[i]), yy) : ctx.moveTo(X(sx[i]), yy);
        }
        ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;

        // network wavefunction, filled to its own energy level
        ctx.beginPath();
        ctx.moveTo(X(sx[0]), Y(st.E));
        for (let i = 0; i < sx.length; i++) ctx.lineTo(X(sx[i]), Y(st.E + sc * val(st.psi[i])));
        ctx.lineTo(X(sx[sx.length - 1]), Y(st.E));
        ctx.closePath();
        ctx.fillStyle = col; ctx.globalAlpha = 0.12; ctx.fill(); ctx.globalAlpha = 1;
        ctx.beginPath();
        for (let i = 0; i < sx.length; i++) {
          const yy = Y(st.E + sc * val(st.psi[i]));
          i ? ctx.lineTo(X(sx[i]), yy) : ctx.moveTo(X(sx[i]), yy);
        }
        ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.stroke();

        labels.push({ y: Y(st.E), k, E: st.E, col });
      });
    }
    ctx.restore();

    // right-hand labels, nudged apart so close levels (tunnelling pairs) stay readable
    labels.sort((a, b) => b.y - a.y);
    for (let i = 1; i < labels.length; i++) if (labels[i - 1].y - labels[i].y < 30) labels[i].y = labels[i - 1].y - 30;
    for (const lb of labels) {
      const yy = Math.max(pad.t + 12, Math.min(pad.t + ph - 12, lb.y));
      ctx.fillStyle = lb.col; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.font = `600 13px ${C.body}`; ctx.fillText(`n = ${lb.k}`, pad.l + pw + 10, yy - 2);
      ctx.font = `11.5px ${C.mono}`; ctx.fillStyle = C.muted; ctx.fillText(`E ${lb.E.toFixed(4)}`, pad.l + pw + 10, yy + 12);
    }

    ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, pw - 1, ph - 1);
  }

  // ------------------------------------------------------------------ convergence chart
  function drawConv() {
    const cv = $("cv-conv");
    const { ctx, W, H } = setup(cv);
    if (!W || !H) return;
    const pad = { l: 46, r: 12, t: 8, b: 24 }, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const snap = state.snap;
    const xmax = Math.max(500, snap ? snap.step : 0);
    const e0 = -6, e1 = 2;
    const X = s => pad.l + s / xmax * pw;
    const Y = v => pad.t + (1 - (Math.log10(Math.max(v, 1e-9)) - e0) / (e1 - e0)) * ph;
    ctx.font = `11px ${C.mono}`; ctx.strokeStyle = C.line; ctx.fillStyle = C.muted; ctx.lineWidth = 1;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let e = e0; e <= e1; e++) {
      const yy = Math.round(Y(Math.pow(10, e))) + 0.5;
      ctx.globalAlpha = 0.55; ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke(); ctx.globalAlpha = 1;
      if ((e - e0) % 2 === 0) ctx.fillText("1e" + e, pad.l - 6, yy);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const v of niceTicks(0, xmax, 5)) ctx.fillText(fmtTick(v), X(v), pad.t + ph + 6);
    if (snap) {
      ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, pw, ph); ctx.clip();
      const hs = snap.history.step;
      snap.history.err.forEach((arr, k) => {
        if (!arr.length) return;
        ctx.strokeStyle = C.states[k % 4]; ctx.lineWidth = 1.7; ctx.lineJoin = "round";
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) i ? ctx.lineTo(X(hs[i]), Y(arr[i])) : ctx.moveTo(X(hs[i]), Y(arr[i]));
        ctx.stroke();
        const j = arr.length - 1;
        ctx.fillStyle = C.states[k % 4]; ctx.beginPath(); ctx.arc(X(hs[j]), Y(arr[j]), 3, 0, 7); ctx.fill();
      });
      ctx.restore();
    }
    ctx.strokeStyle = C.line; ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, pw - 1, ph - 1);
  }

  // ------------------------------------------------------------------ panels
  const pct = (v, d = 2) => (v * 100).toFixed(d) + "%";

  function renderPanels() {
    const snap = state.snap;
    const st = snap ? snap.states : [];
    $("status").innerHTML = snap
      ? `${snap.done ? "done · " : ""}step <b>${snap.step.toLocaleString()}</b><br>seed ${state.seed} · ${host.mode === "worker" ? "background thread" : "main thread"}`
      : "starting…";
    $("prog").style.width = snap ? Math.min(100, snap.step / snap.limit * 100) + "%" : "0";
    $("results").innerHTML = st.map(s => `
      <tr><td><span class="dot" style="background:${C.states[s.n % 4]}"></span>n = ${s.n}</td>
      <td>${s.Eexact.toFixed(4)}</td><td>${s.E.toFixed(4)}</td><td>${s.err.toExponential(1)}</td><td>${pct(s.fidelity)}</td></tr>`).join("")
      || `<tr><td colspan="5" style="color:var(--muted)">Initialising…</td></tr>`;

    const b = $("banner");
    if (!snap || snap.step < 150) {
      b.className = "banner"; b.innerHTML = "<b>Starting from random networks</b>Each state begins as noise. Watch the energies fall.";
    } else if (st.every(s => s.fidelity > 0.999 && s.err / Math.max(0.05, Math.abs(s.Eexact)) < 0.005)) {
      b.className = "banner ok"; b.innerHTML = `<b>Solved</b>All ${st.length} states match the exact solution with fidelity above 99.9%.`;
    } else if (snap.done) {
      b.className = "banner"; b.innerHTML = "<b>Paused at the step limit</b>Press Train more to keep refining.";
    } else {
      b.className = "banner"; b.innerHTML = "<b>Training</b>Lowering each state's energy while keeping it orthogonal to the states below.";
    }

    if (st.length) {
      const s0 = st[0];
      $("st-err").textContent = pct(s0.err / Math.max(0.05, Math.abs(s0.Eexact)), s0.err / Math.abs(s0.Eexact) < 0.001 ? 3 : 2);
      $("st-fid").textContent = pct(st.reduce((a, s) => a + s.fidelity, 0) / st.length);
      $("st-speed").textContent = Math.round(snap.stepsPerSec).toLocaleString();
      $("st-speed-s").textContent = `steps/s · ${host.mode === "worker" ? "background thread" : "main thread"}`;
    }
    updateToggle();
  }

  function updateToggle() {
    const done = state.snap && state.snap.done;
    $("btn-toggle").textContent = state.running && !done ? "Pause" : done ? "Train more" : "Resume";
  }

  function renderPotentialInfo() {
    if (state.potential.kind === "samples") {
      $("pot-name").textContent = "Your potential";
      $("pot-formula").textContent = "V(x) drawn by hand";
      $("pot-blurb").textContent = "The exact solver re-solves whatever you draw in a few milliseconds, so the network is always graded against the right answer.";
    } else {
      const p = P.PRESETS.find(q => q.id === state.potential.id);
      $("pot-name").textContent = p.name;
      $("pot-formula").textContent = p.formula;
      $("pot-blurb").textContent = p.blurb;
    }
  }

  function showError(msg) {
    const b = $("banner");
    b.className = "banner"; b.innerHTML = `<b>Training stopped</b>${msg}. Press New seed to try again.`;
  }

  // ------------------------------------------------------------------ controls
  function buildControls() {
    const wrap = $("presets");
    const chips = P.PRESETS.map(p => ({ id: p.id, label: p.name.replace(/ \(.*\)$/, "") }));
    wrap.innerHTML = chips.map(c => `<button class="chip" data-id="${c.id}" aria-pressed="false">${c.label}</button>`).join("")
      + `<button class="chip draw" data-id="draw" aria-pressed="false">Draw your own</button>`;
    wrap.addEventListener("click", e => {
      const btn = e.target.closest("button");
      if (!btn) return;
      const id = btn.dataset.id;
      if (id === "draw") { enterDraw(); return; }
      exitDraw(false);
      state.potential = { kind: "preset", id };
      renderPotentialInfo(); syncChips(); startTraining();
    });

    $("n-seg").innerHTML = N_CHOICES.map(n => `<button data-n="${n}" aria-pressed="${n === state.nStates}">${n}</button>`).join("");
    $("n-seg").addEventListener("click", e => {
      const btn = e.target.closest("button");
      if (!btn) return;
      state.nStates = +btn.dataset.n;
      [...$("n-seg").children].forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.n === state.nStates)));
      $("n-out").textContent = state.nStates;
      startTraining();
    });
    $("n-out").textContent = state.nStates;

    for (const [id, v] of [["view-psi", "psi"], ["view-prob", "prob"]]) {
      $(id).addEventListener("click", () => {
        state.view = v;
        $("view-psi").setAttribute("aria-pressed", String(v === "psi"));
        $("view-prob").setAttribute("aria-pressed", String(v === "prob"));
        state.dirty = true;
      });
    }
    $("btn-toggle").addEventListener("click", () => {
      const done = state.snap && state.snap.done;
      if (state.running && !done) host.pause(); else host.resume();
      updateToggle();
    });
    $("btn-restart").addEventListener("click", () => { state.seed++; startTraining(); });
    $("draw-done").addEventListener("click", () => exitDraw(true));
    $("draw-reset").addEventListener("click", () => {
      state.customSamples = Float64Array.from({ length: DRAW_N }, (_, i) => { const x = -L + 2 * L * i / (DRAW_N - 1); return Math.abs(x) < 3 ? 0 : 8; });
      state.customSamples = smooth(state.customSamples, 2);
      state.potential = { kind: "samples" };
      state.dirty = true;
      startTraining();
    });
  }

  function syncChips() {
    const active = state.drawing || state.potential.kind === "samples" ? "draw" : state.potential.id;
    document.querySelectorAll("#presets .chip").forEach(c => c.setAttribute("aria-pressed", String(c.dataset.id === active)));
  }

  // ------------------------------------------------------------------ draw mode
  function enterDraw() {
    if (!state.customSamples || state.potential.kind !== "samples") state.customSamples = samplesFrom(currentV());
    state.potential = { kind: "samples" };
    state.drawing = true;
    $("drawbar").hidden = false;
    $("cv-main").classList.add("drawing");
    renderPotentialInfo(); syncChips();
    state.dirty = true;
  }

  function exitDraw(train) {
    if (!state.drawing) return;
    state.drawing = false;
    $("drawbar").hidden = true;
    $("cv-main").classList.remove("drawing");
    syncChips();
    state.dirty = true;
    if (train && !state.running) host.resume();
  }

  function wireDrawing() {
    const cv = $("cv-main");
    let last = null;
    const toData = e => {
      const r = cv.getBoundingClientRect(), v = mainView;
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const x = v.x0 + (px - v.pad.l) / v.pw * (v.x1 - v.x0);
      const y = v.y1 - (py - v.pad.t) / v.ph * (v.y1 - v.y0);
      return { i: Math.round((x + L) / (2 * L) * (DRAW_N - 1)), y: Math.max(0, Math.min(DRAW_MAX, y)) };
    };
    const paint = p => {
      const s = state.customSamples;
      const from = last || p;
      const lo = Math.min(from.i, p.i), hi = Math.max(from.i, p.i);
      for (let i = Math.max(0, lo); i <= Math.min(DRAW_N - 1, hi); i++) {
        const t = hi === lo ? 1 : (i - from.i) / (p.i - from.i);
        s[i] = from.y + (p.y - from.y) * t;
      }
      last = p;
      state.dirty = true;
    };
    cv.addEventListener("pointerdown", e => {
      if (!state.drawing || !mainView) return;
      try { cv.setPointerCapture(e.pointerId); } catch (err) { /* capture is a nicety, not required */ }
      last = null;
      paint(toData(e));
      e.preventDefault();
    });
    cv.addEventListener("pointermove", e => { if (state.drawing && last) paint(toData(e)); });
    const end = () => {
      if (!state.drawing || !last) return;
      last = null;
      state.customSamples = smooth(state.customSamples);
      startTraining(); // retrain on every stroke so results appear while drawing
    };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
  }

  // ------------------------------------------------------------------ verification
  function runChecks() {
    const res = NQ.tests.runTests({ includeSlow: false });
    const ok = res.filter(r => r.pass).length;
    $("st-tests").textContent = `${ok}/${res.length}`;
    const sum = $("ver-summary");
    sum.textContent = `${ok} of ${res.length} checks passed in ${res.reduce((a, r) => a + r.ms, 0).toFixed(0)} ms`;
    sum.className = "pill" + (ok === res.length ? " ok" : "");
    $("checks").innerHTML = res.map(r => `
      <div class="check"><span class="r${r.pass ? "" : " fail"}">${r.pass ? "PASS" : "FAIL"}</span>
      <span class="t"><small>${r.group}</small>${r.name}</span><span class="d">${r.detail}</span></div>`).join("");
  }

  // ------------------------------------------------------------------ loop
  let lastPanels = 0;
  function loop(now) {
    host.frame();
    if (state.dirty) {
      state.dirty = false;
      drawMain();
      drawConv();
      if (now - lastPanels > 150) { renderPanels(); lastPanels = now; }
      else state.dirty = true; // make sure the panels catch up next frame
    }
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------ boot
  readTheme();
  buildControls();
  renderPotentialInfo();
  syncChips();
  wireDrawing();
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readTheme); } catch (e) { /* older browsers */ }
  new MutationObserver(readTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  new ResizeObserver(() => { state.dirty = true; }).observe(document.querySelector(".wrap"));
  startTraining();
  requestAnimationFrame(loop);
  setTimeout(runChecks, 400);

  // Exposed for debugging from the console.
  NQ.app = { state, host, render: () => { drawMain(); drawConv(); renderPanels(); } };
})();
