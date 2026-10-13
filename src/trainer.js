/**
 * trainer.js — owns one training run: the potential, the exact reference solution,
 * the neural solver, the learning-rate schedule and the convergence history.
 *
 * The same class runs inside the Web Worker (src/worker.js) and, as a fallback, on the
 * main thread. It only exchanges plain data (snapshots), so either host can drive it.
 */
(function (NQ) {
  "use strict";

  const CONFIG = {
    M: 120,               // quadrature points
    hidden: [32, 32],     // hidden layer widths
    inputScale: 0.5,      // network input = 0.5·x
    lr0: 2e-2, lr1: 3e-4, // Adam learning rate, decayed exponentially …
    decaySteps: 3000,     // … over this many steps, then held
    maxSteps: 4000,
    historyEvery: 10,
  };

  /** Resolves a serialisable potential description to a function V(x). */
  function resolvePotential(desc) {
    if (desc.kind === "samples") return NQ.potentials.fromSamples(desc.samples);
    const p = NQ.potentials.PRESETS.find(q => q.id === desc.id);
    if (!p) throw new Error("unknown potential " + desc.id);
    return p.V;
  }

  class Trainer {
    /** @param {{potential: object, nStates: number, seed?: number}} cfg */
    constructor(cfg) {
      const L = NQ.potentials.L;
      this.cfg = { ...CONFIG, ...cfg };
      this.V = resolvePotential(cfg.potential);
      const t0 = Date.now();
      this.exact = NQ.eigen.solve(this.V, L, cfg.nStates);
      this.exactMs = Date.now() - t0;
      this.solver = new NQ.VariationalSolver({
        V: this.V, L, nStates: cfg.nStates, M: this.cfg.M,
        hidden: this.cfg.hidden, inputScale: this.cfg.inputScale, seed: cfg.seed ?? 1,
      });
      this.limit = this.cfg.maxSteps;
      this.history = { step: [], err: Array.from({ length: cfg.nStates }, () => []) };
      this.trainMs = 0;
    }

    get step() { return this.solver.step; }
    get done() { return this.step >= this.limit; }

    lr(step) {
      const { lr0, lr1, decaySteps } = this.cfg;
      return lr0 * Math.pow(lr1 / lr0, Math.min(step / decaySteps, 1));
    }

    /** Trains for up to `budgetMs` milliseconds. Returns the number of steps taken. */
    run(budgetMs) {
      const t0 = Date.now();
      let n = 0;
      while (!this.done && Date.now() - t0 < budgetMs) {
        const I = this.solver.train(this.lr(this.step));
        n++;
        if (this.step % this.cfg.historyEvery === 0) {
          this.history.step.push(this.step);
          I.forEach((s, k) => this.history.err[k].push(Math.abs(s.E - this.exact.energies[k])));
        }
      }
      this.trainMs += Date.now() - t0;
      return n;
    }

    /** Allows training to continue past the current limit. */
    extend(steps = 2000) { this.limit = this.step + steps; }

    /** Plain-data view of the current state, safe to postMessage. */
    snapshot() {
      const s = this.solver;
      const states = s.stats ? NQ.compare(s, this.exact) : [];
      return {
        step: this.step, limit: this.limit, done: this.done,
        stepsPerSec: this.trainMs ? this.step / (this.trainMs / 1000) : 0,
        x: Array.from(s.x),
        exactEnergies: this.exact.energies.slice(),
        exactMs: this.exactMs,
        states: states.map(q => ({ n: q.n, E: q.E, Eexact: q.Eexact, err: q.err, fidelity: q.fidelity, psi: Array.from(q.psi), ref: Array.from(q.ref) })),
        history: { step: this.history.step.slice(), err: this.history.err.map(a => a.slice()) },
      };
    }
  }

  NQ.Trainer = Trainer;
  NQ.TRAINER_CONFIG = CONFIG;
})(self.NQ = self.NQ || {});
