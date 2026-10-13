/**
 * tests.js — verification suite. Runs on the main page (fast tests) and in
 * tests/index.html (all tests, including the slow integration test).
 *
 * Each test returns { pass, detail }. NQ.runTests(options) returns the results with timings.
 */
(function (NQ) {
  "use strict";
  const E = NQ.engine;

  const fmt = v => v.toExponential(1);

  /**
   * Max relative error between analytic gradient g and central finite differences of f over params p.
   * The denominator has a floor of 1e-6 × the largest gradient component. Without it, parameters
   * whose true gradient is ~1e-14 (saturated units) would be judged against finite-difference
   * round-off (~1e-9) and fail spuriously.
   */
  function gradCheck(p, g, f, h = 1e-3) {
    const num = new Float64Array(p.length);
    for (let i = 0; i < p.length; i++) {
      const o = p[i];
      const at = d => { p[i] = o + d; const v = f(); p[i] = o; return v; };
      // fourth-order central difference: truncation error O(h⁴) instead of O(h²)
      num[i] = (8 * (at(h) - at(-h)) - (at(2 * h) - at(-2 * h))) / (12 * h);
    }
    let gmax = 0;
    for (const v of num) gmax = Math.max(gmax, Math.abs(v));
    const floor = Math.max(1e-12, 1e-6 * gmax);
    let worst = 0;
    for (let i = 0; i < p.length; i++) {
      worst = Math.max(worst, Math.abs(num[i] - g[i]) / Math.max(floor, Math.abs(num[i]) + Math.abs(g[i])));
    }
    return worst;
  }

  const TESTS = [
    {
      group: "Engine", name: "Network derivative u′(x) matches finite differences",
      run() {
        const net = E.makeNet([1, 8, 8, 1], 3, { scale: 0.3 });
        let worst = 0;
        for (const x of [-4, -1.3, 0, 0.7, 2.9]) {
          const [, u1] = E.forward(net, x);
          const h = 1e-5, fd = (E.predict(net, x + h) - E.predict(net, x - h)) / (2 * h);
          worst = Math.max(worst, Math.abs(u1 - fd) / Math.max(1e-8, Math.abs(fd)));
        }
        return { pass: worst < 1e-6, detail: `max relative error ${fmt(worst)}` };
      },
    },
    {
      group: "Engine", name: "Backpropagation through the Taylor pass matches finite differences",
      run() {
        const net = E.makeNet([1, 7, 6, 1], 11, { scale: 0.25 });
        const pts = [[-2, 0.8, -1.1], [0.4, -0.3, 0.6], [1.9, 1.2, 0.9]]; // x, weight on u, weight on u'
        const f = () => {
          let s = 0;
          for (const [x, a, b] of pts) { const [u, u1] = E.forward(net, x); s += a * u * u + b * u1 * u1; }
          return s;
        };
        E.zeroGrad(net);
        for (const [x, a, b] of pts) { const [u, u1] = E.forward(net, x); E.backward(net, 2 * a * u, 2 * b * u1); }
        const worst = gradCheck(net.p, Float64Array.from(net.g), f);
        return { pass: worst < 1e-6, detail: `${net.n} parameters, max relative error ${fmt(worst)}` };
      },
    },
    {
      group: "Engine", name: "Adam drives a network output to a target",
      run() {
        const net = E.makeNet([1, 8, 1], 5);
        for (let i = 0; i < 600; i++) {
          E.zeroGrad(net);
          const [u] = E.forward(net, 0.5);
          E.backward(net, 2 * (u - 3), 0);
          E.adam(net, 1e-2);
        }
        const err = Math.abs(E.predict(net, 0.5) - 3);
        return { pass: err < 1e-3, detail: `|u − 3| = ${fmt(err)} after 600 steps` };
      },
    },
    {
      group: "Exact solver", name: "Harmonic oscillator energies equal n + ½",
      run() {
        const sol = NQ.eigen.solve(x => 0.5 * x * x, 6, 5);
        const worst = Math.max(...sol.energies.map((e, n) => Math.abs(e - (n + 0.5))));
        return { pass: worst < 1e-7, detail: `n = 0…4, max error ${fmt(worst)}` };
      },
    },
    {
      group: "Exact solver", name: "Infinite square well energies equal n²π²/8L²",
      run() {
        const L = 3, sol = NQ.eigen.solve(() => 0, L, 4);
        const worst = Math.max(...sol.energies.map((e, i) => { const n = i + 1, ex = n * n * Math.PI ** 2 / (8 * L * L); return Math.abs(e - ex) / ex; }));
        return { pass: worst < 1e-8, detail: `n = 1…4, max relative error ${fmt(worst)}` };
      },
    },
    {
      group: "Exact solver", name: "Morse energies match ω(n+½) − [ω(n+½)]²/4D",
      run() {
        const D = 8, a = 0.7, w = a * Math.sqrt(2 * D);
        const sol = NQ.eigen.solve(x => Math.min(40, D * (1 - Math.exp(-a * (x + 2))) ** 2), 6, 3);
        const worst = Math.max(...sol.energies.map((e, n) => { const ex = w * (n + 0.5) - (w * (n + 0.5)) ** 2 / (4 * D); return Math.abs(e - ex) / ex; }));
        return { pass: worst < 5e-6, detail: `n = 0…2, max relative error ${fmt(worst)}` };
      },
    },
    {
      group: "Exact solver", name: "Eigenvectors are orthonormal",
      run() {
        const sol = NQ.eigen.solve(x => 0.3 * (x * x - 4) ** 2, 6, 4);
        let worst = 0;
        for (let i = 0; i < 4; i++) for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let q = 0; q < sol.x.length; q++) s += sol.h * sol.states[i][q] * sol.states[j][q];
          worst = Math.max(worst, Math.abs(s - (i === j ? 1 : 0)));
        }
        return { pass: worst < 1e-8, detail: `double well, max |⟨ψᵢ|ψⱼ⟩ − δᵢⱼ| = ${fmt(worst)}` };
      },
    },
    {
      group: "Exact solver", name: "Eigenpairs satisfy Hψ = Eψ",
      run() {
        const sol = NQ.eigen.solve(x => 8 * (1 - Math.exp(-0.7 * (x + 2))) ** 2, 6, 3);
        const { d, e } = sol;
        let worst = 0;
        sol.states.forEach((v, k) => {
          const lam = sol.energiesFine[k];
          let r = 0, nv = 0;
          for (let i = 0; i < v.length; i++) {
            const Hv = d[i] * v[i] + (i > 0 ? e * v[i - 1] : 0) + (i < v.length - 1 ? e * v[i + 1] : 0);
            r += (Hv - lam * v[i]) ** 2; nv += v[i] * v[i];
          }
          worst = Math.max(worst, Math.sqrt(r / nv) / Math.abs(lam));
        });
        return { pass: worst < 1e-8, detail: `Morse, max ‖Hψ − Eψ‖ / (E‖ψ‖) = ${fmt(worst)}` };
      },
    },
    {
      group: "Physics", name: "Trapezoid quadrature integrates a Gaussian to machine precision",
      run() {
        const s = new NQ.VariationalSolver({ V: () => 0, L: 6, nStates: 1, M: 200 });
        let I = 0;
        for (const x of s.x) I += s.h * Math.exp(-x * x);
        const err = Math.abs(I - Math.sqrt(Math.PI));
        return { pass: err < 1e-10, detail: `|∫e^(−x²) − √π| = ${fmt(err)}` };
      },
    },
    {
      group: "Physics", name: "Variational principle: random trial states never fall below E₀",
      run() {
        let lowest = Infinity;
        for (let seed = 1; seed <= 25; seed++) {
          const s = new NQ.VariationalSolver({ V: x => 0.5 * x * x, L: 6, nStates: 1, seed: seed * 101 });
          s.evaluate(0);
          lowest = Math.min(lowest, s.project(0).E);
        }
        return { pass: lowest >= 0.5 - 1e-9, detail: `lowest of 25 random networks: E = ${lowest.toFixed(4)} ≥ 0.5` };
      },
    },
    {
      group: "Physics", name: "Energy gradient through the Gram–Schmidt projection matches finite differences",
      run() {
        const s = new NQ.VariationalSolver({ V: x => 0.2 * (x * x - 3) ** 2, L: 6, nStates: 3, M: 40, hidden: [6, 5], seed: 9 });
        let worst = 0, n = 0;
        for (let k = 0; k < 3; k++) {
          let Ik;
          for (let j = 0; j <= k; j++) { s.evaluate(j); Ik = s.project(j); }
          s.gradient(k, Ik);
          worst = Math.max(worst, gradCheck(s.nets[k].p, Float64Array.from(s.nets[k].g), () => s.loss(k)));
          n += s.nets[k].n;
        }
        return { pass: worst < 1e-5, detail: `3 states, ${n} parameters, max relative error ${fmt(worst)}` };
      },
    },
    {
      group: "Integration", slow: true, name: "Training finds the harmonic ground state",
      run() {
        const s = new NQ.VariationalSolver({ V: x => 0.5 * x * x, L: 6, nStates: 1, seed: 1 });
        let I;
        for (let i = 0; i < 1500; i++) I = s.train(1e-2 * Math.pow(0.05, i / 1500));
        const err = Math.abs(I[0].E - 0.5);
        return { pass: err < 1e-3, detail: `E₀ = ${I[0].E.toFixed(6)} after 1500 steps (exact 0.5)` };
      },
    },
  ];

  /** Runs the suite. options.includeSlow adds integration tests. */
  function runTests(options = {}) {
    return TESTS.filter(t => options.includeSlow || !t.slow).map(t => {
      const t0 = performance.now();
      let res;
      try { res = t.run(); } catch (err) { res = { pass: false, detail: "threw: " + err.message }; }
      return { group: t.group, name: t.name, slow: !!t.slow, ms: performance.now() - t0, ...res };
    });
  }

  NQ.tests = { TESTS, runTests, gradCheck };
})(self.NQ = self.NQ || {});
