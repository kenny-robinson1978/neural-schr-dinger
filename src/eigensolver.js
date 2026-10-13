/**
 * eigensolver.js — the exact reference that the neural network is graded against.
 *
 * Discretises H = -½ d²/dx² + V(x) on (-L, L) with hard walls (ψ(±L) = 0) using the
 * three-point finite-difference Laplacian. H becomes a symmetric tridiagonal matrix:
 *
 *   diag_i = 1/h² + V(x_i),   offdiag = -1/(2h²)
 *
 * Eigenvalues come from Sturm-sequence bisection, which is exact up to round-off and
 * finds the k-th eigenvalue directly. Eigenvectors come from inverse iteration with
 * the Thomas algorithm. Richardson extrapolation on grids h and h/2 cancels the O(h²)
 * discretisation error, leaving O(h⁴).
 */
(function (NQ) {
  "use strict";

  /** Number of eigenvalues of the tridiagonal (d, e) strictly below lambda. */
  function sturmCount(d, e2, lambda) {
    let count = 0, q = d[0] - lambda;
    if (q < 0) count++;
    for (let i = 1; i < d.length; i++) {
      q = d[i] - lambda - e2 / (q === 0 ? 1e-300 : q);
      if (q < 0) count++;
    }
    return count;
  }

  /** The k lowest eigenvalues of a symmetric tridiagonal matrix with constant off-diagonal e. */
  function lowestEigenvalues(d, e, k) {
    const e2 = e * e;
    let lo = Infinity, hi = -Infinity;
    for (const di of d) { lo = Math.min(lo, di - 2 * Math.abs(e)); hi = Math.max(hi, di + 2 * Math.abs(e)); }
    const out = [];
    for (let j = 0; j < k; j++) {
      let a = j ? out[j - 1] : lo, b = hi;
      for (let it = 0; it < 200 && b - a > 1e-13 * Math.max(1, Math.abs(a) + Math.abs(b)); it++) {
        const mid = 0.5 * (a + b);
        if (sturmCount(d, e2, mid) > j) b = mid; else a = mid;
      }
      out.push(0.5 * (a + b));
    }
    return out;
  }

  /** Solves (T - shift·I) x = rhs for tridiagonal T = (d, e) in O(n). */
  function thomas(d, e, shift, rhs) {
    const n = d.length, c = new Float64Array(n), x = new Float64Array(n);
    let beta = d[0] - shift;
    if (beta === 0) beta = 1e-300;
    x[0] = rhs[0] / beta;
    for (let i = 1; i < n; i++) {
      c[i] = e / beta;
      beta = d[i] - shift - e * c[i];
      if (beta === 0) beta = 1e-300;
      x[i] = (rhs[i] - e * x[i - 1]) / beta;
    }
    for (let i = n - 2; i >= 0; i--) x[i] -= c[i + 1] * x[i + 1];
    return x;
  }

  /** Eigenvector for a known eigenvalue via inverse iteration. */
  function eigenvector(d, e, lambda) {
    const n = d.length;
    let v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = 1 + 0.01 * Math.sin(i * 1.7); // generic start vector
    const shift = lambda + 1e-10 * Math.max(1, Math.abs(lambda));
    for (let it = 0; it < 4; it++) {
      v = thomas(d, e, shift, v);
      let nrm = 0;
      for (let i = 0; i < n; i++) nrm += v[i] * v[i];
      nrm = Math.sqrt(nrm);
      for (let i = 0; i < n; i++) v[i] /= nrm;
    }
    return v;
  }

  /**
   * Single-grid solve. Returns grid points, energies and eigenvectors normalised so that
   * Σ h ψ² = 1 (the discrete version of ∫ψ² dx = 1), with a fixed sign convention.
   */
  function solveGrid(V, L, N, k) {
    const h = 2 * L / (N + 1);
    const x = new Float64Array(N), d = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      x[i] = -L + (i + 1) * h;
      d[i] = 1 / (h * h) + V(x[i]);
    }
    const e = -1 / (2 * h * h);
    const energies = lowestEigenvalues(d, e, k);
    const states = energies.map(E => {
      const v = eigenvector(d, e, E);
      const s = 1 / Math.sqrt(h);
      let ref = 0;
      for (let i = 0; i < N; i++) { v[i] *= s; if (Math.abs(v[i]) > Math.abs(ref)) ref = v[i]; }
      // sign convention: first lobe (from the left) positive
      let first = 0;
      for (let i = 0; i < N; i++) if (Math.abs(v[i]) > 0.05 * Math.abs(ref)) { first = v[i]; break; }
      if (first < 0) for (let i = 0; i < N; i++) v[i] = -v[i];
      return v;
    });
    return { x, h, d, e, energies, states };
  }

  /**
   * Richardson-extrapolated solve: E = (4 E(h/2) - E(h)) / 3.
   * Eigenvectors are taken from the fine grid.
   */
  function solve(V, L, k, N = 799) {
    const coarse = solveGrid(V, L, N, k);
    const fine = solveGrid(V, L, 2 * N + 1, k);
    const energies = fine.energies.map((Ef, i) => (4 * Ef - coarse.energies[i]) / 3);
    return { x: fine.x, h: fine.h, energies, energiesFine: fine.energies, states: fine.states, d: fine.d, e: fine.e };
  }

  /** Linear interpolation of a sampled function (on solve()'s grid) at point xq. */
  function sample(sol, values, xq) {
    const L = -sol.x[0] + sol.h; // grid starts at -L + h
    const t = (xq + L) / sol.h - 1;
    if (t <= 0) return values[0] * Math.max(0, t + 1);
    if (t >= values.length - 1) return values[values.length - 1] * Math.max(0, values.length - t);
    const i = Math.floor(t), f = t - i;
    return values[i] * (1 - f) + values[i + 1] * f;
  }

  NQ.eigen = { sturmCount, lowestEigenvalues, thomas, eigenvector, solveGrid, solve, sample };
})(self.NQ = self.NQ || {});
