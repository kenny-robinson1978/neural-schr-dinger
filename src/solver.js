/**
 * solver.js — neural variational solver for the 1D time-independent Schrödinger equation
 * (units ħ = m = 1):   -½ ψ''(x) + V(x) ψ(x) = E ψ(x),   ψ(±L) = 0.
 *
 * Each state n is a separate network N_n. The raw trial function is
 *
 *   χ_n(x) = (1 - x²/L²) · N_n(x)
 *
 * where the envelope enforces the boundary conditions exactly.
 *
 * Excited states use Gram–Schmidt deflation. The state is projected onto the subspace
 * orthogonal to the (already orthonormalised) lower states ψ̂_j, which are held fixed:
 *
 *   φ_n = χ_n − Σ_{j<n} c_j ψ̂_j,   c_j = ⟨ψ̂_j|χ_n⟩
 *
 * Training minimises the Rayleigh quotient of φ_n (the variational energy). The kinetic term is
 * used in its integrated-by-parts form, so only first derivatives are needed:
 *
 *   E[φ] = ( ½∫φ'² + ∫Vφ² ) / ∫φ²
 *
 * For n = 0 this is the variational principle, E[φ] ≥ E_0. For n > 0, minimising inside the
 * orthogonal complement gives E_n. The gradient flows through the projection coefficients c_j,
 * so the result is exact, with no penalty weight to tune. Integrals use the trapezoid rule on
 * interior points (the envelope makes the endpoint terms vanish).
 */
(function (NQ) {
  "use strict";
  const E = NQ.engine;

  class VariationalSolver {
    /**
     * @param {object} cfg
     * @param {(x:number)=>number} cfg.V   potential
     * @param {number} cfg.L               half-width of the box
     * @param {number} [cfg.nStates=3]
     * @param {number} [cfg.M=140]         quadrature points
     * @param {number[]} [cfg.hidden=[24,24]]
     * @param {number} [cfg.inputScale=0.5] network sees inputScale·x
     * @param {number} [cfg.seed=1]
     */
    constructor(cfg) {
      const { V, L, nStates = 3, M = 140, hidden = [24, 24], inputScale = 0.5, seed = 1 } = cfg;
      this.L = L; this.M = M; this.nStates = nStates;
      this.h = 2 * L / (M + 1);
      this.x = new Float64Array(M);
      this.V = new Float64Array(M);
      this.env = new Float64Array(M);   // envelope g(x)
      this.denv = new Float64Array(M);  // g'(x)
      for (let i = 0; i < M; i++) {
        const x = -L + (i + 1) * this.h;
        this.x[i] = x;
        this.V[i] = V(x);
        this.env[i] = 1 - (x * x) / (L * L);
        this.denv[i] = -2 * x / (L * L);
      }
      const sizes = [1, ...hidden, 1];
      const arr = () => Array.from({ length: nStates }, () => new Float64Array(M));
      this.nets = Array.from({ length: nStates }, (_, k) => E.makeNet(sizes, seed + 7919 * k, { scale: inputScale }));
      this.banks = this.nets.map(() => Array.from({ length: M }, () => E.makeActs(sizes)));
      this.chi = arr(); this.dchi = arr();   // raw network trial functions
      this.phi = arr(); this.dphi = arr();   // after projection
      this.hat = arr(); this.dhat = arr();   // normalised φ (the reported states)
      this.coef = [];                        // projection coefficients per state
      this.step = 0;
      this.stats = null;
    }

    /** Forward pass for state k at every quadrature point; caches activations. */
    evaluate(k) {
      const net = this.nets[k], bank = this.banks[k], chi = this.chi[k], dchi = this.dchi[k];
      for (let i = 0; i < this.M; i++) {
        const [n, n1] = E.forward(net, this.x[i], bank[i]);
        chi[i] = this.env[i] * n;
        dchi[i] = this.denv[i] * n + this.env[i] * n1;
      }
    }

    /** Projects χ_k against ψ̂_0..ψ̂_{k-1}, computes integrals, and stores ψ̂_k. */
    project(k) {
      const M = this.M, w = this.h;
      const chi = this.chi[k], dchi = this.dchi[k], phi = this.phi[k], dphi = this.dphi[k];
      phi.set(chi); dphi.set(dchi);
      const c = [];
      for (let j = 0; j < k; j++) {
        const hj = this.hat[j], dhj = this.dhat[j];
        let cj = 0;
        for (let i = 0; i < M; i++) cj += w * hj[i] * chi[i];
        for (let i = 0; i < M; i++) { phi[i] -= cj * hj[i]; dphi[i] -= cj * dhj[i]; }
        c.push(cj);
      }
      this.coef[k] = c;
      let D = 0, T = 0, U = 0;
      for (let i = 0; i < M; i++) {
        const p = phi[i];
        D += w * p * p;
        T += 0.5 * w * dphi[i] * dphi[i];
        U += w * this.V[i] * p * p;
      }
      const s = 1 / Math.sqrt(D);
      for (let i = 0; i < M; i++) { this.hat[k][i] = phi[i] * s; this.dhat[k][i] = dphi[i] * s; }
      return { D, T, U, E: (T + U) / D };
    }

    /** Computes ∂E[φ_k]/∂θ_k into nets[k].g. Requires evaluate(k) and project(0..k). */
    gradient(k, Ik) {
      const M = this.M, w = this.h, V = this.V;
      const phi = this.phi[k], dphi = this.dphi[k];
      const { D, E: Ek } = Ik;
      // dE/dφ_i and dE/dφ'_i
      const gPhi = new Float64Array(M), gDphi = new Float64Array(M);
      for (let i = 0; i < M; i++) {
        gPhi[i] = 2 * w * (V[i] - Ek) * phi[i] / D;
        gDphi[i] = w * dphi[i] / D;
      }
      // dE/dc_j = −Σ_i (gPhi ψ̂_j + gDphi ψ̂'_j); then dc_j/dχ_i = w ψ̂_j,i
      const gChi = Float64Array.from(gPhi);
      for (let j = 0; j < k; j++) {
        const hj = this.hat[j], dhj = this.dhat[j];
        let gc = 0;
        for (let i = 0; i < M; i++) gc -= gPhi[i] * hj[i] + gDphi[i] * dhj[i];
        for (let i = 0; i < M; i++) gChi[i] += gc * w * hj[i];
      }
      const net = this.nets[k], bank = this.banks[k];
      E.zeroGrad(net);
      for (let i = 0; i < M; i++) {
        // chain rule through χ = g·N, χ' = g'·N + g·N'   (dE/dχ'_i = gDphi_i)
        E.backward(net, gChi[i] * this.env[i] + gDphi[i] * this.denv[i], gDphi[i] * this.env[i], bank[i]);
      }
    }

    /** One optimisation step for every state. Returns per-state integrals and energies. */
    train(lr) {
      const I = [];
      for (let k = 0; k < this.nStates; k++) {
        this.evaluate(k);
        I.push(this.project(k));
        this.gradient(k, I[k]);
      }
      for (let k = 0; k < this.nStates; k++) E.adam(this.nets[k], lr);
      this.step++;
      this.stats = I;
      return I;
    }

    /** Energy of state k from the current parameters, lower states held as they are. */
    loss(k) {
      for (let j = 0; j <= k; j++) this.evaluate(j);
      let Ik;
      for (let j = 0; j <= k; j++) Ik = this.project(j);
      return Ik.E;
    }

    /** Normalised state k on the quadrature grid, as of the last training step. */
    normalized(k) { return Float64Array.from(this.hat[k]); }
  }

  /**
   * Compares the network's states with an exact solution from NQ.eigen.solve.
   * Returns energy errors and fidelities |⟨ψ_net|ψ_exact⟩|². Aligns signs for plotting.
   */
  function compare(solver, exact) {
    const out = [];
    for (let k = 0; k < solver.nStates; k++) {
      const psi = solver.normalized(k);
      const ref = new Float64Array(solver.M);
      let ov = 0, nr = 0;
      for (let i = 0; i < solver.M; i++) {
        ref[i] = NQ.eigen.sample(exact, exact.states[k], solver.x[i]);
        ov += solver.h * psi[i] * ref[i];
        nr += solver.h * ref[i] * ref[i];
      }
      const fidelity = (ov * ov) / nr;
      if (ov < 0) for (let i = 0; i < solver.M; i++) psi[i] = -psi[i];
      const En = solver.stats ? solver.stats[k].E : solver.loss(k);
      out.push({ n: k, E: En, Eexact: exact.energies[k], err: Math.abs(En - exact.energies[k]), fidelity, psi, ref });
    }
    return out;
  }

  NQ.VariationalSolver = VariationalSolver;
  NQ.compare = compare;
})(self.NQ = self.NQ || {});
