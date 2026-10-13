/**
 * engine.js — a small automatic-differentiation engine for scalar-input MLPs.
 *
 * The forward pass propagates first-order Taylor coefficients (value and d/dx)
 * through every layer, so one call returns both u(x) and u'(x). The backward pass
 * is reverse-mode differentiation through that Taylor pass, derived by hand:
 *
 *   forward  (tanh layer):  a = tanh z,  s = 1 - a²,  a' = s z'
 *   backward (tanh layer):  ḡz  = ḡa s - 2 a s z' ḡa'
 *                           ḡz' = ḡa' s
 *
 * Activations live in an "acts" bank. Callers can keep one bank per input point,
 * so a training step can run every forward pass first and every backward pass later
 * without recomputing anything.
 */
(function (NQ) {
  "use strict";

  /** Deterministic PRNG (mulberry32). Returns a function producing floats in [0, 1). */
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Standard normal sample via Box–Muller. */
  function gauss(r) {
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Allocates an activation bank for a network with the given layer sizes. */
  function makeActs(sizes) {
    return sizes.map(s => ({ a: new Float64Array(s), a1: new Float64Array(s), z1: new Float64Array(s) }));
  }

  /**
   * Creates a tanh MLP. All parameters live in one flat Float64Array so the
   * optimiser and gradient checks can treat them uniformly.
   * @param {number[]} sizes  e.g. [1, 32, 32, 1]
   * @param {number} seed
   * @param {{scale?: number, shift?: number}} opts  input map x -> scale*x + shift
   */
  function makeNet(sizes, seed, opts = {}) {
    const layers = [];
    let n = 0;
    for (let l = 0; l < sizes.length - 1; l++) {
      const fin = sizes[l], fout = sizes[l + 1];
      layers.push({ fin, fout, w: n, b: n + fin * fout });
      n += fin * fout + fout;
    }
    const net = {
      sizes, layers, n,
      p: new Float64Array(n), g: new Float64Array(n),
      m: new Float64Array(n), v: new Float64Array(n), t: 0,
      scale: opts.scale ?? 1, shift: opts.shift ?? 0,
      acts: makeActs(sizes),
      grads: sizes.map(s => ({ g0: new Float64Array(s), g1: new Float64Array(s) })),
    };
    const r = rng(seed);
    for (const L of layers) {
      const std = Math.sqrt(2 / (L.fin + L.fout)); // Xavier / Glorot normal
      for (let i = 0; i < L.fin * L.fout; i++) net.p[L.w + i] = std * gauss(r);
    }
    return net;
  }

  /**
   * Forward pass with first-order Taylor propagation.
   * Writes activations into `acts` and returns [u, du/dx].
   */
  function forward(net, x, acts = net.acts) {
    const { p, layers } = net;
    const nL = layers.length;
    acts[0].a[0] = net.scale * x + net.shift;
    acts[0].a1[0] = net.scale;
    for (let l = 0; l < nL; l++) {
      const L = layers[l], inp = acts[l], out = acts[l + 1];
      const ia = inp.a, ia1 = inp.a1, fin = L.fin, last = l === nL - 1;
      for (let j = 0; j < L.fout; j++) {
        let s0 = p[L.b + j], s1 = 0;
        const row = L.w + j * fin;
        for (let i = 0; i < fin; i++) {
          const w = p[row + i];
          s0 += w * ia[i];
          s1 += w * ia1[i];
        }
        if (last) {
          out.a[j] = s0; out.a1[j] = s1;
        } else {
          const th = Math.tanh(s0), s = 1 - th * th;
          out.z1[j] = s1;
          out.a[j] = th;
          out.a1[j] = s * s1;
        }
      }
    }
    const o = acts[nL];
    return [o.a[0], o.a1[0]];
  }

  /** Value-only forward pass, for plotting and finite-difference checks. */
  function predict(net, x) {
    const { p, layers, acts } = net;
    const nL = layers.length;
    acts[0].a[0] = net.scale * x + net.shift;
    for (let l = 0; l < nL; l++) {
      const L = layers[l], ia = acts[l].a, oa = acts[l + 1].a, fin = L.fin, last = l === nL - 1;
      for (let j = 0; j < L.fout; j++) {
        let s0 = p[L.b + j];
        const row = L.w + j * fin;
        for (let i = 0; i < fin; i++) s0 += p[row + i] * ia[i];
        oa[j] = last ? s0 : Math.tanh(s0);
      }
    }
    return acts[nL].a[0];
  }

  /**
   * Reverse pass. Given dLoss/du and dLoss/du' at one input point (whose activations
   * are in `acts`), accumulates dLoss/dθ into net.g.
   */
  function backward(net, gu, gu1, acts = net.acts) {
    const { p, g, layers, grads } = net;
    const nL = layers.length;
    let G = grads[nL];
    G.g0[0] = gu;
    G.g1[0] = gu1;
    for (let l = nL - 1; l >= 0; l--) {
      const L = layers[l], inp = acts[l], fin = L.fin, fout = L.fout;
      const ia = inp.a, ia1 = inp.a1, g0 = G.g0, g1 = G.g1;
      for (let j = 0; j < fout; j++) {
        const c0 = g0[j], c1 = g1[j];
        g[L.b + j] += c0;
        const row = L.w + j * fin;
        for (let i = 0; i < fin; i++) g[row + i] += c0 * ia[i] + c1 * ia1[i];
      }
      if (l === 0) break;
      const P = grads[l], h0 = P.g0, h1 = P.g1;
      for (let i = 0; i < fin; i++) {
        let b0 = 0, b1 = 0;
        for (let j = 0; j < fout; j++) {
          const w = p[L.w + j * fin + i];
          b0 += w * g0[j];
          b1 += w * g1[j];
        }
        const th = ia[i], s = 1 - th * th;
        h0[i] = b0 * s - 2 * th * s * inp.z1[i] * b1;
        h1[i] = b1 * s;
      }
      G = P;
    }
  }

  function zeroGrad(net) { net.g.fill(0); }

  /** One Adam update (Kingma & Ba, 2015) using the gradient currently in net.g. */
  function adam(net, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    net.t++;
    const bc1 = 1 - Math.pow(b1, net.t), bc2 = 1 - Math.pow(b2, net.t);
    const { p, g, m, v } = net;
    for (let i = 0; i < net.n; i++) {
      const gi = g[i];
      m[i] = b1 * m[i] + (1 - b1) * gi;
      v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      p[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
    }
  }

  NQ.engine = { rng, gauss, makeActs, makeNet, forward, predict, backward, zeroGrad, adam };
})(self.NQ = self.NQ || {});
