/**
 * potentials.js — preset potentials (ħ = m = 1, box half-width L = 6).
 * Every potential is clipped at VMAX. Clipping only changes V where ψ is already
 * vanishingly small, and the network and the exact solver see the same clipped V.
 */
(function (NQ) {
  "use strict";
  const L = 6, VMAX = 40;
  const clip = f => x => Math.min(VMAX, f(x));

  const PRESETS = [
    {
      id: "harmonic", name: "Harmonic oscillator", formula: "V = ½x²",
      blurb: "The textbook case. Exact energies are ½, 3⁄2, 5⁄2, … so you can check the network by hand.",
      V: clip(x => 0.5 * x * x),
    },
    {
      id: "double", name: "Double well", formula: "V = 0.2(x² − 3)²",
      blurb: "Two wells separated by a barrier. The two lowest states form a pair split by quantum tunnelling: one symmetric, one antisymmetric.",
      V: clip(x => 0.2 * (x * x - 3) * (x * x - 3)),
    },
    {
      id: "well", name: "Finite square well", formula: "V = 3(1 + tanh((|x| − 2)/0.15))",
      blurb: "A flat well with steep walls of finite height. The wavefunctions leak into the walls, which classical physics forbids.",
      V: clip(x => 3 * (1 + Math.tanh((Math.abs(x) - 2) / 0.15))),
    },
    {
      id: "anharmonic", name: "Anharmonic", formula: "V = ½x² + 0.15x⁴",
      blurb: "A stiffened spring. Levels spread apart as energy rises, unlike the evenly spaced harmonic ladder.",
      V: clip(x => 0.5 * x * x + 0.15 * x ** 4),
    },
    {
      id: "morse", name: "Morse (molecular bond)", formula: "V = 8(1 − e^(−0.7(x+2)))²",
      blurb: "A model of a chemical bond. Asymmetric, with levels crowding together toward the dissociation energy.",
      V: clip(x => 8 * (1 - Math.exp(-0.7 * (x + 2))) ** 2),
    },
  ];

  /** Builds a potential function from samples drawn by the user on a uniform grid over [-L, L]. */
  function fromSamples(samples) {
    const n = samples.length;
    return x => {
      const t = (x + L) / (2 * L) * (n - 1);
      const i = Math.max(0, Math.min(n - 2, Math.floor(t))), f = t - i;
      return Math.min(VMAX, samples[i] * (1 - f) + samples[i + 1] * f);
    };
  }

  NQ.potentials = { L, VMAX, PRESETS, fromSamples };
})(self.NQ = self.NQ || {});
