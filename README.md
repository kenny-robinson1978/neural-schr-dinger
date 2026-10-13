# Neural Schrödinger

**Neural networks that solve the Schrödinger equation from the variational principle, verified against an exact solver.**

The network is never shown a correct answer. Each wavefunction is a small neural network, trained only to minimise its energy, because the true ground state is the state of lowest energy. Excited states come from minimising the energy while staying orthogonal to every state below. Every result is graded live against an independent, high-accuracy eigensolver.

Pick a preset potential, or **draw your own with the mouse**, and watch the quantized energy levels appear.

## Results

3,000 training steps, seed 1, network 1→32→32→1 tanh per state. Fidelity is |⟨ψ_net|ψ_exact⟩|².

| Potential | State | Exact E | Network E | Rel. error | Fidelity |
|---|---|---|---|---|---|
| Harmonic ½x² | 0 | 0.50000 | 0.50001 | 0.002% | 100.000% |
| | 1 | 1.50000 | 1.50057 | 0.038% | 99.993% |
| | 2 | 2.50000 | 2.50030 | 0.012% | 99.997% |
| | 3 | 3.50000 | 3.50032 | 0.009% | 99.998% |
| Double well 0.2(x²−3)² | 0 | 0.93532 | 0.93553 | 0.023% | 99.999% |
| | 1 | 1.04851 | 1.04917 | 0.063% | 99.997% |
| | 2 | 2.30957 | 2.31275 | 0.137% | 99.975% |
| | 3 | 3.19052 | 3.19519 | 0.146% | 99.948% |
| Finite square well | 0 | 0.24631 | 0.24641 | 0.039% | 100.000% |
| | 1 | 0.97556 | 0.97749 | 0.198% | 99.990% |
| | 2 | 2.15571 | 2.15597 | 0.012% | 99.998% |
| Anharmonic ½x² + 0.15x⁴ | 0 | 0.58202 | 0.58215 | 0.022% | 99.999% |
| | 1 | 1.86624 | 1.86758 | 0.072% | 99.990% |
| | 2 | 3.35286 | 3.35568 | 0.084% | 99.986% |
| Morse 8(1−e^(−0.7(x+2)))² | 0 | 1.33875 | 1.33881 | 0.004% | 100.000% |
| | 1 | 3.64875 | 3.65005 | 0.036% | 99.993% |
| | 2 | 5.46875 | 5.48043 | 0.214% | 99.882% |

All 17 states land within 0.22% of the exact energy, with fidelity above 99.88%. Because of the variational principle, the ground-state estimates are upper bounds, and every one sits just above the exact value.

The double well's two lowest states are a **tunnelling pair** (splitting 0.113). The network resolves both the symmetric and the antisymmetric state, not a mixture localised in one well.

## Method

**Trial function.** State n is χₙ(x) = (1 − x²/L²) · Nₙ(x) on a box of half-width L = 6. The envelope enforces ψ(±L) = 0 for every set of weights.

**Loss = energy.** The Rayleigh quotient, with the kinetic term integrated by parts so only first derivatives are needed:

```
E[φ] = ( ½∫φ′² dx + ∫Vφ² dx ) / ∫φ² dx   ≥ E₀
```

**Excited states by Gram–Schmidt deflation.** State n is projected onto the complement of the already-orthonormalised lower states, which are held fixed, and its energy is minimised there:

```
φₙ = χₙ − Σ_{j<n} ⟨ψ̂ⱼ|χₙ⟩ ψ̂ⱼ
```

The gradient flows through the projection coefficients, so orthogonality is exact and there is no penalty weight to tune. An earlier version used an overlap penalty instead. Its excited states converged poorly (fidelity about 0.93 on the double well), and switching to projection fixed that.

**Quadrature.** The trapezoid rule on 120 interior points. Because every integrand vanishes at the walls, this is spectrally accurate for smooth functions (a Gaussian integrates to 2×10⁻¹⁶).

**Exact reference.** A three-point finite-difference Hamiltonian, which is symmetric tridiagonal.
- Eigenvalues come from Sturm-sequence bisection, which finds the k-th eigenvalue directly.
- Eigenvectors come from inverse iteration with the Thomas algorithm.
- Richardson extrapolation over grids of 799 and 1,599 points cancels the O(h²) error.

## Engineering

```
index.html            page
src/engine.js         autodiff for scalar-input MLPs (Taylor-mode forward, hand-derived reverse), Adam
src/solver.js         variational solver: envelope, quadrature, Rayleigh quotient, deflation + gradient
src/eigensolver.js    exact reference: Sturm bisection, inverse iteration, Richardson extrapolation
src/trainer.js        one training run: LR schedule, history, plain-data snapshots
src/worker.js         runs the trainer in a Web Worker
src/potentials.js     presets and user-drawn potentials
src/app.js            UI: rendering, controls, draw mode
tests/tests.js        12 tests
tests/index.html      test runner
```

* **No ML libraries and no build step.** The autodiff engine is about 180 lines. The kinetic energy only needs ψ′, so a first-order Taylor pass returns ψ and ψ′ together. Activations are cached per quadrature point, so each step runs every forward pass once.
* **Training runs off the main thread** in a Web Worker, so the UI stays smooth. If workers are unavailable (for example, when the page is opened from `file://`), the same `Trainer` runs on the main thread in small time slices.
* **Every run carries a generation number.** Snapshots from a superseded run, such as one from before you changed the potential, are discarded.
* **Deterministic.** Seeded PRNG, so a given seed reproduces the same training run.

### Tests

The fast tests run on every page load. Open `tests/index.html` to run all of them.

| Area | Test | Typical result |
|---|---|---|
| Engine | u′(x) vs finite differences | 2×10⁻¹⁰ |
| Engine | backprop vs finite differences | 3×10⁻¹² |
| Engine | Adam reaches a target | 7×10⁻¹⁴ |
| Exact solver | harmonic spectrum = n + ½ | 6×10⁻⁹ |
| Exact solver | infinite well = n²π²/8L² | 5×10⁻¹¹ |
| Exact solver | Morse = ω(n+½) − [ω(n+½)]²/4D | 8×10⁻⁷ |
| Exact solver | eigenvectors orthonormal | 2×10⁻¹³ |
| Exact solver | ‖Hψ − Eψ‖ residual | 2×10⁻¹² |
| Physics | trapezoid rule on a Gaussian | 2×10⁻¹⁶ |
| Physics | variational bound on 25 random networks | holds |
| Physics | gradient through Gram–Schmidt vs finite differences | 8×10⁻⁶ |
| Integration | trains the harmonic ground state | E₀ = 0.50019 |

Gradient checks use a fourth-order central difference. The relative-error denominator has a floor of 10⁻⁶ × the largest gradient component. Without the floor, saturated units with true gradients around 10⁻¹⁴ would be compared against finite-difference round-off and fail spuriously. I found this while debugging and verified it by checking that the error scales as h².

## Run it

Serve the folder with any static server and open `index.html`, for example:

```
python -m http.server
```

Opening `index.html` directly from disk also works. Training then runs on the main thread.

## References

* G. Carleo, M. Troyer, "Solving the quantum many-body problem with artificial neural networks," *Science* 355 (2017).
* D. Pfau, J. S. Spencer, A. G. D. G. Matthews, W. M. C. Foulkes, "Ab initio solution of the many-electron Schrödinger equation with deep neural networks," *Phys. Rev. Research* 2 (2020) (FermiNet).
* Units ħ = m = 1 throughout.
