# Handshake showcase entry (copy and paste)

## Title
Neural Schrödinger: Neural Networks That Solve Quantum Mechanics

## One-line summary
An interactive app where neural networks discover quantum wavefunctions and energy levels using only the variational principle. The results match an exact solver to within 0.22%, and you can draw any potential with your mouse.

## Description
Can a neural network solve the Schrödinger equation without ever seeing a correct answer? I built Neural Schrödinger to find out.

Each wavefunction is a small neural network trained on one law of physics: the true ground state has the lowest possible energy. The loss function is the energy itself. Excited states are found by minimising energy while staying exactly orthogonal to the states below, using a Gram–Schmidt projection that gradients flow through. This is the same idea behind neural quantum state methods such as DeepMind's FermiNet, built from scratch in one dimension.

**Results.** Across five potentials (harmonic, double well, finite well, anharmonic and Morse), all 17 computed states land within 0.22% of the exact energy, with wavefunction fidelity above 99.88%. In the double well, the network correctly resolves the tunnelling pair: a symmetric state and an antisymmetric state split by quantum tunnelling.

**Engineering highlights**
- Autodiff engine written from scratch, with no ML libraries: Taylor-mode forward pass and hand-derived backpropagation.
- An independent exact eigensolver to grade the network: Sturm-sequence bisection, inverse iteration and Richardson extrapolation. It matches the analytic harmonic, infinite-well and Morse spectra to 10⁻⁶ or better.
- Training runs in a Web Worker so the UI stays smooth, with automatic fallback to the main thread.
- A 12-test suite runs on page load: gradient checks against finite differences, analytic spectra, orthonormality, the variational bound and a training integration test.
- Engineering judgement: I replaced an overlap-penalty method with exact projection after finding that excited-state fidelity stalled at about 0.93. I also diagnosed a spurious gradient-check failure as finite-difference truncation error by showing that it scales as h².

## Skills to tag
Quantum Mechanics · Machine Learning · Neural Networks · Scientific Computing · Numerical Linear Algebra · JavaScript · Software Testing · Optimization · Data Visualization

## Links
- Live demo: <paste your shared artifact link>
- Code: <paste your GitHub repo link>

---

## Short version (if there's a character limit)
Built Neural Schrödinger, an interactive app where neural networks solve the Schrödinger equation using only the variational principle, with no training data. Across five potentials, all 17 states match an exact eigensolver within 0.22% (fidelity > 99.88%), including a resolved tunnelling pair. From-scratch autodiff, Web Worker training, and a 12-test verification suite.
