# InfluenceFlow
Normalizing flows using influence functions. Code to accompany the paper *Recovering weak signals with normalizing flows* (under review).

## Novelty and Significance

This work introduces a principled framework for weak signal recovery by coupling **influence functions** from classical statistics with **normalizing flows** from deep generative modeling. While normalizing flows excel at learning complex posterior distributions, they often require extensive labeled data and can drift toward low-probability regions. InfluenceFlow grounds neural flows in the algebraic structure of inverse problems by using influence function Jacobians to guide training—effectively leveraging the geometric insights of classical methods (elastic net regression, radio telecope calibration) while gaining the expressive power of neural generative models. The resulting flow learns the full posterior distribution of hidden signals conditioned on noisy observations, enabling detection of faint signals below classical thresholds. By enforcing signal-residual alignment constraints via ADMM and training on influence-corrected targets, the approach combines the stability of traditional solvers with the flexibility of deep learning, demonstrating broad applicability across linear regression and radio interferometry domains.

## Linear model
See directory './linear'

## Radio interferometric model
See directory './radio'

vr  4 sep 2026 10:14:16 CEST
