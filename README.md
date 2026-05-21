# Radiation Event Classification from Oscilloscope Trace Parameters

Classifying radiation events (alphas, betas, and candidate protons) from the
shape of their digitized detector pulses, using only the parameters of an
analytic pulse-fit — no energy information.

---

## Motivation

In a Micromegas-style ionization detector, different particles produce
characteristically different current pulses. Alpha particles deposit their
energy densely over a short track and produce fast, sharply-peaked signals;
beta particles spread their energy along longer, more variable trajectories
and produce slower, broader pulses; protons sit somewhere in between, both
in deposited energy and in ionization density.

The textbook way to separate these populations is by deposited energy,
because the energy distributions are well-separated for most sources. But
energy alone fails in two important regimes: (1) when a third particle
species (e.g. protons from a contaminating source) sits between the alpha
and beta energy bands, and (2) when calibration drift smears the energy
spectrum.

This project asks whether the **pulse shape alone** — independent of
amplitude or deposited energy — carries enough information to identify
particle type, and whether shape-based classification can flag events that
don't belong to either training class.

## Approach

Pulses are fit to a piecewise erf-rise / exponential-decay function

$$
f(x) = \begin{cases}
  C + A\,\mathrm{erf}\!\left(\dfrac{x-\mu}{\sigma}\right) & x < \mu + 2\sigma \\[4pt]
  (A+C)\,\exp\!\left(-\dfrac{x-\mu-2\sigma}{\tau}\right) & x \ge \mu + 2\sigma
\end{cases}
$$

producing five fit parameters per event: baseline $C$, half-amplitude $A$,
center time $\mu$, rise width $\sigma$, and decay constant $\tau$.

A correlation analysis shows that $A$, $C$, and the derived rise rate all
carry > 96% Pearson correlation with deposited energy — they are effectively
amplitude features, and using them would just re-derive the energy threshold
that defined the training labels in the first place. The genuine
shape-discriminating features are $\mu$, $\log\sigma$, and $\log\tau$ (log
transforms because $\sigma$ and $\tau$ each span more than an order of
magnitude).

Classification proceeds in two stages:

1. **Supervised binary classifier.** A scikit-learn pipeline with feature
   scaling and a choice of estimators (logistic regression, kNN, SVM,
   decision tree, random forest, gradient boosting), trained on labeled
   alpha/beta events with stratified 5-fold cross-validation and grid
   search over hyperparameters. Balanced accuracy is the scoring metric to
   handle the mild class imbalance.

2. **Out-of-distribution detection.** A class-conditional novelty detector
   built from per-class Gaussian Mixture Models plus a global Isolation
   Forest. Events whose shape is anomalous relative to their predicted
   class's training distribution are flagged as candidates for a third
   particle population. Thresholds are set at the 5th/95th percentile of
   training scores, giving a 5% false-OOD rate by construction.

## Results

Shape-based binary classification of alphas vs betas is essentially perfect
(CV balanced accuracy ≥ 99% across all tested estimators) — confirming that
deposited energy is not required to separate the two classes.

Applying the trained pipeline to a held-out sample of intermediate-energy
events surfaces direct evidence for a third population: classifier
confidence drops sharply, distribution tests reject the hypothesis that the
events come from the training distribution, and unsupervised model selection
(GMM BIC) prefers three components over two on the test set alone. See
`Proton_Search.ipynb` for the full analysis and caveats.

## Repository contents

| File | Description |
|------|-------------|
| `FinalProject_cleaned.ipynb` | End-to-end pipeline: data loading, energy-leak audit, classifier comparison, permutation importance, PCA, OOD detection |
| `Proton_Search.ipynb` | Application of the trained pipeline to intermediate-energy events; statistical and unsupervised evidence for a third population |
| `traceDataMoreTrainEntries.txt` | Labeled training set (594 events: 365 betas, 229 alphas) |
| `traceDataMoreTestEntries.txt` | Unlabeled intermediate-energy test sample (93 events) |
| `test_event_classifications.csv` | Per-event classifier output, OOD flags, and unsupervised cluster assignments |

## Data format

Whitespace-separated, no header, ten columns:

```
Entry  Class  Occurrence  Energy  C  A  mu  sigma  tau  rise_rate
```

- `Class`: 1 = beta, 2 = alpha, 3 = intermediate-energy (unlabeled) test events
- `Energy`: from the Micromegas detector readout
- `C, A, mu, sigma, tau`: pulse-fit parameters (see Approach)
- `rise_rate`: derived feature, $A/\sigma$

## Running it

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
jupyter notebook FinalProject_cleaned.ipynb
```

Notebooks are reproducible — all random seeds are pinned to `RNG = 42`.
Both `.ipynb` files include cached outputs, so they can be read end-to-end
without re-execution.

## Methodology notes

A few design choices worth flagging for readers familiar with the area:

- **Feature scaling lives inside the cross-validation pipeline.** Without
  it, distance-based classifiers (kNN, SVM) are dominated by $\tau$, which
  spans values in the tens of thousands while $\sigma$ stays in the tens.
  Putting the scaler inside the `Pipeline` also prevents the test fold from
  leaking into the standardization step during CV.
- **Log-transforming $\sigma$ and $\tau$** is what makes logistic regression
  competitive with tree ensembles on this problem. In linear space the two
  classes are not linearly separable; in log space they are.
- **The OOD detector is class-conditional.** A global anomaly threshold
  would preferentially flag betas (which form a diffuse cluster) and miss
  proton candidates that happen to look superficially alpha-like. Scoring
  each event against its *predicted* class's training distribution avoids
  this bias.
- **Distribution-level tests (KS, Mahalanobis) and unsupervised model
  selection (GMM BIC)** are reported alongside the OOD flags, because no
  single detector should be trusted for a discovery claim. The strongest
  evidence in this analysis is the BIC preference for three components on
  the test set, which doesn't depend on any classifier or threshold.

## Limitations and next steps

- All analysis runs on pulse-fit *parameters*. The piecewise fit imposes a
  specific functional form (continuous in value but not in slope at the
  join point); pulses that don't match this form are summarized poorly.
  Working from raw traces would remove this constraint.
- Sample sizes are modest — 594 training events and 93 test events. The
  unsupervised three-component preference is statistically strong (ΔBIC of
  113), but the *identity* of the third population should be confirmed by
  inspecting raw traces of the OOD-flagged events directly.
- A coherent population shift in $\tau$ is consistent with either a new
  particle class or calibration drift between training and test data
  acquisition. Distinguishing these would require timestamps or external
  calibration information.
- Planned extension: train a 1D CNN directly on baseline-normalized raw
  traces, using the fit-parameter classifier as a baseline. With ~600
  events this is borderline; k-fold CV and heavy augmentation (time
  shifts, additive noise) will be essential.
