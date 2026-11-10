# Core Research Direction: The Diagnostic Stress-Suite Protocol

**Working Title:** *The Illusion of In-Distribution Accuracy: Probing Industry-Standard Copolymer Machine Learning Models Under Fixed-Training Stress Suites*  
**Primary Venue Target:** *ACS Journal of Chemical Information and Modeling (JCIM)* / *RSC Digital Discovery*

---

## 1. The Core Scientific Argument

In both industrial formulation and academic literature, machine learning models for copolymer property prediction are almost universally trained and evaluated under a single paradigm:
1. Collect all available copolymer data points.
2. Randomly split them into 80% train and 20% test (IID split).
3. Train the model once, observe a high test score ($R^2 \approx 0.90 - 0.98$, $\text{MAE} < 5\text{ K}$), and declare the model validated for deployment.

**Our Core Finding:**  
This evaluation protocol conceals a critical failure mode. When the **exact same trained model**—frozen after its successful random evaluation—is probed with targeted, stress-testing test suites (evaluating ratio extrapolation, one-sided data starvation, or equimolar gap interpolation), models that looked equally accurate on the random split diverge dramatically.

---

## 2. Experimental Architecture: "Train Once, Probe Multiple Suites"

Unlike previous setups where models were retrained on different subsets, here **training is executed exactly once** for each model candidate. This guarantees that differences in test performance reflect **inherent architectural inductive biases**, not artifacts of different training data sizes or compositions.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   UNIFIED TRAINING SET (D_train)                       │
│  - General Monomer Pairs: Full natural composition coverage            │
│  - Pair Group 1 (Core-Only): Censored to f ∈ [0.25, 0.75]              │
│  - Pair Group 2 (One-Sided): Censored to f ≤ 0.50                      │
│  - Pair Group 3 (Tails-Only): Censored to f < 0.25 and f > 0.75        │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                     Train Each Model Paradigm ONCE
                                     │
                                     ▼
                     FROZEN TRAINED MODEL WEIGHTS
                                     │
        ┌────────────────────────────┼───────────────────────────┐
        ▼                            ▼                           ▼
 ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
 │ Suite 1 (IID) │           │ Suite 2 (OOD) │           │ Suite 3 (OOD) │ ...
 │ Random Test   │           │ Core-to-Tail  │           │ One-Sided     │
 │ Holdout       │           │ Extrapolation │           │ Extrapolation │
 └───────────────┘           └───────────────┘           └───────────────┘
```

### The 4 Evaluation Test Suites (Evaluated on the Same Trained Models):
1. **Suite 1: Standard Random Holdout (The Baseline Benchmark)**  
   - Held-out points randomly sampled across general pairs.  
   - *Purpose:* Demonstrates the "industry baseline" where all candidate models achieve comparable, high $R^2$ scores.
2. **Suite 2: Core-to-Tail Extrapolation Probe**  
   - The censored tail compositions ($f < 0.25$ and $f > 0.75$) of **Pair Group 1**.  
   - *Question asked:* The model saw this pair's center, but never its tails. Does it extrapolate cleanly or collapse?
3. **Suite 3: One-Sided Asymmetric Extrapolation Probe**  
   - The censored upper compositions ($f > 0.50$) of **Pair Group 2**.  
   - *Question asked:* The model saw the A-rich side. Can it predict the B-rich side?
4. **Suite 4: Equimolar Gap / Synergy Interpolation Probe**  
   - The censored core compositions ($f \in [0.25, 0.75]$) of **Pair Group 3**.  
   - *Question asked:* The model saw the boundaries. Can it capture the nonlinear excess mixing in the middle?

---

## 3. Model Candidates: Industry and Literature Paradigms

Rather than arbitrary network variants, candidate models are chosen to directly represent the primary paradigms currently deployed in industry and published in literature:

### Paradigm 1: Tabular Industry Workhorses (Tree-Based on Chemical Descriptors)
* **Random Forest (RF)**
* **Gradient Boosted Trees (XGBoost / LightGBM)**
  - *Features:* Molecular descriptors computed via RDKit (molecular weight, polar surface area, hydrogen bond counts, rotatable bonds, partial charges) for each monomer + blend fraction $f$.
  - *Role:* Represents >80% of current industrial cheminformatics deployment. Fast, non-deep-learning baseline.

### Paradigm 2: Polymer Language Models (Sequence / String Paradigm)
* **Pretrained Polymer Transformer (PolyBERT / ChemBERTa on BigSMILES)**
  - *Features:* Copolymer BigSMILES strings mapped through pretrained macromolecular language models + MLP regression head.
  - *Role:* Represents the state-of-the-art "foundation model / chemical LLM" academic trend.

### Paradigm 3: Deep Neural Networks on Embeddings (Vector Operators)
* **Unconstrained Feature Concatenation MLP (Naive Neural Network)**
  - *Input:* $[e_A \parallel e_B \parallel f]$
  - *Role:* The default deep learning baseline for multi-component formulation.
* **Latent Convex Combination Network (Linear Geometric Manifold)**
  - *Input:* $f z_A + (1-f) z_B$
  - *Role:* Evaluates the pure linear mixing inductive bias in embedding space.
* **Thermodynamic Hybrid Network (Physics-Informed Architecture)**
  - *Input:* Gordon-Taylor free-volume core + bounded Margules interaction envelope.
  - *Role:* Evaluates whether embedding physical mixing boundaries into neural representations prevents OOD breakdown.

### Paradigm 4: Classical Chemical Physics (Analytical Reference)
* **Gordon-Taylor & Fox Equations**
  - *Role:* Ground truth theoretical benchmark. Reveals whether multi-parameter ML actually outperforms 70-year-old classical thermodynamic theory.

---

## 4. Expected Deliverables & Reviewer Impact

1. **A Single Comparative Matrix:** One primary table showing each industry paradigm's performance across Suite 1 (Random), Suite 2 (Core-to-Tail), Suite 3 (One-Sided), and Suite 4 (Gap).
2. **The Divergence Finding:** Clear visual proof that models tied on Suite 1 diverge widely on Suites 2–4.
3. **Actionable Guidance:** Concrete evidence showing experimental polymer chemists which popular models are safe to deploy for composition optimization, and which models carry hidden extrapolation risks.
