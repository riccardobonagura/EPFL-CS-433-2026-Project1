# Project 1 Bible

## 0. Executive summary

The project predicts `_MICHD` from a large BRFSS health-survey dataset using only Python, NumPy, and visualization libraries.

The diagram below describes the current workflow and dependencies, not a verified chronology of development. Implementation claims are grounded in the linked source; numerical results refer to saved local runs checked against that source. Explanations of motivation are retrospective interpretations, and proposed improvements are hypotheses rather than measured outcomes.

```text
Raw survey data
      ↓
Exploratory analysis
      ↓
Mandatory-model diagnostics
      ↓
Augmented optimization/model toolbox
      ↓
Reusable feature-processing system
      ↓
Processing ablations
      ↓
Three experiment directions / representations
      ↓
3-fold confirmation of three finalists
      ↓
Full-data refit + submissions
```

The most important empirical findings were:

1. **The problem is strongly imbalanced.** Only 8.83% of training examples are positive, making 91.17% raw accuracy achievable by predicting every subject as negative. Balanced accuracy, recall and F1 therefore became the meaningful local metrics.

2. **The raw feature matrix is not a conventional continuous matrix.** Most predictors behave as categorical survey variables, missingness is extensive and structured, many numeric values are survey codes, and several raw/derived variables are redundant.

3. **Missingness is associated with the target.** 133 predictors met the EDA's descriptive flag: at least 100 missing and 100 observed rows, with an absolute positive-rate difference of at least 0.03. This is not a significance test or a causal finding.

4. **Numeric storage alone does not establish survey semantics, and conservative processing is not automatically more predictive.** The processing ablation found the minimally transformed A0/A1 representations tied under the fixed diagnostic logistic configuration: selected-threshold balanced accuracy 0.7671 versus 0.7618 for A6. This does not identify which individual transformation caused the difference.

5. **Regularized logistic regression emerged as the strongest course-family baseline.** On the original crude numerical diagnostic representation, regularized logistic regression reached balanced accuracy 0.7170, versus 0.6367 for unregularized logistic, 0.6175 for ridge and 0.5000 for the majority classifier.

6. **Optimization loss and classification quality can diverge sharply.** For MSE-SGD, increasing updates from 1,000 to 100,000 reduced MSE substantially while balanced accuracy collapsed from 0.7144 to 0.5194. Similar minority-recall collapse occurred for full-batch GD and logistic regression at fixed thresholds. These observations provide a rationale for the augmented monitoring, stopping and threshold-selection facilities, but do not prove that loss-based early stopping improves BA.

7. **The three finalists produced the following measured results.** After 3-fold confirmation:

| Strategy | Representation | Exploratory BA | 3-fold BA | Fold SD | 3-fold F1 |
|---|---|---:|---:|---:|---:|
| Selective target-encoded L2 logistic | TE2 selective recovery | 0.7791 | **0.7757** | 0.0019 | 0.3532 |
| Curated L2 logistic | R2 curated | 0.7779 | 0.7745 | 0.0020 | 0.3521 |
| Early-stopped MSE-SGD | S0 / R2 curated | 0.7750 | 0.7710 | **0.0006** | 0.3525 |



The target-encoded logistic model has the **highest observed mean**. The code ranks the three finalists and exports all three; it does not designate one canonical primary model. Its 0.00125 BA lead over curated logistic (rounded to five decimals) is smaller than the individual fold SDs, which is descriptive context rather than a paired significance test. Recipe selection overlapped the confirmation data, so these results are conditional confirmation, not an independent estimate of the entire selection process.

---

# 1. How exploratory analysis was conducted

## 1.1 Goals

The [EDA implementation](<Exploratory Analysis/eda.py>) characterizes the raw matrix and generates preprocessing/model suggestions. Its outputs address two practical questions:

1. **How must the raw survey variables be transformed before modeling?**
2. **Which model families are plausible under the NumPy-only constraint?**

The EDA script characterizes the raw representation without recoding values or fitting a classifier. Its three recommendations are L2 logistic with encoding/missingness, class-weighted nonlinear L2 logistic, and ridge classification (`eda.STRATEGIES`); these differ from the three implemented experiment directions. The EDA recommendations are not evidence that class weighting or final ridge experiments were executed.

---

## 1.2 Dataset integrity

The EDA operated on:

```text
328,135 training observations
321 predictors excluding Id
```

ID uniqueness, duplicate IDs, target IDs, matching ID sets and exact row ordering were checked before any target association calculations.

This was important because a silent row mismatch would invalidate every subsequent association and model result.

---

## 1.3 Reproducibility of EDA

The EDA used seed:

```text
433
```

Full-data calculations were used for:

- feature statistics;
- percentiles;
- category rates;
- target summaries;
- exact duplicate checks;
- missingness analysis.

For more expensive pairwise calculations:

```text
50,000 rows
```

were sampled without replacement for correlation analysis.

Categorical agreement used 5,000 rows from that same deterministic sample.

The EDA therefore mixed full-data exact summaries with deterministic sampling only where computationally sensible.

---

# 2. What EDA discovered

## 2.1 Severe class imbalance

The target distribution was:

```text
Negative (-1): 299,160 = 91.17%
Positive (+1):  28,975 =  8.83%
```

An always-negative classifier therefore achieves:

```text
accuracy = 0.9117
balanced accuracy = 0.5000
recall = 0
F1 = 0
```



### Design consequence

Raw accuracy became a secondary diagnostic only.

The primary comparison metric was chosen as:

```text
balanced accuracy
```

with:

```text
F1
recall
precision
specificity
```

reported alongside it.

This decision remained consistent through preprocessing ablations and final model confirmation.

---

## 2.2 Most variables are not ordinary continuous features

The EDA heuristic classification was:

| Candidate type | Count |
|---|---:|
| Binary | 13 |
| Constant | 6 |
| Continuous-like numeric | 46 |
| Date-like | 7 |
| Identifier | 2 |
| Low-cardinality categorical | 199 |
| Medium-cardinality categorical | 25 |
| Ordinal/discrete numeric | 23 |



Thus the dataset was dominated by encoded categorical/discrete survey fields rather than ordinary continuous measurements.

### Design consequence

We rejected the naive assumption:

```text
CSV numeric column → continuous variable
```

and introduced feature-type-aware preprocessing.

---

## 2.3 Missingness is extreme and structured

Feature-level missingness:

| Missing fraction | Number of predictors |
|---|---:|
| 0% | 82 |
| 0–1% | 18 |
| 1–10% | 39 |
| 10–30% | 5 |
| 30–50% | 30 |
| 50–80% | 31 |
| 80–95% | 48 |
| >95% | 68 |

A total of:

```text
147 / 321 predictors
```

were more than 50% missing.

At the row level the median subject lacked:

```text
139 / 321 predictors = 43.3%
```

and the 99th percentile subject lacked 205 predictors.

In total, 133 predictors met `strong_missingness_flag`: both groups had at least 100 rows and their absolute positive-rate difference was at least 0.03. These are descriptive thresholds, without a significance or multiple-testing correction.

Example:

```text
DIABAGE2
P(MICHD | missing) ≈ 0.0687
P(MICHD | present) ≈ 0.2213
```

### Interpretation

Observed missingness is associated with the label. Conditional survey routing is a plausible explanation, but the code does not reconstruct routing or establish why a cell is missing.

### Design consequence

We did **not** simply impute everything and discard the missing mask.

Instead:

- continuous variables could receive median imputation plus missing indicators;
- categorical missingness could become an explicit category;
- extremely sparse raw values could be dropped while retaining presence information;
- a global `missing_count` could be added.

---

## 2.4 Survey response codes are dangerous

The EDA identified:

```text
424 suspicious feature/code combinations
across 227 predictors
```

using repeated-digit patterns with lengths one through five, reporting only feature/code combinations observed at least 100 times. Examples:

```text
7, 8, 9
77, 88, 99
777, 888, 999
```



Examples included:

```text
WEIGHT2 max = 9999
HEIGHT3 max = 9999
```

while variables such as `MENTHLTH`, `CHILDREN`, `ADMOVE`, etc. frequently contained values such as `88`.

### Design stance

Frequency alone does **not** prove that `88` or `99` means missing/refused.

Therefore we rejected:

```text
all 88/99/999 → NaN
```

as unsafe.

Special codes were decoded only when explicitly verified for a specific field.

The processing configuration declares approved per-field code mappings for:

```text
GENHLTH
BPHIGH4
BPMEDS
_RFHYPE5
```

and declares numeric semantic overrides for age, height, weight and BMI variables. `feature_config.py` records the CDC codebook reference; the implementation applies those configured meanings rather than verifying clinical semantics itself. Numeric code mappings are disabled when `encode_categorical=False`, as in A0-A2.

---

## 2.5 Numeric scales differed enormously

Among numeric candidates, the ratio between largest and smallest nonzero standard deviation was approximately:

```text
1.517 × 10^6
```



This makes fixed-step gradient descent highly sensitive to scaling.

### Design consequence

For validated genuine numeric variables:

```text
verified code decoding
→ median imputation
→ train-fitted standardization
```

was used.

Parameters were learned only from the training partition.

---

## 2.6 Redundancy and rank deficiency

Exact duplicates included:

```text
SEQNO   ↔ _PSU
CTELNUM1 ↔ CELLFON2
_CRACE1 ↔ _CPRACE
```

Strong numeric correlations included:

```text
_STATE    ↔ _STSTR       r ≈ 1.0000
HTIN4     ↔ HTM4         r ≈ 0.9996
MAXVO2_   ↔ FC60_        r ≈ 0.9980
EXERHMM2  ↔ PADUR2_      r ≈ 0.9973
EXERHMM1  ↔ PADUR1_      r ≈ 0.9958
AVEDRNK2  ↔ _DRNKWEK     r ≈ 0.9802
```



The mandatory baseline later confirmed this numerically: the 322-column design including intercept had numerical Gram rank:

```text
309 / 322
```

and unregularized normal equations were singular.

### Design consequence

- declared duplicate copies are removed only after a training-row equality check;
- strongly correlated but non-identical variables were not blindly averaged;
- L2 regularization was prioritized;
- ridge / regularized logistic were preferred over unregularized direct solves;
- semantic representation groups were introduced.

---

## 2.7 Strong univariate signals

Leading target associations included:

```text
GENHLTH
CVDASPRN
_AGEG5YR
_RFHLTH
EMPLOY1
_AGE80
_AGE_G
BPHIGH4
_RFHYPE5
BPMEDS
DIFFWALK
CVDSTRK3
_AGE65YR
```



### Interpretation

The signal structure suggested:

- age is important;
- perceived/general health is important;
- hypertension and mobility/health-state variables are important;
- missingness itself often encodes health-state routing.

However, univariate association was never interpreted as proof of causal or multivariate importance.

---

## 2.8 Leakage / provenance risk

The EDA marked 92 predictors for semantic/provenance review.

The code uses explicit cardiovascular, derived-variable and survey-metadata name heuristics to request provenance review. It loads `_MICHD` from the supplied target CSV; it does not construct the label or verify the clinical derivation/timing of those predictors. Quarantine is therefore a precaution, not proof of leakage.

### Design consequence

A quarantine mechanism was created so potentially leakage-sensitive variables would not silently enter the default representation.

The final `curated_v1` excluded:

```text
24 quarantined sources
```

by default.

---

# 3. How exploration drove feature processing and engineering

## 3.1 Processing architecture

The [preprocessing implementation](feature_processing/preprocessing.py) provides a reusable:

```text
fit → state → transform
```

system.

The fitted preprocessing statistics below are estimated using training rows only. Explicit policy/type hints are a separate input; the standalone runner may load candidate types from full-data EDA, while experiment recipes use empty EDA type hints.

- medians;
- means/stds;
- category vocabularies;
- rare groups;
- target encodings;
- clipping thresholds;
- normalization parameters.

The saved state was reloaded and verified to reproduce validation transformations exactly.

This isolates fitted preprocessing statistics from validation/test rows. It does not establish the absence of all leakage: semantic policy choices can use earlier full-data EDA, and model/threshold selection reuses development labels.

---

# 3.2 `curated_v1` final structure

The saved full-data **standalone processing run** (`feature_processing/outputs/preprocessing_state.json`) used EDA type hints and produced the following counts on its 262,508 / 65,627 split. These are fitted-state counts, not universal properties of the profile. The experiment wrapper obtains recipes directly from `ablation_configs()` without loading those EDA hints; sample size and input hints can change the fitted schema.

| Quantity | Value |
|---|---:|
| Raw predictors | 321 |
| Dropped raw predictors | 202 |
| Output dimensions, excluding model intercept | 541 |
| Verified binary sources | 1 |
| One-hot categorical sources | 114 |
| Numeric sources | 4 |
| Missing/presence indicators | 62 |
| Engineered recipes | 0 |
| Constants removed | 5 |
| Exact duplicates removed | 2 |
| Extremely sparse raw values removed | 61 |
| Quarantined sources excluded | 24 |
| Unused medium categorical sources | 41 |
| Unverified numeric sources excluded | 33 |
| dtype | float32 |
| Train matrix | 262,508 × 541 |
| Validation matrix | 65,627 × 541 |
| Train + validation matrix memory | 0.661 GiB |



The processing report records the following wall-clock time for that local execution; it is not a portable runtime guarantee:

```text
72.45 s
```

---

# 3.3 Categorical handling

The processing system:

- kept `__MISSING__` distinct from `__OTHER__`;
- learned category vocabularies on training only;
- reserved representations for unseen validation/test categories;
- used reference levels to avoid redundant full dummy sets;
- avoided blanket one-hot expansion.



For context, the EDA estimates the total design width when every candidate categorical source is fully one-hot encoded, missing levels are added and the remaining raw columns are retained:

```text
1,690 columns
≈ 4.13 GiB
```

for one dense float64 matrix before optimizer copies. This is a hypothetical width estimate, not an allocated final representation.

---

# 3.4 Missingness engineering

Processing introduced:

- categorical missing levels;
- deduplicated numeric missing/presence flags;
- a scaled `missing_count`.

The global missing count uses **raw NaNs before code decoding or imputation**, excluding IDs, configured duplicate-copy names and quarantined sources. The saved standalone state counts 293 eligible sources. Other dropped raw fields, including sparse or canonically excluded fields, can still contribute to this summary; exclusion of a raw value does not necessarily exclude its missingness. Decoded survey sentinels do not increase this global raw-NaN count.

This illustrates an important project principle:

> Quarantined/identifier sources must stay out of summary features; other raw-value removals may intentionally retain missingness information.

---

# 3.5 Redundancy handling

The design rejected indiscriminate averaging.

For example:

```text
HTIN4 and HTM4
```

are likely alternate unit representations. Their mean has no sensible physical interpretation.

Instead, the pipeline used explicit concept profiles, allowing one representation of:

- age;
- height;
- health;
- blood pressure;
- physical activity;
- smoking;
- race;

to be selected or compared.

---

# 3.6 Controlled nonlinear engineering

Based on the EDA, candidate nonlinear terms included ideas such as:

```text
age²
BMI²
age × general health
age × hypertension
age × smoking
age × diabetes
```

The code implements only explicitly configured recipes, not full polynomial expansion. In the executed A5/A6 ablations the only engineered recipes were squares of the standardized/imputed `_AGE80` and `_BMI5` columns, followed by train-fitted normalization. The listed interactions were ideas, not evaluated features; interaction dictionaries and the condition-count list were empty by default.

The executed A4-to-A5 comparison of those two square terms improved selected-threshold balanced accuracy by only:

```text
+0.00011
```

approximately one hundredth of a percentage point.

Thus these two square terms produced little BA improvement under the fixed evaluator. This does not test the listed interactions or rule out useful nonlinear features.

---

# 4. Feature-processing ablation: what actually helped?

A controlled ablation used 100,000 sampled observations and the same deterministic split.

The [ablation evaluator](feature_processing/evaluate_processing.py) fits every representation on identical training rows and uses the following L2-logistic setup:

```text
batch size = 1024
early stopping = true
evaluation every 50 updates
gamma = 0.03
lambda = 0.01
max_iters = 1500
patience = 5
improvement tolerance = 1e-6
no intercept penalty
zero initialization
seed = 42
no class weighting
```



All A0-A6 stages exclude IDs and the default quarantine; A0 is not an exact reproduction of the mandatory baseline's all-321-predictor matrix. A3 jointly changes decoding, binary maps, categorical handling and unsupported-value exclusions; its delta is not a one-hot-only ablation. A5 adds the two square terms and A6 adds 41 OOF-encoded sources on top of A5. Dimensions below exclude the modeling intercept.

Results:

| Stage | Dimensions | Selected BA | F1 | Recall | Precision | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| A0 minimal | 295 | **0.7671** | **0.3580** | 0.7871 | 0.2317 | 0.1135 |
| A1 + safe removals | 288 | 0.7671 | 0.3580 | 0.7871 | 0.2317 | 0.1136 |
| A2 + missingness | 400 | 0.7668 | 0.3447 | 0.8165 | 0.2185 | 0.0886 |
| A3 + categorical policy | 586 | 0.7639 | 0.3349 | 0.8313 | 0.2097 | 0.0855 |
| A4 + canonical representatives | 537 | 0.7614 | 0.3262 | 0.8465 | 0.2020 | 0.0818 |
| A5 + controlled engineering | 539 | 0.7615 | 0.3231 | 0.8567 | 0.1991 | 0.0784 |
| A6 + target encoding | 580 | 0.7618 | 0.3233 | 0.8573 | 0.1992 | 0.0784 |



---

## 4.1 Important deltas

```text
A0 → A1:  +0.00000 BA
A1 → A2:  -0.00030 BA
A2 → A3:  -0.00289 BA
A3 → A4:  -0.00258 BA
A4 → A5:  +0.00011 BA
A5 → A6:  +0.00031 BA
```



### Major lesson

The later stages impose explicit semantic and representation policies, but were not more predictive under this fixed evaluator. All stages use training-fitted statistics and the same default quarantine; this comparison does not establish that one stage is categorically free of leakage or safer in every respect.

The standalone full-data curated run excluded:

```text
41 medium-cardinality categorical variables
33 unverified numeric variables
```

from its raw-value representation. The 100k A4/A5 ablation state instead records **31** unverified numeric exclusions and 41 unused-medium exclusions. Counts depend on the training partition and hints; A6 encodes the 41 medium sources rather than dropping them.

A rationale consistent with these measured comparisons is:

> Cleaning is necessary, but every aggressive removal/transformation must earn its place empirically.

This provides a rationale for the implemented **selective recovery** target-encoding experiment; the ablation alone does not prove that recovery will help.

---

# 5. Mandatory functions: original implementation

The [mandatory module](<Mandatory Functions/implementations.py>) is separate and implements the six required signatures and local course conventions. Local unit tests exercise it; this repository does not demonstrate an official autograder result.

The six functions were:

```text
mean_squared_error_gd
mean_squared_error_sgd
least_squares
ridge_regression
logistic_regression
reg_logistic_regression
```

The implementation followed course conventions:

- MSE contains the factor `0.5`;
- SGD draws batch size 1;
- least squares/ridge use normal equations;
- logistic labels are `{0,1}`;
- regularized returned loss excludes the penalty;
- weights are one-dimensional.

 

Processing/experiment runners check fingerprints to prevent changes to the mandatory module during execution. The saved processing-run fingerprint matches the current file.

---

# 6. Mandatory baseline experiments

The separate baseline diagnostic evaluates all six mandatory methods on a deliberately crude common representation:

- 100,000-row stratified diagnostic sample;
- 80,000 training;
- 20,000 validation;
- all 321 predictors retained numerically;
- actual missing cells median-imputed;
- columns standardized;
- categorical integer codes still treated numerically;
- no one-hot;
- no class weighting;
- no threshold search.



This was not intended as the final representation. It was intended to diagnose optimization and model-family behavior.

---

## 6.1 Exact mandatory hyperparameters

### MSE-GD

Iteration budgets:

```text
10
50
100
```

Training-derived learning rate:

```text
gamma ≈ 0.047343
```

### MSE-SGD

The diagnostic resets NumPy's global RNG to **1043** for each fit and restores the caller's RNG afterward, so independent budgets use the same with-replacement sampling prefix. This differs from the augmented fitter's seeded, shuffled epoch batches.

Iteration budgets:

```text
1,000
10,000
100,000
```

Learning rate:

```text
gamma ≈ 1.4376 × 10^-5
```

### Ridge

Lambdas:

```text
0
0.0001
0.01
1.0
```

### Logistic regression

Iteration budgets:

```text
10
50
100
```

Learning rate approximately:

```text
gamma ≈ 0.189372
```

### Regularized logistic regression

Lambdas:

```text
0.0001
0.01
1.0
```

Iteration budgets:

```text
10
50
100
```

Learning rates were adjusted according to the regularized smoothness estimate.

In [baseline_diagnostics.py](<Exploratory Analysis/baseline_diagnostics.py>), `L` is the largest eigenvalue of the training Gram matrix `tx.T @ tx / N`; the SGD bound uses the maximum training-row squared norm. The step-size rules are:

```text
GD gamma       = 1 / L
SGD gamma      = 1 / max_i ||tx_i||^2
logistic gamma = 1 / (0.25 L)
reg-log gamma  = 1 / (0.25 L + 2 lambda)
```



---

## 6.2 Mandatory baseline results

Best observed configuration per mandatory method:

| Model | BA | Recall | F1 |
|---|---:|---:|---:|
| Majority negative | 0.5000 | 0.0000 | 0.0000 |
| MSE-GD | 0.6286 | 0.3058 | 0.3383 |
| MSE-SGD | 0.7144 | 0.7424 | 0.2982 |
| Ridge | 0.6175 | 0.2746 | 0.3261 |
| Logistic | 0.6367 | 0.3296 | 0.3452 |
| Regularized logistic | **0.7170** | 0.5895 | **0.3691** |



---

# 7. What the mandatory baselines taught us

## 7.1 The normal-equation OLS solve failed on this design

The code estimated the **numerical rank of the normalized training Gram matrix**, using the tolerance `largest_eigenvalue * 322 * machine_epsilon`, as:

```text
309 / 322
```

and:

```text
least_squares
ridge(lambda=0)
```

both failed with a singular matrix.

This confirmed the EDA's redundancy findings and justified regularization / feature cleanup.

---

## 7.2 L2 regularization mattered strongly

For logistic regression:

```text
unregularized best diagnostic BA ≈ 0.6367
regularized λ=1 BA ≈ 0.7170
```

Difference:

```text
+0.0803 balanced accuracy
```

Regularized logistic also improved recall from about:

```text
0.330 → 0.589
```

under the compared diagnostic configurations.

This supports testing regularized logistic under this diagnostic setup. The comparison also changes learning rate through the curvature rule and regularizes the intercept; it does not isolate a universally beneficial L2 effect.

---

## 7.3 But the mandatory regularization had an undesirable side effect

The course function applies:

```text
2 λ w
```

to **all coefficients**.

If column zero is an intercept, the intercept is also regularized.

The gradient confirms that the intercept is shrunk along with feature coefficients. This could affect calibration, but the diagnostic does not run a matched intercept-exempt control and cannot attribute the large-lambda gain specifically to intercept shrinkage.

This motivated the augmented option:

```text
penalize_intercept=False
```

for the actual modeling experiments.

---

## 7.4 Optimizing training loss did not imply better classification

MSE-GD:

```text
10 updates:
BA = 0.6286
recall = 0.3058

100 updates:
BA = 0.5066
recall = 0.0136
```

despite MSE falling.

MSE-SGD:

```text
1,000 updates:
BA = 0.7144
recall = 0.7424

100,000 updates:
BA = 0.5194
recall = 0.0419
```

while training MSE fell from approximately 0.486 to 0.156.

Logistic similarly showed lower loss but worse classification at the fixed 0.5 threshold across longer runs.

### Important interpretation

The saved results show lower training loss and more majority-negative predictions at longer budgets. They do not isolate statistical overfitting, optimization progress and decision-threshold effects. A plausible mechanism involves:

```text
loss optimization
+
severe class imbalance
+
fixed classification threshold
```

changing the score distribution relative to the fixed decision threshold. No threshold-retuned control was executed in these mandatory diagnostics.

This observation drove several augmented-function changes.

---

# 8. Why the mandatory functions were augmented

The mandatory implementations expose the minimal six-function API. The [augmented module](<Mandatory Functions/implementations_augmented.py>) adds experimental facilities without replacing those functions; local tests check specified behavior rather than proving correctness on every possible input.

A separate toolbox supplies those facilities; the later processing and experiment runners import it and check ML-module fingerprints.

The augmentations addressed specific observed dataset/model problems.

---

## 8.1 Mini-batch optimization

### Problem

There are more than 300,000 observations and potentially hundreds of processed dimensions.

Full-batch gradient updates repeatedly multiply the entire matrix.

### Augmentation

Support:

```text
batch_size=None → full batch (iterative logistic/SGD interfaces)
batch_size=1    → SGD
batch_size>1    → mini-batch GD
```

### Why

This reduces per-update gradient work while remaining NumPy-based. `fit_mse_gd` stays full-batch; minibatch selection applies to `fit_mse_sgd` and the logistic fitters. Checkpoint monitoring still computes full training/validation losses and, with history enabled, a full training gradient.

---

## 8.2 Explicit seeded RNG

### Mandatory behavior

The course SGD uses global NumPy randomness.

### Augmentation

A local:

```python
np.random.default_rng(seed)
```

was introduced.

### Why

Each experiment can be reproduced independently without hidden dependence on unrelated random calls.

---

## 8.3 Early stopping and validation monitoring

### Problem

The mandatory diagnostics showed that more optimization could *reduce* balanced accuracy even while loss decreased.

### Augmentation

Training can monitor **unweighted, unregularized validation data loss**, remember a checkpoint and stop after `patience` evaluation checkpoints without an improvement strictly greater than the absolute `tol`. Checkpoint zero, every `eval_every` updates and the final requested update are evaluated. With early stopping enabled, returned weights restore the stored best checkpoint even if the run reaches its cap; `n_updates` still records performed updates. Patience is not a count of epochs or individual gradient updates, and the monitored loss is not BA/F1.

### Why

Iteration count is no longer treated as “the more the better.”

---

## 8.4 Training histories and rich results

The augmented interface returns diagnostics such as:

```text
weights
unregularized training data loss at the returned weights
number of updates
stop reason
best iteration
best validation loss
history
```

rather than only `(w, loss)`.

### Why

These diagnostics support inspection of optimization and generalization. `converged=True` also covers patience-based early stopping; it is not proof that the objective reached its mathematical optimum.

---

## 8.5 No intercept penalty

Augmented ridge/logistic can regularize:

```text
feature coefficients
```

without regularizing:

```text
intercept
```

### Why

The mandatory penalty includes the intercept, so interpretation of the large-lambda result is confounded without a matched control. The augmented option removes that penalty when the caller explicitly supplies an intercept; it does not add or detect an intercept automatically.

---

## 8.6 Sample/class weighting support

Optional sample weights were introduced because the positive class is only 8.83%.

The processing evaluator and all three experiment workflows explicitly use **no class weighting**; they tune thresholds instead.

Why?

The diagnostic large-λ logistic model already predicted about 19.4% positives despite an 8.83% true positive prevalence.

The implementation separates:

```text
model fitting
```

from:

```text
decision threshold selection
```

without assuming that class weighting is needed to increase minority recall. The predicted-positive fraction is an observation; it does not by itself establish which weighting policy would be best.

---

## 8.7 Threshold optimization

The augmented helper selects a decision threshold on validation data from at most 103 default candidates: 101 score quantiles, an inferred default (0.5 for scores entirely in [0,1], otherwise 0), and a boundary above the maximum. It does not optimize over every possible threshold. Experiments separately evaluate the explicit model defaults, 0.5 for logistic and 0 for MSE.

This became essential.

In the processing ablation, A0 had:

```text
default-threshold BA = 0.5338
selected-threshold BA = 0.7671
```



Difference:

```text
+0.2333 balanced accuracy
```

without changing a single model weight.

This demonstrates the effect of changing the decision rule on the same validation scores. It does not measure probability calibration or compare its importance causally with model training; the selected score is optimistic because the same validation labels choose and evaluate the threshold.

---

## 8.8 Common metrics

Feature-processing evaluation and the three experiment workflows use the augmented NumPy implementation of:

```text
accuracy
balanced accuracy
precision
recall
specificity
F1
confusion counts
```

for consistent definitions across those workflows. Mandatory baseline diagnostics retain a separate implementation with the same formulas and different confusion-count key names; there is not one metric implementation across the entire repository.

---

## 8.9 Robust least-squares option

The augmented system allowed an explicit pseudo-inverse solution when needed.

This was not a silent fallback.

### Why

The mandatory normal-equation solver legitimately failed on the rank-deficient design.

---

# 9. How exploration and preprocessing determined the three ML strategies

## Strategy choice was evidence-driven

The chosen three experiments were not three arbitrary algorithms.

They corresponded to three specific hypotheses uncovered during EDA and baseline diagnostics.

---

# 10. Strategy 1 — L2 logistic regression on a curated representation

## Hypothesis

A binary likelihood model with L2 stabilization should work well because:

- the target is binary;
- the dataset is large;
- many predictors are redundant;
- encoded designs can remain rank-deficient, including reserved all-zero columns despite reference coding;
- mandatory diagnostics strongly favored regularized over unregularized logistic.

---

## Representation

The final confirmed strategy is reported as:

```text
R2 curated
```

and produced:

```text
exploratory BA = 0.7779
3-fold BA      = 0.7745
3-fold SD      = 0.0020
3-fold F1      = 0.3521
```



---

## Executed search design

Source: [Strategy 1 run_config.txt](ml_attempts/01_logistic_representation/outputs/run_config.txt). This is the executed search configuration, not a list of inferred winning values.

```text
lambda values       = {0.001, 0.01, 0.1, 1.0}
gamma values        = {0.01, 0.03, 0.1}
initial gamma       = 0.03
batch size          = 1024
maximum updates     = 5000
evaluation interval = 100
patience            = 5
tolerance           = 1e-5
seed                = 42
early stopping      = enabled
penalize intercept  = false
controlled engineering = disabled (test_engineering=false)
meaningful gain criterion = 0.002
```

R0/minimal, R1/missingness and R2/curated each received four lambda tests at gamma 0.03, followed by the other two gammas at that representation's best lambda: **18 configurations**, not a Cartesian sweep. No engineering block was evaluated in this run.

All 328,135 observations were used in the shared deterministic split: **262,508 training / 65,627 validation**, sampling seed 42 and split seed 43. Strategies 2 and 3 used these identical rows (see Section 17).

## Retrieved winner and provenance

The selected exploratory run was `run_018`: **R2 curated, lambda=0.001, gamma=0.1, batch=1024, constant schedule**. It performed 5,000 updates and returned checkpoint 5,000, reaching `max_iters`; convergence is not established. Its exploratory selected threshold was **0.08437629535507744**.

These values were retrieved from [selection.json](ml_attempts/01_logistic_representation/outputs/selection.json) and checked against [all_runs.csv](ml_attempts/01_logistic_representation/outputs/all_runs.csv). [validation_metrics.csv](ml_attempts/01_logistic_representation/outputs/validation_metrics.csv) and the [experiment report](ml_attempts/01_logistic_representation/outputs/experiment_report.md) provide the corresponding evaluation.

The all-training refit is a separate artifact: its frozen threshold is **0.0862364357063921**, with **6,252 updates**, as recorded in [frozen_final_configuration.json](ml_attempts/01_logistic_representation/outputs/frozen_final_configuration.json). Do not substitute the exploratory threshold or 5,000-update search cap for these final settings.

---

# 11. Strategy 2 — selective target-encoded L2 logistic regression

## Why this strategy changed during the project

Initially, target encoding was motivated mainly by memory:

- 237 categorical candidates;
- dense one-hot could become very wide.

However, the preprocessing ablation showed:

```text
A5 selected BA = 0.7615
A6 + OOF target encoding = 0.7618
```

only:

```text
+0.00031
```

improvement.

So the final hypothesis was refined.

Instead of:

> “Target encoding itself will beat one-hot encoding,”

the hypothesis became:

> “Selective target encoding may recover predictive information that conservative preprocessing excluded.”

The saved standalone and 100k curated states each excluded 41 sources with drop reason `unused_medium_cardinality`. This is the pipeline's effective-cardinality policy, not the EDA's separate 11-30-level type definition; the two categories must not be conflated.

---

## Fold-isolated target transformation

For category \(c\):

\[
TE_c=
\frac{n_c\bar y_c+\alpha\bar y}
{n_c+\alpha}.
\]

Critically:

- training target encoding was cross-fitted / out-of-fold;
- validation/test mappings were learned from training data only.

The encoding code fits each OOF fold's vocabulary, rare groups, prior and rates without that fold's labels. OOF fold assignment is a label-independent seeded permutation, not stratified splitting. Encoded rates remain on the probability scale, avoiding full-OOF mean/std normalization that could indirectly transmit a row's label. Source eligibility is selected from the whole current training partition using feature/support statistics, without target labels.

---

## Executed selective-recovery configuration

Source: [Strategy 2 run_config.txt](ml_attempts/02_logistic_target_encoded/outputs/run_config.txt).

```text
default smoothing alpha     = 20
additional alpha values     = {5, 100} (competitive variants only)
OOF target-encoding folds   = 5
target-encoding seed        = 42
maximum recovered features = 8
minimum observed support   = 1000
meaningful gain criterion  = 0.002
competitive margin         = 0.002
```

Candidate recovery priority, in order:

```text
EDUCA
INCOME2
EMPLOY1
_INCOMG
_EDUCAG
HHADULT
CHILDREN
CRGVREL1
ADPLEASR
ADDOWN
ADSLEEP
ADENERGY
ADEAT1
ADFAIL
ADTHINK
ADMOVE
```

TE1 tested the compact curated profile; TE2 selectively recovered supported, unused medium-cardinality fields on the actual Strategy 1 winning base. Training-only support checks and quarantine/identifier exclusions constrained recovery; this was **not blanket target encoding**.

Both variants tested lambda*/10, lambda*, and 10*lambda* around the Strategy 1 winner, inheriting its gamma and optimizer settings. In this run the lambda neighborhood was **{0.0001, 0.001, 0.01}**, at **gamma=0.1**, batch 1024, maximum 5,000 updates, evaluation every 100, patience 5, tolerance 1e-5 and optimizer seed 42. Only the leading representation is retested at alpha 5/100, and only if it lies within the competitive margin of the Strategy 1 benchmark. Here that was TE2 at its selected lambda; two alpha checks brought the executed total to **eight configurations**. All used the same 262,508 / 65,627 full-data split as Strategy 1.

**Five OOF folds encode training rows; three outer folds confirm the finalist. These fold counts describe different operations.** Each OOF encoding excludes its own fold's labels and fits vocabulary, support groups and prior using the remaining training folds. Full-training encoders apply only to unseen validation/test rows.

## Retrieved winner and provenance

[selection.json](ml_attempts/02_logistic_target_encoded/outputs/selection.json) and [all_runs.csv](ml_attempts/02_logistic_target_encoded/outputs/all_runs.csv) identify `run_004`: **TE2 selective recovery, alpha=20, lambda=0.0001, gamma=0.1, batch=1024, constant schedule**. It performed and selected checkpoint 5,000 (`max_iters`). The exploratory threshold was **0.08541188081395756**.

The seven recovered sources in the selected exploration and saved final representation were:

```text
EDUCA, INCOME2, EMPLOY1, _INCOMG, HHADULT, CHILDREN, CRGVREL1
```

The candidate list is not the selected subset. The final subset is independently recorded in [preprocessing_state/representation.json](ml_attempts/02_logistic_target_encoded/outputs/preprocessing_state/representation.json); confirmation fits its own training-dependent encoders within each fold.

The frozen all-training threshold is **0.08216501573231662**, with **6,252 updates**, from [frozen_final_configuration.json](ml_attempts/02_logistic_target_encoded/outputs/frozen_final_configuration.json). These replace the exploration threshold/budget for final inference. See also [validation_metrics.csv](ml_attempts/02_logistic_target_encoded/outputs/validation_metrics.csv) and the [experiment report](ml_attempts/02_logistic_target_encoded/outputs/experiment_report.md).

---

## Final confirmed result

The finalist was:

```text
TE2 selective recovery
```

Results:

```text
exploratory BA = 0.7791
3-fold BA      = 0.7757
fold SD        = 0.0019
3-fold F1      = 0.3532
```



This was the highest observed confirmation mean. The exploratory TE2 gain over tuned R2 was only **0.0012233059 BA**, below the configured **0.002** material-gain criterion, and TE2 selected a different lambda (0.0001 versus 0.001). The run does not isolate an encoding effect, demonstrate a material benefit or achieve equivalent performance with 20% fewer dimensions; it adds seven columns to the winning base. The alpha 5/20/100 checks had identical selected BA at the winning lambda.

---

# 12. Strategy 3 — early-stopped MSE-SGD

## Why use squared-error classification at all?

The mandatory diagnostic produced a surprising result:

```text
MSE-SGD, 1,000 updates:
BA ≈ 0.7144
```

which was already close to:

```text
regularized logistic:
BA ≈ 0.7170
```

on the crude representation.

Thus MSE-SGD offered a potentially inexpensive linear classifier.

The risk was also clear: continued optimization collapsed minority performance.

---

## Final representation

The final confirmed attempt used:

```text
S0 R2 curated
```

so that MSE-SGD was compared using the same strong representation as the logistic approach.

---

## Executed search design

Source: [Strategy 3 run_config.txt](ml_attempts/03_mse_sgd/outputs/run_config.txt).

```text
gamma values             = {0.001, 0.01, 0.03}
batch sizes              = {1, 256, 1024}
initial batch size       = 256
initial maximum updates  = 4000
checkpoint budgets       = {500, 2000, 8000, 16000}
evaluation interval      = 500
patience                 = 5
tolerance                = 1e-5
seed                     = 42
early stopping           = enabled during exploration/confirmation
baseline schedule        = constant
additional schedule      = inverse
inverse lr decay         = 0.0001
```

The same 262,508 / 65,627 split was used. The staged search first compared gammas at batch 256, then batch sizes at the chosen gamma, then one inverse schedule. Budget-prefix diagnostics independently replayed the same RNG and schedule from zero. **Ten configurations** were executed; the initial 4,000-update cap and the separate prefix budgets are different controls.

The augmented toolbox supports other schedules, but **sqrt was not tested in this experiment**. Validation-MSE early stopping returns the best stored checkpoint under the absolute improvement tolerance, which need not be the literal minimum over every update. Classification metrics and threshold selection rank those returned weights across runs. Requested budget, performed updates and returned checkpoint must remain separate.

## Retrieved winner and provenance

[selection.json](ml_attempts/03_mse_sgd/outputs/selection.json), checked against [all_runs.csv](ml_attempts/03_mse_sgd/outputs/all_runs.csv), identifies `run_002`: **S0 R2 curated, gamma=0.01, batch=256, constant schedule, no L2 penalty**. It performed 4,000 updates and selected checkpoint 4,000 (`max_iters`); its exploratory threshold was **-0.7618374809736443**.

The all-training refit instead uses **5,001 updates** and frozen threshold **-0.743225991984627**, with early stopping disabled and the budget fixed from confirmation. These settings come from [frozen_final_configuration.json](ml_attempts/03_mse_sgd/outputs/frozen_final_configuration.json), not from the search space. See also [validation_metrics.csv](ml_attempts/03_mse_sgd/outputs/validation_metrics.csv) and the [experiment report](ml_attempts/03_mse_sgd/outputs/experiment_report.md).

---

## Final result

```text
exploratory BA = 0.7750
3-fold BA      = 0.7710
fold SD        = 0.0006
3-fold F1      = 0.3525
```



It was only:

```text
0.7757 - 0.7710 = 0.0047
```

behind the best target-encoded logistic model.

Its fold SD was the smallest of the three finalists, although three folds are not enough to claim intrinsically greater stability.

---

# 13. Final model comparison

## 13.1 Cross-validation ranking

| Rank | Model | CV BA |
|---|---|---:|
| 1 | TE2 target-encoded L2 logistic | **0.7757** |
| 2 | R2 curated L2 logistic | 0.7745 |
| 3 | R2 early-stopped MSE-SGD | 0.7710 |

Differences:

```text
TE2 - R2 logistic = +0.0012 BA
TE2 - MSE-SGD     = +0.0047 BA
```

---

## 13.2 Exploratory-to-confirmation differences

Exploratory → three-fold confirmation, with differences computed from the unrounded stored metrics:

```text
TE2 logistic: 0.7791342751 → 0.7757009977 = -0.00343
R2 logistic:  0.7779109692 → 0.7744557458 = -0.00346
MSE-SGD:      0.7749911674 → 0.7709909260 = -0.00400
```

These are observed score drops, not an isolated measurement of selection optimism. Confirmation trains each fold model on about 53% of the pool rather than the exploratory 80%, changes training budgets and uses different stopping/calibration/validation rows. Prior recipe selection also overlaps the confirmation data. Those differences prevent assigning the drop to one cause.

---

## 13.3 Statistical caution

The [confirmation code](ml_attempts/confirmation.py) evaluates only three preselected finalists and reports unweighted fold means and sample SDs (`ddof=1`), not confidence intervals or a paired significance test. Individual fold variability alone does not establish whether a paired model difference is significant.

Therefore the defensible report wording is:

> Selective target encoding achieved the highest mean 3-fold balanced accuracy, but its advantage over curated logistic regression was small relative to fold-to-fold variability.

Not:

> Target encoding significantly outperformed curated logistic regression.

---

# 14. Complete baseline / ablation narrative

A concise descriptive development comparison for the report is:

### Baseline 0 — majority classifier

```text
BA = 0.5000
F1 = 0
```

Shows raw accuracy is misleading.

### Mandatory unregularized logistic

```text
BA ≈ 0.6367
```

### Mandatory L2 logistic

```text
BA ≈ 0.7170
```

Gain over unregularized diagnostic logistic:

```text
+0.0803 BA
```

Evidence for testing regularization under the diagnostic setup, with the learning-rate/intercept caveats in Section 7.

### Processing A0

```text
selected BA ≈ 0.7671
```

Gain versus crude mandatory regularized logistic:

```text
≈ +0.0501 BA
```

This combines representation changes, augmented optimization and threshold tuning, so it must not be attributed to one processing change.

### A0 → A6 processing stack

```text
0.7671 → 0.7618
```

Net:

```text
-0.0053 BA
```

under one fixed logistic configuration.

This shows the bundled processing policy performed worse under one fixed optimizer setup. It does not identify discarded signal as the cause: optimization, coding and threshold changes are also bundled.

### Final TE2 selective recovery

Exploratory:

```text
0.7791
```

versus preprocessing A0:

```text
+0.0120 BA
```

but these are not a perfectly controlled same-stage comparison, so this should be described as development progression rather than a single isolated causal effect.

### Final 3-fold confirmation

```text
TE2 = 0.7757
R2  = 0.7745
SGD = 0.7710
```

This is the strongest generalization evidence currently available.

---

# 15. Overfitting / underfitting assessment

## 15.1 Is there evidence of severe overfitting?

The available results do not establish or rule out severe overfitting. The measured fold SDs are:

```text
TE2 fold SD = 0.0019
R2 fold SD  = 0.0020
SGD fold SD = 0.0006
```

and exploratory-to-CV performance dropped only around 0.003–0.004 BA.

Scores varied little across these three specific folds, but three conditional finalist evaluations do not measure the variance or bias of the whole selection procedure. Small fold SD does not prove absence of overfitting; the exploratory-to-confirmation gap also mixes training-set-size and protocol changes.

---

## 15.2 Where overfitting risk actually entered

The code exposes a concrete **selection-overfitting risk**; it does not quantify that risk relative to parameter overfitting.

Development validation labels were reused within workflows for:

- preprocessing comparisons;
- early stopping;
- threshold selection;
- representation selection;
- hyperparameter search.

The processing report explicitly warns that these exploratory comparisons are not untouched final estimates.

The later three-fold confirmation provides another comparison, but its source observations overlap earlier recipe selection. The 100k processing split and the full-data experiment split also overlap source data without being identical validation partitions.

---

## 15.3 Does the model underfit?

The runs do not establish how much signal remains uncaptured. Logistic and MSE have similar measured BA, two standardized square terms added little under one fixed evaluator, and TE2's small gain changes regularization as well as representation. None of these is a broad test of nonlinear capacity or an estimate of an achievable performance ceiling.

Selected logistic exploratory runs reached their 5,000-update cap, so optimization undertraining also remains possible. All finalists are linear in their processed features; one-hot and target encoding already make those features nonlinear functions of raw survey codes. The code does not show that an additive model captures most available signal or that the optimal raw-feature boundary is linear.

There may still be unexploited:

- interactions;
- nonlinear age/risk relationships;
- conditional survey-routing effects;
- semantically recoverable variables.

---

## 15.4 Training loss versus classification metrics

The project uncovered an important distinction:

> Better optimization of the surrogate objective did not necessarily improve the desired classification metric.

For MSE-SGD:

```text
loss ↓ strongly
balanced accuracy ↓ strongly
```

at long training budgets.

This motivates monitoring losses and classification separately. It does not prove the benefit of validation-MSE early stopping: the mandatory budgets used fixed thresholds, and the later MSE search kept early stopping enabled throughout rather than running a matched on/off ablation. In that search the selected-threshold BA peaked at the 4,000-update run and fell at 8,000 while best validation MSE improved; selecting the lowest MSE alone would not identify the classification winner.

---

## 15.5 The decision threshold had a large measured effect

A0:

```text
BA at threshold 0.5 = 0.5338
BA at selected threshold = 0.7671
```

This is a huge difference.

It shows that the same validation scores support very different BA at different decision thresholds. It does not establish probability miscalibration or quantify ranking quality independently. A threshold selected for BA can differ from 0.5 even for calibrated probabilities under class imbalance; calibration requires separate diagnostics. This is also a selected-on-validation result, not an untouched performance estimate.

---

# 16. Error analysis: what is known and what remains

The code saves aggregate confusion metrics and feature associations, plus per-row exploratory `runs/run_*/validation_scores.npy` aligned with shared validation indices. Confirmation saves each fold's indices, weights and preprocessing state, but does not export per-row outer scores. No dedicated subgroup or individual-level false-positive/false-negative analysis is implemented; fold scores for such an analysis would need to be regenerated from the saved models.

Therefore a claim such as:

> “False negatives are mainly elderly diabetic subjects”

would currently be unsupported.

### A useful final diagnostic, if time permits

For each finalist, compare:

```text
true positives vs false negatives
true negatives vs false positives
```

over a small set of interpretable variables:

```text
age
GENHLTH
BPHIGH4 / hypertension
DIFFWALK
smoking state
diabetes state
BMI
missing_count
```

Questions:

- Are false negatives concentrated near the decision threshold?
- Do false negatives have systematically more missing data?
- Are particular age/health categories poorly represented?
- Do false positives arise from generally poor health without MICHD?
- Does target encoding mainly change difficult borderline cases?

A high-value plot would be the score distribution for:

```text
TP / FN / FP / TN
```

for the final TE2 model.

---

# 17. Reproducibility ledger

## Dataset

```text
x_train.csv
y_train.csv
x_test.csv
```

Raw files are not modified.

---

## EDA

```text
seed = 433
full-data descriptive summaries
50,000-row deterministic correlation sample
5,000-row categorical agreement subset
```

---

## Mandatory diagnostic

```text
diagnostic rows = 100,000
training rows   = 80,000
validation rows = 20,000

sampling seed = 42
split seed    = 43
SGD sampling seed = 1043 (global RNG reset per fit)
```



---

## Processing ablation

```text
100,000-row sample
same split hash across representations
sampling seed = 42
split seed = 43
```

Fixed evaluator:

```text
batch_size = 1024
gamma = 0.03
lambda = 0.01
max_iters = 1500
patience = 5
tol = 1e-6
eval_every = 50
penalize_intercept = False
seed = 42
```



---

## Full curated processing

```text
training rows   = 262,508
validation rows = 65,627
float32 representation
541 output dimensions, excluding intercept
EDA type hints loaded for this standalone run
```



---

## Shared full-data experiment split

The three executed `run_config.txt` files map to **Strategy 1: curated / representation-tuned L2 logistic**, **Strategy 2: selective target-encoded L2 logistic**, and **Strategy 3: early-stopped MSE-SGD**, respectively. Their paths are linked in Sections 10-12.

```text
source/sample rows = 328135 (all labeled observations)
training rows      = 262508
validation rows    = 65627
sampling seed      = 42
split seed         = 43
validation fraction = 0.2
split SHA256 = 0e013bf3db5065c7801bec38f60df87560b47efc0d4efb457fd44be9ca91f4a1
```

All three run configurations record the identical split and source hashes. Saved indices refer to original zero-based CSV data rows. This full-data experiment split is distinct from the earlier 100,000-row baseline/processing-ablation split.

## Finalist confirmation versus target-encoding folds

```text
finalist confirmation = 3 stratified outer folds, seed 2026
finalists             = one preselected model per strategy
inner stopping split  = 20% of each outer-training partition
inner split seeds     = 901, 902, 903
Strategy 2 encoding   = 5 OOF training folds, seed 42
```

Preprocessing and OOF target encoding are refitted on each inner-training partition. Inner stopping labels choose checkpoints and thresholds; outer labels only score the frozen fold model. There is no outer-training refit after calibration: each fold model fits about 53% of the full pool. The shared protocol is implemented in [confirmation.py](ml_attempts/confirmation.py), with fold results in each attempt's `outputs/cv_metrics.csv`.

Three-fold finalist confirmation and five-fold target encoding are separate operations, not interchangeable descriptions of cross-validation. Earlier recipe selection used overlapping source observations, so confirmation is conditional rather than a fully nested audit of the search.

Mean confirmation BA:

```text
TE2 logistic = 0.7757
R2 logistic  = 0.7745
MSE-SGD      = 0.7710
```

## Final refit and artifact authority

After confirmation, each model freezes the median inner-selected threshold and median selected training-data-pass budget. The budget is rescaled to all **328,135** labeled rows; the resulting update counts need not equal the exploration caps. Final refits disable early stopping because no validation partition remains. Test data are used only for inference after all requested confirmations finish.

- `run_config.txt`: executed search settings and source/split provenance; it does not identify the winner.
- `selection.json`, `all_runs.csv`, `validation_metrics.csv`, `experiment_report.md`: selected exploratory recipe, optimizer values, checkpoints, thresholds and supporting comparisons.
- `cv_metrics.csv`: individual confirmation fold metrics, stopping points and thresholds.
- `frozen_final_configuration.json`, `final_refit.json`, `preprocessing_state/representation.json`: frozen final settings, actual all-training fit, and recovered subset/schema.

These generated artifacts are local and Git-ignored. Exact winner values below were retrieved from the available files, not inferred from candidate grids; if artifacts are absent in another checkout, retrieve them before revising those values.

---

# 18. Parameter table for the final report

Search settings are documented in Sections 10-12. The table below separates **selected exploratory runs** from **frozen all-training refits**, using the local selection/final metadata linked there.

| Quantity | Strategy 1: R2 logistic | Strategy 2: TE2 logistic | Strategy 3: MSE-SGD |
|---|---|---|---|
| Representation | R2 curated | TE2 selective recovery | S0 R2 curated |
| Objective | logistic NLL + L2 | logistic NLL + L2 | half-MSE, no L2 |
| Selected exploratory run | run_018 | run_004 | run_002 |
| Selected lambda | 0.001 | 0.0001 | N/A |
| Selected gamma | 0.1 | 0.1 | 0.01 |
| Batch size | 1024 | 1024 | 256 |
| Selected LR schedule | constant | constant | constant |
| Selected LR decay | 0 | 0 | 0 |
| Exploratory requested updates | 5000 | 5000 | 4000 |
| Exploratory performed / best update | 5000 / 5000 | 5000 / 5000 | 4000 / 4000 |
| Exploratory stop reason | max_iters | max_iters | max_iters |
| Exploratory evaluation interval | 100 | 100 | 500 |
| Exploration/confirmation patience | 5 | 5 | 5 |
| Tolerance / optimizer seed | 1e-5 / 42 | 1e-5 / 42 | 1e-5 / 42 |
| Exploratory selected threshold | 0.08437629535507744 | 0.08541188081395756 | -0.7618374809736443 |
| Frozen final threshold | 0.0862364357063921 | 0.08216501573231662 | -0.743225991984627 |
| All-training refit updates | 6252 | 6252 | 5001 |
| All-training refit early stopping | disabled | disabled | disabled |
| All-training rows | 328135 | 328135 | 328135 |
| Target-encoding alpha | N/A | 20 | N/A |
| Target-encoding OOF folds / seed | N/A | 5 / 42 | N/A |
| Selected recovered sources | N/A | 7; listed in Section 11 | N/A |
| Finalist confirmation outer folds | 3 | 3 | 3 |
| Mean confirmation BA | 0.7745 | 0.7757 | 0.7710 |
| Mean confirmation F1 | 0.3521 | 0.3532 | 0.3525 |

The patience and evaluation intervals describe model selection, not an enabled final-refit stopping rule. Confirmation may adjust evaluation frequency to its smaller budget; inspect fold metadata for actual checkpoint counts. The two threshold rows serve different stages: only the frozen final threshold belongs with the exported final weights.

No winner-specific placeholders remain: the winner values and final recovered subset were available locally and checked against saved artifacts. Any future missing winner-specific value must be retrieved from those artifacts and explicitly marked unresolved until verified; `run_config.txt` alone is not evidence of selection.

---

# 19. Low-hanging fruit for further improvement

The following are **proposed, unevaluated follow-ups**, not implemented improvements or a ranking established by the code. Their benefit and cost remain unknown. A matched-base regularization control and larger optimization budgets are needed before attributing TE2's small gain to encoding or treating capped logistic fits as converged.

## Priority 1 — verify more survey codes / recover excluded numeric information

The saved standalone full-data `curated_v1` state excluded:

```text
33 unverified numeric sources
41 medium-cardinality categorical sources
```



TE2's small observed advantage is compatible with useful recovered information, but its changed regularization prevents attributing that advantage to the recovered block. The 100k ablation state records 31 rather than 33 unverified numeric exclusions; these are partition/configuration-dependent counts.

A promising next step is:

```text
semantic verification
→ safe code decoding
→ selectively reintroduce
→ validate
```

rather than broadly restoring every excluded feature.

This is a hypothesis to test; no implemented comparison ranks it above the other follow-ups.

---

## Priority 2 — extend selective recovery beyond categorical target encoding

The implemented TE2 comparison does not isolate a causal benefit from selective recovery. A matched-base control would be needed before that conclusion.

Potential next candidates:

- verified ordinal encoding;
- safe numeric restoration;
- log transforms for genuinely continuous skewed variables;
- individually validated raw/derived concept alternatives.

Again: one group at a time, with ablation.

---

## Priority 3 — blend the two logistic finalists

TE2 and R2 logistic have nearly identical aggregate performance but different representations.

If their errors are not identical, a convex probability blend could improve performance:

\[
p_{\text{blend}}
=
\alpha p_{\text{TE2}}
+
(1-\alpha)p_{\text{R2}}.
\]

Try only a few weights:

```text
α ∈ {0.25, 0.5, 0.75}
```

and retune the blend threshold using held-out predictions. The code saves outer-fold models/indices, not their per-row scores, so producing the necessary OOF predictions requires inference through those saved models. Selecting blend weights on those predictions creates another selection stage and needs separate evaluation; previously selected recipes also overlap this pool.

A probability blend can be implemented with NumPy without retraining these saved models, but prediction regeneration and additional evaluation still have a cost. Neither its benefit nor its priority over feature changes has been measured.

---

## Priority 4 — error-overlap analysis

Before blending, compute:

```text
examples wrong by both models
examples wrong only by TE2
examples wrong only by R2
examples wrong only by SGD
```

If models make almost identical errors, ensembling is unlikely to help.

Different error sets could motivate a blend experiment, but do not guarantee a better probability blend or thresholded classifier.

---

## Priority 5 — modest class-weight experiments

Current experiment fitters pass no class weights, while threshold tuning increases minority recall in the observed comparisons. The code does not establish that weighting would be redundant or harmful.

A small experiment could nevertheless compare:

```text
unweighted logistic + tuned threshold
```

against:

```text
mildly weighted logistic + tuned threshold
```

Do not use the inverse class-frequency weight blindly.

Any weighted-model experiment should reselect its threshold: it changes the fitting objective and may change the score distribution. Weighting alone does not establish whether probabilities are calibrated.

---

## Priority 6 — calibration diagnostics

Since the validation-selected logistic thresholds are far below 0.5, examine:

- reliability / calibration bins;
- positive prediction fraction;
- score histograms.

These diagnostics could assess probability calibration separately from BA threshold selection. The current threshold results alone do not demonstrate miscalibration, and no calibration experiment is implemented.

---

# 20. More ambitious untackled ML strategies

## 20.1 Elastic-net / sparse logistic regression

Potential advantage:

- feature selection;
- reduced redundancy;
- potentially simpler final model.

Cost:

- implementing an L1-capable optimizer correctly is more work.

Not low-hanging fruit.

---

## 20.2 Coordinate-descent or proximal logistic regression

Could support L1 / elastic-net regularization efficiently.

These optimizers are not implemented. Their implementation cost and attainable improvement have not been measured; small differences among current finalists do not bound the remaining performance margin.

---

## 20.3 Carefully implemented decision tree / ensemble

A nonlinear tree model could capture interactions without explicitly generating them.

However:

- 328k rows;
- 321 raw survey features;
- categorical handling;
- the current NumPy/stdlib modeling approach;
- correct fast splitting and pruning;

make a high-quality ensemble expensive to implement from scratch.

No such model is implemented or evaluated in the current project code, so there is no measured tree-versus-linear comparison.

---

## 20.4 More structured nonlinear logistic basis

A cheaper alternative to trees would be limited basis expansion:

- age bins;
- BMI bins;
- selected piecewise terms;
- a few domain-driven interactions.

The earlier engineered block tested only standardized age/BMI squares and produced little BA gain under the fixed evaluator. It does not test these richer bases; any new terms would require their own controlled evaluation.

---

# 21. What we deliberately did not do

These are observable properties or omissions of the current workflows. The explanations below are design rationales, not a recoverable record of developers' original intent.

### We did not trust raw accuracy

Because 91.17% is obtainable without detecting a single positive.

### We did not blindly one-hot every feature

Because of dense-memory cost and survey semantics.

### We did not blindly convert 88/99/999 to NaN

Because survey codes are variable-specific.

### We did not average correlated raw/derived features

Because correlated representations can be in different units or encode different semantics.

### We did not automatically drop every sparse feature

Because presence/missingness can itself be predictive.

### We did not automatically include cardiovascular-looking features

Because of potential target leakage/provenance overlap.

### Experiments do not modify the mandatory functions

They import separate augmented fitters and check protected-file hashes. This preserves the local mandatory API; official autograder success is not established by those checks.

### We did not trust training loss as the classification objective

Because the baseline experiments showed strong loss/metric divergence.

### The implemented workflow does not use leaderboard feedback

There is no leaderboard integration or upload step in the modeling code. It selects using local validation/confirmation data and writes local submissions; activity outside this repository cannot be established from the code.

---

# 22. Final scientific narrative for the report

The strongest short-form narrative is:

> Full-data EDA characterized the survey matrix as strongly imbalanced, heavily categorical and extensively missing, with heuristic sentinel candidates and redundant raw/derived variables. Missingness was associated with labels; the code does not establish its clinical/routing cause. Mandatory-model diagnostics recorded a rank-deficient training Gram matrix, singular normal-equation fits and higher BA for the tested regularized logistic configurations, with learning-rate and intercept-penalty caveats. Longer optimization could lower training loss while reducing minority-class metrics at fixed thresholds. A separate augmented NumPy layer supplies mini-batches, deterministic RNG, validation monitoring, early stopping, optional intercept-exempt regularization and threshold selection while retaining the mandatory API.

> The reusable fit/transform pipeline fits data-dependent preprocessing on training partitions, including OOF target encoding where requested. Under one fixed evaluator, minimal A0/A1 tied above the more structured variants; this bundled comparison does not attribute the difference to information removal alone. The final experiments tuned three representations for L2 logistic, compared compact/selective OOF target encoding, and evaluated MSE-SGD on the actual winning R2 base. TE2's small advantage also changes regularization, and its recovered fields remain subject to the configured provenance exclusions rather than a proof of semantic safety.

> Three-fold finalist confirmation gave mean balanced accuracies of 0.7757 for TE2 logistic, 0.7745 for R2 logistic and 0.7710 for MSE-SGD. TE2 obtained the highest observed mean, with no demonstrated material or statistically tested advantage. Confirmation fits preprocessing within inner-training partitions, but earlier recipe selection overlaps the confirmation pool, so it is not an independent or fully nested evaluation. More complex models were not evaluated, and the code does not establish their relative value. Each finalist was refitted on all labeled rows with frozen settings and exported locally.

---

# 23. Main takeaways

If only a few findings fit into the two-page report, prioritize these:

1. **Imbalance made raw accuracy misleading.**
2. **EDA showed that survey semantics and missingness matter; numeric CSV storage is not numeric meaning.**
3. **Tested regularized logistic configurations scored higher than the unregularized diagnostic runs, with learning-rate/intercept caveats; loss and classification metrics diverged.**
4. **Threshold tuning produced enormous improvements under imbalance.**
5. **Aggressive preprocessing did not monotonically improve performance; this was an important negative result.**
6. **Selective target encoding added seven previously excluded categorical sources, but its small measured gain is confounded by regularization retuning.**
7. **The three final linear approaches were close, with TE2 logistic highest at 0.7757 3-fold BA.**
8. **The small final differences mean evaluation rigor matters more than claiming a dramatic algorithmic win.**

These are the measured findings and limitations available for the project report.

---

# 24. Decision log: append-only project reasoning

## 2026-10-27 - Preserve decision history separately from the current guide

Decision: `agentic_readme.md` links to this bible as the detailed record of the reasoning behind project design choices. Agents must append new design decisions here before finishing their work, while keeping the guide's repository map and current status concise and up to date.

Rationale: preserving earlier reasoning, including rejected alternatives and later reversals, lets future agents understand how the project evolved without losing context when the operational guide is refreshed.

Policy: append dated entries at the end with the decision, rationale, supporting evidence/artifact links and relevant alternatives or tradeoffs. Never edit, delete or reorder existing entries. Record corrections and superseding decisions in new entries that reference the earlier section; historical statements may no longer describe the current implementation.

Reference: [Agentic Repository Guide](agentic_readme.md).


## 2026-10-27 - Reconcile reproducibility with executed run artifacts

The user explicitly authorized a one-time in-place refactor of this bible. Sections 10-12 and 17-18 now replace approximate search descriptions with the three executed run configurations, distinguish five-fold OOF target encoding from three-fold finalist confirmation, and separate exploratory winners from frozen all-training refits. Available selection/CSV/final metadata also resolved the former winner-specific placeholders. No model code or run artifacts were changed and no training was rerun.

Rationale: a search grid cannot establish selected hyperparameters, checkpoints or inference thresholds; recording artifact provenance prevents those stages from being conflated. Earlier descriptions of patience 20, incorrect evaluation intervals, and an MSE sqrt-schedule experiment were corrected using saved settings and execution code. This explicit exception does not relax the append-only policy for future changes.


## 2026-10-27 - Converged Newton L2 logistic (Attempt 4) and matched-lambda TE2 control

Decision: add `fit_reg_logistic_newton` (full-batch Newton/IRLS with exact chunked Hessian and Armijo backtracking, intercept-exempt L2, gradient-tolerance stopping) to `Mandatory Functions/implementations_augmented.py`, and a fourth experiment `ml_attempts/04_logistic_newton/`. It fits the Attempt 1 winning base (R2 curated) and its TE2 selective-recovery extension (alpha 20) at every lambda of one shared grid {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, each to full-objective gradient norm <= 1e-6 (cap 50 steps). The overall winner goes through the unchanged shared 3-fold confirmation and all-row refit; for Newton these run to the same tolerance instead of rescaling a selected update budget. The mandatory `implementations.py` is unchanged.

Rationale: Sections 10, 11 and 15.3 recorded that every selected logistic run stopped at the 5,000-update cap (convergence not established) and that TE2's gain over R2 was confounded by a different selected lambda. Newton removes both confounds: about 550 dimensions make the Hessian solve cheap (5-7 s per full-data fit), and quadratic convergence reaches the regularized optimum in 5-7 steps. With an exact optimum there is no early stopping; lambda alone regularizes.

Evidence (same 262,508 / 65,627 split, split SHA256 0e013bf3...; `ml_attempts/04_logistic_newton/outputs/`):
- All 16 fits stopped with `tolerance` in 5-7 Newton steps. Local unit tests confirm agreement with 20,000-step full-batch GD and with the mandatory `reg_logistic_regression` to about 1e-10.
- Matched-lambda TE2 minus R2 selected BA was positive at all eight lambdas: +0.00094, +0.00084, +0.00142, +0.00028, +0.00001, +0.00029, +0.00039, +0.00047 (`matched_lambda_results.csv`). This is the controlled encoding comparison that Section 11 lacked; the effect is small (at most 0.0014 BA) and below the 0.002 material-gain criterion at every lambda.
- Solver effect at the selected configurations (`solver_comparison.csv`): R2 Newton (lambda 3e-4) 0.77915 versus GD run_018 0.77791 (+0.00124); TE2 Newton (lambda 1e-4) 0.78019 versus GD 0.77913 at the same lambda (+0.00105). The capped GD fits were therefore measurably undertrained, not merely unconfirmed.
- Selected exploratory run: run_011, N-TE2 selective recovery, lambda=1e-4, 7 steps, BA 0.78019, F1 0.35502.
- 3-fold confirmation (same protocol hash as Attempts 1-3): mean BA 0.7790, fold SD 0.0014, mean F1 0.3612. Per-fold paired differences on identical outer folds: versus GD TE2 (Attempt 2) +0.0037/+0.0038/+0.0025; versus R2 GD (Attempt 1) +0.0051/+0.0050/+0.0036; versus MSE-SGD +0.0058/+0.0090/+0.0093. Three folds are not a significance test, but the sign is consistent on every fold.
- All-row refit: 7 Newton steps to tolerance, frozen threshold 0.08467454931987088 (median inner threshold), 547 features, test predicted-positive fraction 0.3107. Its labels agree with the Attempt 2 submission on 98.1% of test rows.

Interpretation: most of the confirmed gain of Attempt 4 over Attempt 2 comes from optimization (reaching the optimum), not from representation; the encoding effect itself is consistently positive but small. This supersedes the Section 11 caveat that the TE2 gain could not be separated from lambda retuning, and the Section 15.3 open question about logistic undertraining for these representations. Lambda and threshold are still validation-selected and recipe selection still overlaps the confirmation pool, so the Section 13.3 caution about conditional confirmation still applies.

Alternatives considered: raising the GD update cap (slower and still needs a stopping rule), an L-BFGS fitter (more code for no benefit at this dimension), and a self-contained attempt folder with its own confirmation loop (rejected: it would duplicate `confirmation.py`, against the repository rules).

Reproducibility note: the full pipeline (EDA, baseline diagnostics, processing, ablations, Attempts 1-3) was regenerated on 2026-10-27 in this checkout with Miniconda Python 3.13.5 / NumPy 2.2.6; all numbers in Sections 2-18 that were rechecked (baseline best BA, A0-A6 ablation, 541-dimension curated state, Attempt 1-3 exploration, confirmation means/SDs, frozen thresholds and refit update counts) reproduced exactly.


## 2026-10-27 - Codebook-decoded feature recovery (Attempt 5) and finalist blending (06_finalist_blend)

Decision: add Attempt 5 (`ml_attempts/05_feature_recovery/`). It applies three cumulative blocks to the Attempt 1 base, all fitted with the Attempt 4 Newton fitter on one shared lambda grid {3e-5, 1e-4, 3e-4, 1e-3}:
- F1 decodes day/count/drinking fields as numeric using codebook codes (88 = none -> 0; 77/99 = missing; `ALCDAY5` converted to drinking days per 30; `_DRNKWEK` 99900 = missing, then log1p).
- F2 recovers `DIABETE3`, `CHECKUP1`, `CHOLCHK`, `MARITAL`, `EDUCA`, `EMPLOY1` and `INCOME2` as one-hot fields after decoding don't-know/refused codes to missing (`onehot_limits` 8).
- F3 adds seven age x condition dummy interactions plus the existing age/BMI squares.

Also add a post-confirmation step (`ml_attempts/blending.py`, `ml_attempts/06_finalist_blend/`). It rebuilds out-of-fold scores from the saved confirmation-fold models, measures pairwise error overlap, and tests convex probability blends with the best logistic finalist, choosing alpha and the threshold on inner stopping rows only. Only existing `feature_processing` settings were used; neither ML implementation file changed.

Rationale: Section 15.3 asked whether the curated representation underfits. Before this attempt, R2 dropped 33 `unverified_numeric` and 41 `unused_medium_cardinality` fields, including diabetes status, which is partition-dependent: on the full split code 7 clears rare support, giving 6 effective levels, above the limit of 5. The only nonlinear terms tested were the squares (+0.0001). Converged Newton fits make a matched-lambda block ablation cheap and free of the update-cap confound. Blending was the remaining open follow-up recorded in `agentic_readme.md`. The overlap analysis also provides the error analysis that the report guidelines ask for.

Code verification: each mapping follows the BRFSS 2015 codebook value labels. It was spot-checked in `x_train`, where the codes appear as isolated high-frequency values. For example, `PHYSHLTH` has 88 on 204,264 rows, `ALCDAY5` has 888 on 154,556 rows, and `_DRNKWEK` has 99900 on 18,967 rows. Quarantine is unchanged.

Evidence (same split, SHA256 0e013bf3...; `ml_attempts/05_feature_recovery/outputs/`):
- Every fit (16) stopped at `tolerance` in 6-7 Newton steps. Feature counts were F0 541, F1 560, F2 608 and F3 617.
- Matched-lambda block gains in selected BA (`matched_lambda_results.csv`):
  - F1-F0: +0.00076, +0.00095, +0.00046, +0.00039.
  - F2-F1: +0.00131, +0.00131, +0.00200, +0.00141.
  - F3-F2: +0.00037, +0.00066, +0.00008, -0.00019.
  - The categorical recovery block is the largest and most consistent. Interactions add little and are negative at the largest lambda.
- Best-lambda F3 minus F0: +0.00254 BA, above the predeclared 0.002 material-gain criterion. The selected run was run_014 (F3, lambda=1e-4, 7 steps): BA 0.78169, F1 0.36519. That is +0.00150 over the Attempt 4 selection on the same split.
- 3-fold confirmation (same protocol hash as Attempts 1-4): mean BA 0.7812, fold SD 0.0015, mean F1 0.3647.
  - Paired per-fold differences versus Attempt 4: +0.0022/+0.0028/+0.0014.
  - Versus Attempt 2: +0.0059/+0.0066/+0.0040.
- All-row refit: 7 Newton steps to tolerance and 616 features. The frozen threshold is 0.08597367186933232; it happens to equal fold 1's inner threshold, which is the median of the three. The test predicted-positive fraction is 0.3104. Labels agree with the Attempt 4 submission on 97.1% of test rows.
- Blend (`ml_attempts/06_finalist_blend/outputs/`):
  - Rebuilt out-of-fold scores reproduce every stored confirmation mean exactly (`reproduction_check.csv`).
  - Pairwise error-set Jaccard is 0.76-0.96. The most different pair is MSE-SGD versus Attempt 5 (0.756, disagreement 7.2%), and the Newton pair 04/05 scores 0.889.
  - With Attempt 5 as anchor, inner stopping rows chose alpha = 1 (anchor alone) in 8 of 9 fold/partner cases. The one exception (alpha 0.75 with Attempt 1, fold 3) lost 0.0008 on the outer rows.
  - No blend is recommended, and none was exported. `recommendation.json` points to the Attempt 5 submission.
- Error analysis of the anchor (`error_analysis.csv`, out-of-fold):
  - Errors follow the base rate. Recall is 0.29 at ages 18-44 (prevalence 1.3%) and 0.94 at 75+ (prevalence 20.6%), where specificity falls to 0.28.
  - Similarly, GENHLTH = poor has recall 0.98 and specificity 0.18.
  - With a single global threshold, the model flags almost every older or unhealthy respondent and misses most young positives. Per-group BA ranges from 0.58 to 0.75, against 0.78 overall.

Interpretation:
- Recovering fields the curated profile had discarded for cardinality or unknown codes helps more than target-encoding them (Attempt 4's TE2 block: at most +0.0014). Hand-picked interactions are close to neutral.
- Blending linear models that share most of their inputs gives no measurable benefit, because their errors overlap heavily. This answers the blending follow-up negatively for these finalists.
- The error analysis suggests that remaining gains need features that separate risk within an age/health stratum, not more global capacity.
- This supersedes Section 15.3's open underfitting question for these field groups. Lambda and threshold are still validation-selected, so the Section 13.3 caution about conditional confirmation still applies.

Alternatives considered:
- Target-encoding the recovered categoricals instead: already tested as TE2, with a smaller gain.
- Decoding the raw fruit/vegetable frequency fields: rejected for their complex unit codes; the derived `_FRUTSUM`/`_VEGESUM` remain excluded and untested.
- Piecewise age bins: not supported without new preprocessing code; the age square covers curvature.
- Blending in score space with the MSE finalist: would need a fitted calibration map, so it was left to the overlap table only.

## 2026-10-27 - Remove the blending step (06_finalist_blend)

Decision: at the user's request, `ml_attempts/blending.py`, `ml_attempts/06_finalist_blend/` (including its outputs) and the orchestration/test hooks were removed. This partially supersedes the previous entry: Attempt 5 stays, the blending step does not. Its recorded findings remain valid evidence (error-set Jaccard 0.76-0.96 between finalists; no inner-selected blend beat Attempt 5 alone; out-of-fold errors track the base rate across age and general-health strata), but the artifacts no longer exist in the checkout and would need the removed code to regenerate. No submission changed: the selected submission is still `ml_attempts/05_feature_recovery/outputs/submission.csv`.

## 2026-10-27 - Submission header matches the sample file

Decision: `write_submission` (`ml_attempts/confirmation.py`) now writes the header `Id,Prediction`, exactly as `dataset/sample_submission.csv` does, instead of `Id,_MICHD`. The five existing `submission.csv` files had only their header line rewritten; labels, IDs, ID order and CRLF line endings are unchanged, and each file was checked against the sample (same 109,379 IDs in the same order, labels in {-1, 1}). `test_predictions.csv` keeps its internal `_MICHD` column name. Rationale: the platform may match columns by name, so the sample format is authoritative.

## 2026-10-27 - Select Attempt 5 thresholds and runs by F1 (leaderboard metric)

Decision: the BA-threshold Attempt 5 submission scored F1 0.370 / accuracy 0.750 on the platform, which ranks by F1 and then accuracy. Attempt 5 (`config.py`: `selection_metric: 'f1'`) now ranks runs by validation F1 and picks every threshold (exploration, inner confirmation folds, frozen final threshold) by F1. Its lambda grid was widened to {1e-5, ..., 3e-3}. The new `selection_metric` setting lives in `ml_attempts/common.py`, and `confirm_finalist` passes it to `find_best_threshold`. Attempts 1-4 keep the BA default, and their recorded results are unchanged. `summary.md` is now ranked by 3-fold F1, then accuracy, and shows each attempt's threshold metric. This supersedes the earlier convention, first used in Section 2, that BA is the selection metric everywhere; BA remains reported.

Rationale: the platform result reproduced the local 3-fold estimate at the BA threshold (F1 0.365, accuracy 0.747), so the gap was the operating point, not overfitting. With about 9% positives, the BA-optimal threshold (about 0.086) flags 31% of rows and produces mostly false positives. A pre-change check on the saved BA-selected fold models gave outer F1 0.422 / accuracy 0.857 with an inner-F1 threshold (about 0.18) on every fold.

Evidence (`ml_attempts/05_feature_recovery/outputs/`):
- All 24 Newton fits converged in 6-7 steps.
- Matched-lambda block gains in F1:
  - F1-F0: +0.0009 to +0.0017.
  - F2-F1: -0.0001 to +0.0030.
  - F3-F2: -0.0004 to +0.0028.
- Best-lambda F3 minus F0: +0.0041 F1.
- Selected run_020: F3, lambda=3e-5, exploratory F1 0.42828 / accuracy 0.87519 / BA 0.71905.
- 3-fold confirmation (same protocol hash): F1 0.4221 (SD 0.0001), accuracy 0.8581, BA 0.7356, precision 0.330, recall 0.587. Fold thresholds were 0.1802, 0.1860 and 0.1808.
- Versus the BA-threshold Attempt 5 confirmation: F1 +0.0574, accuracy +0.111, BA -0.0456.
- Choosing lambda by F1 rather than BA added only about +0.0003 F1 over thresholding the earlier lambda=1e-4 model (0.4218 in the pre-change check). Nearly all of the gain is the threshold.
- All-row refit: 7 Newton steps, frozen threshold 0.1807767704909474, test predicted-positive fraction 0.159.
- The new submission agrees with the uploaded BA-threshold submission on 84.8% of test rows; the differences are the flags removed by the higher threshold.
- The exploratory F1 (0.428) exceeds the confirmed F1 (0.422), as expected after selecting on validation.

## 2026-10-27 - All attempts select by F1; full pipeline rerun

Decision: Attempts 1-4 also set `selection_metric: 'f1'`, so every attempt ranks its search stages and picks every threshold by F1, the leaderboard metric. This extends the previous entry from Attempt 5 to all attempts. All internal choices in `searches.py` now use the configured metric, including the Attempt 1 lambda/gamma stages and optional-engineering gain, Attempt 2's competitive check and its gain versus the benchmark, and the Attempt 3 gamma/batch/schedule stages. `matched_lambda_results.csv` (Attempt 4) adds F1 columns, and `solver_comparison.csv` adds an F1 difference. The BA-era outputs of Attempts 1-4 were overwritten; a scratch backup was kept outside the repository, and their numbers remain recorded in earlier entries.

Evidence (`python -B -m ml_attempts.run_all`, same split and confirmation protocol):

| Attempt | Selected recipe | 3-fold F1 | Accuracy | BA | Frozen threshold | Test positive fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 05 | F3, lambda 3e-5 | 0.4221 | 0.8581 | 0.7356 | 0.1808 | 0.159 |
| 04 | N-TE2, lambda 1e-5 | 0.4178 | 0.8647 | 0.7225 | 0.1951 | 0.143 |
| 02 | TE1 curated compact, lambda 1e-4 | 0.4135 | 0.8601 | 0.7239 | 0.1847 | 0.149 |
| 01 | R2 curated, lambda 1e-3, gamma 0.1 | 0.4125 | 0.8599 | 0.7231 | 0.1946 | 0.138 |
| 03 | S0 R2 curated MSE-SGD, gamma 0.03, batch 256 | 0.4060 | 0.8523 | 0.7255 | -0.6017 (score) | 0.262 |

- Attempt 5 reproduced its F1-selected result exactly.
- Under F1, Attempt 2 selects TE1 rather than TE2, and Attempt 1 selects lambda 1e-3 / gamma 0.1.
- The ranking of attempts is identical to the BA ranking (05 > 04 > 02 > 01 > 03).
- Attempt 5 beats Attempt 4 on all three folds (fold F1 0.4221/0.4220/0.4222 versus 0.4163/0.4187/0.4182).
- The F1 spread between attempts (0.406-0.422) is much smaller than the gain from the threshold change (about +0.06 F1 per attempt).
- Fold SD of BA rises at F1 thresholds (about 0.012-0.014 for GD/Newton), because BA is sensitive to where the higher threshold falls; fold SD of F1 stays at or below 0.003 for every attempt.

## 2026-10-27 - Release prior stroke (CVDSTRK3) from quarantine as Attempt 5 block F4

Decision: Attempt 5 gains a fifth cumulative block. F4 = F3 + `CVDSTRK3` ("ever told you had a stroke"), removed from the default quarantine and encoded as a verified binary field (1 = yes -> 1, 2 = no -> 0, 7/9 -> missing, with the pipeline's missing indicator). Every other quarantined field is unchanged. The selected finalist and submission are now F4.

Rationale and leakage review:
- The default quarantine excluded all cardiovascular review fields (Section on leakage review; `LEAKAGE_REVIEW_FEATURES`) without separating risk factors from consequences.
- The label `_MICHD` is defined by reported heart attack, angina or coronary heart disease. Those source questions (`CVDINFR4`, `CVDCRHD4`) and `_MICHD` are absent from `x_train.csv`.
- Prior stroke is a distinct cerebrovascular diagnosis, asked of all respondents, and a recognized comorbidity sharing atherosclerotic risk. It plays the same role as the diabetes and hypertension fields already used.
- It is therefore a predictor, not a restatement of the label. By contrast:
  - `HAREHAB1` (cardiac rehabilitation) is asked only after a heart attack and is 100% positive in training.
  - `CVDASPRN`, `RDUCHART` and `RDUCSTRK` record treatment taken because of heart or stroke risk.
  - These remain quarantined as post-event or label-proxy fields, as do `STREHAB1` and the survey/administrative metadata.
- Motivation from the error analysis (2026-10-27 Attempt 5 entry): residual errors were concentrated within age and general-health strata, so information distinguishing risk inside those strata was needed. Stroke history: 37% MICHD rate when yes (13,366 rows) against 8.8% overall.

Evidence (same split, F1 selection, same confirmation protocol hash):
- Matched-lambda F4-F3 F1 gains: +0.0073, +0.0071, +0.0084, +0.0085, +0.0097, +0.0090. All are positive and the largest of any block (F1-F0 at most 0.0017, F2-F1 at most 0.0030).
- Best-lambda F4 minus F0: +0.0120 F1. Selected run: F4, lambda 1e-4, exploratory F1 0.43623.
- 3-fold confirmation:
  - F1 0.4279 (SD 0.0011), accuracy 0.8674, BA 0.7294, frozen threshold 0.1924.
  - Per-fold F1 0.4267/0.4280/0.4289 against F3's 0.4221/0.4220/0.4222, a gain of +0.0046/+0.0060/+0.0067 on every fold.
- All-row refit: 7 Newton steps, 619 features, test predicted-positive fraction 0.144.
- The previous F3 submission is superseded; a copy of its outputs was kept outside the repository.

## 2026-10-28 - Final model, leaderboard result and submission packaging

Decision: Attempt 5 F4 (2026-10-27 stroke entry) is the final model. Feature processing is frozen.

Evidence:
- The F4 `submission.csv` scored **F1 0.442** on the AIcrowd leaderboard, our best upload. The 3-fold estimate was 0.428, so the public score sits slightly above local CV, within the expected spread.
- A residual probe on the F4 validation predictions (65,627 rows) grouped rows by each raw field's values (quantile bins for wide fields) and computed z = sum(y - p) / sqrt(sum p(1 - p)) per cell.
  - Fields: no field, dropped or kept, had a cell beyond |z| = 3.2, which matches what chance gives over about 1,000 cells. Per-field sums of z^2 were 10-29 against about 10-15 expected from noise.
  - Pairs of 18 key risk factors: the worst crossed cells reached about |z| = 4, again near chance. The largest real misfit (respondents without high cholesterol, predicted 4.3% vs observed 3.9%) is too small to move F1.
  - Conclusion: the linear model has used the available information; further gains would need a nonlinear model or the excluded post-event fields, which stay excluded.

Packaging (spec, Step 4), no change to model behaviour:
- `implementations.py` moved to the repository root, content unchanged, where the autograder imports it. The path references in `baseline_diagnostics.py`, `io_utils.py` and the two `Mandatory Functions/` test files were updated.
- New top-level `run.py`: loads the frozen choices from `final_model/frozen_configuration.json` (a copy of `05_feature_recovery/outputs/frozen_final_configuration.json`), refits preprocessing and the Newton model on all 328,135 rows, and writes `submission.csv`.
  - It is byte-identical to the 0.442 upload and takes about 65 s.
  - It retrains rather than loading weights, so no binary artifacts are committed.
  - The 15-minute search and 3-fold confirmation remain in `ml_attempts/05_feature_recovery/run.py`.
- New root `README.md` covering setup, data preparation, feature generation, the validation protocol and results.
- Every module, class and function, tests included, now has a short docstring. An AST comparison against the previous commit, ignoring docstrings, confirms no other code changed. All 94 tests pass.
