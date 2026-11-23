# Sherlock AI training-pipeline audit

Audit date: 2026-11-23  
Audited worker: `neural-worker-v02149.js`  
Audited website: `index.html` (v0.21.70; curriculum identifier v0.19.2)

## Executive finding

The current training-loss and validation-loss curves are **not directly comparable**. The visible gap therefore cannot be interpreted as a pure measure of overfitting without a like-for-like evaluation.

This is a confirmed measurement mismatch in the current worker, not proof that it explains all of the rising validation loss.

## Confirmed findings

### 1. Training loss is measured with dropout enabled

- `trainExample()` calls `forward(..., true)`.
- The forward pass applies 15% hidden-layer dropout during training.
- The reported training moving average is built from these dropout-affected batch losses in `runBatch()`.

### 2. Validation loss is measured with dropout disabled

- `evaluateAsync()` calls `predict()`.
- `predict()` calls `forward()` without the training flag, so dropout is disabled.
- Validation cross-entropy is averaged over those inference-mode predictions.

**Impact:** the training curve includes the extra difficulty/noise caused by dropout, while validation does not. This is an inconsistent comparison. The direction and size of the difference must be measured; it should not be assumed.

### 3. Training and validation use different sampling distributions

- Training positions are selected by task family, with family probability proportional to the square root of that family's number of training target-token positions.
- Within the selected family, a target position is sampled uniformly.
- Validation instead takes up to 640 target positions at evenly spaced intervals across the complete validation target-position array.

**Impact:** training loss is a sampler-weighted estimate, while validation loss is approximately a corpus-token-weighted estimate. If task-family proportions or response lengths differ, their averages can differ even when per-family performance is similar.

### 4. Task-family classification is heuristic

The worker assigns a prompt to a family using keyword/regular-expression checks. For example, arithmetic is identified through words such as “plus”, “minus”, “multiplied”, or “divide”; number-only expressions do not necessarily match those words and can fall through to the conversation family.

**Impact:** the square-root sampler's effective family distribution depends on this classifier. The family labels should be audited against the actual prompts before changing sampling weights.

### 5. Validation is not currently reported by task family

The worker builds validation target positions and context-start indices, but unlike training it does not retain validation positions grouped by task family. The current single validation score can hide a family that is improving while another is deteriorating.

## Recommended next implementation

Add a **diagnostic-only, inference-mode training evaluation** alongside validation:

1. Evaluate a fixed, deterministic sample of training target positions with dropout disabled.
2. Evaluate validation positions using the same inference path and compatible sampling/weighting.
3. Report per-family training and validation cross-entropy, plus sample counts.
4. Keep the existing live training-loss curve as a separate optimisation signal.
5. Add checks for empty/misclassified families and record the actual family counts.

This would separate optimisation loss from comparable generalisation metrics.

## Protected areas

This audit made no changes to:
- model architecture or parameter count;
- optimiser, learning rate, weight decay, or dropout settings;
- training sampler or training algorithm;
- tokenizer or vocabulary IDs;
- checkpoint format, saved model weights, or optimiser state;
- automatic training or stopping behaviour.

## Important limitation

The audit identifies measurement and sampling inconsistencies in the code. It does **not** establish that these are the only causes of the observed validation-loss increase. The next diagnostic should measure the training and validation sets consistently before any training-setting change is proposed.
