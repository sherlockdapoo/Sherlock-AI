# Sherlock AI v0.17.0 Balanced Learning Experiment

## Purpose
This isolated experiment tests two evidence-led changes identified during the v0.16.2 review:
1. Balanced coverage across six task families, addressing weak or absent training coverage for several held-out task families in v0.14.0.
2. Digit-aware tokenisation. Multi-digit numbers are split into individual digit tokens, allowing unseen numeric values to reuse learned digit representations rather than becoming a single unknown token.

## Dataset
- Training: 12,000 examples, exactly 2,000 per task family.
- Validation: 1,200 independent examples, exactly 200 per task family.
- Families: arithmetic, percentages, unit conversions, word problems, number sequences, and letter/instruction tasks.
- Exact normalized prompt overlap: 0. Duplicate validation prompts: 0. Malformed role blocks: 0.
- Validation file size: 91577 bytes, below the 128 KiB loader limit.
- Validation unknown-token rate under the new tokenizer and training-built vocabulary: 0.100%.

## Model
- Architecture unchanged: vocabulary capacity 1,024, context 32, embedding 8, hidden 48, 70,704 parameters.
- Training settings and early-stopping behaviour copied unchanged from v0.16.2.
- Independent IndexedDB database, browser checkpoint keys and dataset revision. This experiment cannot overwrite the main model or archived v0.14.0 code.
- Standalone dataset files; no external dataset download required.

## Limits and evaluation
This is a synthetic curriculum, not a broad-knowledge dataset. Validation uses independent prompts and numeric instances, while templates intentionally match training to measure task generalisation without also changing the language patterns. Train in the browser, record training/validation loss, accuracy, early-stop count and speed, then run the same five-question benchmark used for earlier versions. Compare against stable v0.9.9 and main v0.16.2. Do not promote without repeated evidence of improvement.
