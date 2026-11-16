# Sherlock AI v0.14.0 standalone curriculum experiment

## Dataset
- Training: 10,000 conversations in `data/sherlock-curriculum-v0140-train.txt`.
- Held-out validation: 1,000 independently generated conversations in `data/sherlock-curriculum-v0140-validation.txt`.
- Training examples retained from v0.13.0: 2,174.
- Additional training examples: 7,826.
- Validation contains new task instances, not prompt rewrites of the v0.13.0 validation seeds.
- Exact normalized prompt overlap with training: 0.
- Duplicate validation prompts: 0.
- Malformed role blocks: 0.
- Validation file size: 72083 characters; within the 128 KiB loader limit.

## Validation coverage
- 300 arithmetic questions: addition, subtraction, multiplication, and exact division.
- 100 percentage questions.
- 100 unit-conversion questions.
- 150 multi-step word problems.
- 150 number-pattern questions.
- 200 letter and instruction questions using unseen labels.
- This is a synthetic benchmark of held-out task instances, not a measure of broad real-world knowledge.

## How the training set was made
The v0.13.0 standalone curriculum is preserved as the seed set. Training examples add alternate prompt phrasings and new arithmetic practice. Many entries are controlled rephrasings, not 10,000 entirely independent facts. This increases prompt variation and arithmetic practice, but does not guarantee better answers.

## Model isolation
- Standalone files only. No external dataset downloads.
- Architecture remains context 32, embedding 8, hidden 48, vocabulary 1,024 (70,704 parameters).
- v0.14.0 has independent checkpoint keys and dataset revision.
- v0.13.0 and stable v0.9.9 remain preserved.

## Evaluation
Compare real browser training speed, train and validation loss, next-token accuracy, unknown-token rate, and the same five-question answer benchmark. Validation is now separated by task instances from training, although broad skill types are intentionally shared. Keep stable v0.9.9 unless repeated comparisons show a reliable improvement.
