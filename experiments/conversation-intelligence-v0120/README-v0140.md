# Sherlock AI v0.14.0 standalone curriculum experiment

## Dataset
- Training: 10,000 conversations in `data/sherlock-curriculum-v0140-train.txt`.
- Held-out validation: 1,000 conversations in `data/sherlock-curriculum-v0140-validation.txt`.
- Training file: 670,202 bytes.
- Validation file: 73,637 bytes.
- Source training conversations retained: 2174.
- Unique prompt rewrites added: 6,472.
- New arithmetic training examples: 1,354.
- Held-out source conversations retained: 225.
- Unique held-out prompt rewrites added: 675.
- New held-out arithmetic examples: 100.
- Exact validation/source-training overlap: 0.
- Duplicate blocks within either final split: 0.

## How it was made
The v0.13.0 standalone curriculum is preserved as the seed set. Training examples receive alternate prompt phrasings, with memory-oriented lead-ins for multi-turn conversations. Fresh arithmetic examples fill the remaining training and validation counts using separate number ranges.

Many entries are controlled rephrasings of existing examples, not 10,000 entirely independent facts. This increases prompt variation and arithmetic practice, but does not guarantee better answers.

## Model isolation
- Standalone files only. No external dataset downloads.
- Architecture remains context 32, embedding 8, hidden 48, vocabulary 1,024 (70,704 parameters).
- v0.14.0 has independent checkpoint keys and dataset revision.
- v0.13.0 and stable v0.9.9 remain preserved.

## Evaluation
Check real browser training speed, train and validation loss, next-token accuracy, unknown-token rate, and the same five-question answer benchmark. Keep stable v0.9.9 unless repeated comparisons show a reliable improvement.
