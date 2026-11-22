# Sherlock AI knowledge curriculum v0.19.0

This supplement expands the v0.18.7 conversation family with short factual question-and-answer examples.

## Contents

- Training: 51 examples
- Held-out validation: 15 examples
- Topics: astronomy and space, general science, matter and energy, computing and AI, geography, animals, history, conservation, and scientific methods.
- Format: alternating `user:` and `sherlock:` turns, separated by blank lines.

## Split policy

Validation prompts are separate from training prompts. No validation prompt is duplicated in the training supplement. The app also checks the combined curriculum for cross-split prompt duplication.

## Model compatibility

The supplement uses the frozen v0.18.7 token IDs. New vocabulary items are appended into unused rows of the existing 1,024-token output vocabulary; legacy token IDs are not reordered. This keeps the existing v0.19.1 model parameter shape and old token mapping intact.

The checkpoint compatibility identifier is `sherlock-word-token-v0191-stable-vocab-v0186`. Dataset revision and model compatibility are tracked separately so a dataset update does not force a fresh model.

## Limitations

These are small, curated examples for foundational factual language patterns. They do not provide live information or guarantee factual accuracy. Training on them may improve responses on related topics, but does not establish general intelligence or broad factual reliability.
