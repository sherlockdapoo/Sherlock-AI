# Sherlock GPT v0.10.2 test: character-composed word inputs

This is an isolated experiment. It does not change the stable v0.9.9 homepage, model, or checkpoint.

## What changed

- Keeps the full **1,024-token word vocabulary**. The 37 character embeddings do not take slots away from frequent words.
- Keeps **one token per original word**. An unseen alphanumeric word stays one training position and maps to the `<unk>` output target.
- Adds a separate table of **37 character embeddings**, each 8 values wide. When an unseen word appears in the input context, its input vector is the mean of its character embeddings.
- Character embeddings learn through normal backpropagation when rare words appear in context.
- Returns to an **8-word context**, matching the stable model's sequence length.
- Uses its own browser checkpoint key and worker cache version.

## Model size

- Vocabulary: 1,024 word-level output classes
- Context: 8 tokens
- Embedding: 8 values
- Hidden layer: 48 ReLU units
- Character table: 37 × 8 = 296 additional values
- Total trainable parameters: **61,784**

## How to interpret the metrics

The prediction target remains the same word-level task used by v0.9.9: predict the next word-or-punctuation token. A rare word in the target position maps to `<unk>`; that same rare word in the input context receives a character-composed vector. Accuracy and validation loss are therefore more directly comparable with v0.9.9 than the character-expansion experiments were.

The OOV audit reports how many validation words receive character-composed input vectors. It does not mean the number of unknown output targets has fallen.

## Training

Training still runs locally in the browser worker using the same data loader, training split, held-out validation split, batch size, AdamW optimiser, and learning rate as the previous isolated experiment. This model starts fresh and cannot load checkpoints from v0.9.9, v0.10.0, or v0.10.1.

This is a test candidate, not a replacement for the stable model. Compare speed, training loss, validation loss, and accuracy after a similar amount of training. No performance improvement is assumed until the real device run confirms it.

## Stable version protection

Only this experiment folder is added. The root homepage and original neural worker are untouched.
