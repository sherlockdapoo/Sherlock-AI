# Sherlock AI v0.11.0: assistant-focused training experiment

This isolated browser experiment tests a different training objective without changing the stable model architecture.

## What changes

The stable word-level model samples next-token targets uniformly from the entire flattened conversation stream. This experiment samples targets only from assistant response bodies marked by the existing `sherlock:` turn labels.

The input context still comes from the full conversation stream. That means a response's first tokens can be trained using the preceding user question, while later tokens can use the earlier part of the response.

Validation also measures only assistant-response target positions. These scores are **not directly comparable** with the stable model's all-token validation scores.

## What stays the same

- Vocabulary: 1,024 IDs.
- Context: 8 tokens.
- Embedding size: 8.
- Hidden layer: 48 ReLU units.
- Parameters: 61,488.
- Batch size: 32.
- Optimizer: AdamW, learning rate 0.001 by default.
- Same dataset sources and existing browser training worker architecture.

## Why test it

A flattened corpus includes both user questions and assistant answers. Uniform random sampling spends many updates learning to predict user-side text. Restricting targets to assistant turns may focus the same small model on answer generation.

This is a hypothesis, not a guaranteed improvement. The smaller target pool may increase repeated exposure to some response patterns, and the model still has only an eight-token context window.

## Safety and isolation

This page uses separate checkpoint, archive and legacy-storage keys and a unique worker URL. It does not overwrite the stable v0.9.9 checkpoint or its files.

Training still happens locally in the browser. Automated synthetic tests can check parsing, target selection, gradients, checkpoints and generation, but actual device speed and real-dataset quality need a browser run.
