# Sherlock AI website v0.21.30

Sherlock AI is a browser-based neural-network experiment. Chat and Neural Lab run in the browser; the neural model trains in a dedicated Web Worker on the user's device.

## Current release
- Website: **v0.21.30**
- Corpus-equivalent passes use a monotonic cumulative example counter, so restoring the best validation checkpoint cannot make the displayed total run backwards.
- Exact alphabetic ordering of explicit letter lists is handled by a deterministic operation before neural text generation. This hybrid capability is reliable for character copying/sorting, but is separate from the neural model's learned language-generation accuracy.
- Text generation uses repetition-aware decoding: a mild recent-token penalty, repeated-trigram blocking, a minimum initial output length before EOS, and a slightly narrower top-p nucleus. These decoding changes do not alter model weights or guarantee better reasoning.
- Balanced Learning curriculum: **v0.18.5**
- Training examples: **24,000**
- Held-out validation examples: **1,680**
- Families: arithmetic, percentages, unit conversions, word problems, number sequences, and letter/instruction/language tasks.
- Architecture: word-level MLP, 32-token context, 8-dimensional embeddings, 48 hidden neurons, 1,024-token output vocabulary.
- Parameters: **70,704**
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001, batch size 32.

## Training
Open https://sherlockdapoo.github.io/Sherlock-AI/, choose Neural Lab, and press Start neural training. Training runs locally in a background worker until Stop is pressed. Stop requests a final evaluation and checkpoint save. Keep the page open during a long run; mobile browsers may pause background tabs or suspend work when the device sleeps.

## Data revisions
The v0.18.5 curriculum adds 3,000 training examples and 60 held-out validation examples to v0.18.4, including 1,500 focused letter-ordering examples and 30 held-out letter-ordering tests. Older base and supplement files remain unchanged. v0.18.5 uses a separate dataset cache and checkpoint namespace. The previous v0.18.4 checkpoint is preserved in browser storage but is not automatically loaded: the expanded dataset can change tokenizer ID ordering, so reusing its weights without a verified vocabulary mapping could corrupt the model.

## Scope
The v0.18.5 language-focused examples concentrate on exact preservation and alphabetical ordering of five distinct letters, with separate held-out letter combinations. Deterministic letter ordering is an explicit algorithmic capability, not evidence that the neural model has learned general character-level reasoning. Evaluate neural language generation separately with fresh benchmark questions.


### v0.21.30 — Playground curriculum synchronisation
- Updated the separate Playground to restore the current v0.18.5 checkpoint (including the best checkpoint fallback).
- Playground now loads and combines all four training and four validation curriculum files, matching the Neural Lab vocabulary and dataset revision v0.18.8 cache identity.
- This fixes the stale v0.18.2-only Playground model path; it does not alter saved model weights or trigger retraining.
