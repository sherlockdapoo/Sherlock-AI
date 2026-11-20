# Sherlock AI website v0.21.42

Sherlock AI is a browser-based neural-network experiment. Chat and Neural Lab run in the browser; the neural model trains in a dedicated Web Worker on the user's device.

## Current release
- Website: **v0.21.39**
- Corpus-equivalent passes use a monotonic cumulative example counter, so restoring the best validation checkpoint cannot make the displayed total run backwards.
- Exact alphabetic ordering of explicit letter lists is handled by a deterministic operation before neural text generation. This hybrid capability is reliable for character copying/sorting, but is separate from the neural model's learned language-generation accuracy.
- Text generation uses a inline minimum-output-word checks (removing a fragile helper reference), alongside repetition-aware decoding: a mild recent-token penalty, repeated-trigram blocking, a minimum initial output length before EOS, and a slightly narrower top-p nucleus. These decoding changes do not alter model weights or guarantee better reasoning.
- Balanced Learning curriculum: **v0.18.5**
- Training examples: **24,000**
- Held-out validation examples: **1,680**
- Families: arithmetic, percentages, unit conversions, word problems, number sequences, and letter/instruction/language tasks.
- Architecture: word-level MLP, 48-token context, 12-dimensional embeddings, 96 hidden neurons, 1,024-token output vocabulary.
- Parameters: **167,008** (about 2.36× the previous model).
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001, batch size 32.

## Model v0.19.0 upgrade
- The expanded model uses a 48-token context, 12-dimensional embeddings and 96 hidden neurons (167,008 trainable parameters).
- Its checkpoint namespace is separate from v0.18.5, so the earlier 70,704-parameter model remains untouched and available in browser storage.
- The expanded model starts untrained and requires a fresh training run. It uses the existing v0.18.5 curriculum; the larger network increases capacity but does not add new training knowledge by itself.

## Training
Open https://sherlockdapoo.github.io/Sherlock-AI/, choose Neural Lab, and press Start neural training. Training runs locally in a background worker until Stop is pressed. Stop requests a final evaluation and checkpoint save. Keep the page open during a long run; mobile browsers may pause background tabs or suspend work when the device sleeps.

## Data revisions
The v0.18.5 curriculum adds 3,000 training examples and 60 held-out validation examples to v0.18.4, including 1,500 focused letter-ordering examples and 30 held-out letter-ordering tests. Older base and supplement files remain unchanged. v0.18.5 uses a separate dataset cache and checkpoint namespace. The previous v0.18.4 checkpoint is preserved in browser storage but is not automatically loaded: the expanded dataset can change tokenizer ID ordering, so reusing its weights without a verified vocabulary mapping could corrupt the model.

## Scope
The v0.18.5 language-focused examples concentrate on exact preservation and alphabetical ordering of five distinct letters, with separate held-out letter combinations. Deterministic letter ordering is an explicit algorithmic capability, not evidence that the neural model has learned general character-level reasoning. Evaluate neural language generation separately with fresh benchmark questions.


### v0.21.35 — Playground curriculum synchronisation
- Updated the separate Playground to restore the current v0.18.5 checkpoint (including the best checkpoint fallback).
- Playground now loads and combines all four training and four validation curriculum files, matching the Neural Lab vocabulary and dataset revision v0.18.8 cache identity.
- This fixes the stale v0.18.2-only Playground model path; it does not alter saved model weights or trigger retraining.


### v0.21.35 — Minimum meaningful Playground continuation
- Text generation now waits for at least five whitespace-separated output tokens before EOS is permitted, preventing an immediate empty continuation.
- Deterministic maths and letter-order handlers remain unchanged. Decoder-only change; no retraining or checkpoint rewrite.


### v0.21.35 — Generation stop-rule correction
- Corrected the whitespace counter used by the five-word minimum EOS guard, ensuring it counts actual word boundaries.
- Updated worker cache keys. No checkpoint or training-data changes.


### v0.21.35 — Unique worker filename cache bypass
- Added `neural-worker-v02135.js` as an identical copy of the verified v0.21.34 worker, which contains no `hasMinimumOutputWords` references.
- Updated the main Neural Lab and Playground to load the uniquely named worker file, bypassing any stale cached `neural-worker.js` response.
- No model weights, training data, or checkpoints were changed; no retraining is required.


### v0.21.39 — Generation v2 constrained decoding
- Filters digit-only tokens and percentage terms in ordinary prose unless the prompt asks for numeric or percentage content.
- Prevents newline-only continuations, stray closing brackets, repeated punctuation, and punctuation before the first words.
- Uses alphabetic word counts for the minimum output threshold and can stop after a completed sentence.
- Normalises spacing around punctuation. These are decoding heuristics, not new learned knowledge; the saved checkpoint and curriculum are unchanged, and no retraining is required.


### v0.21.39 — generation quality guard
- Filters stray standalone numbers and percentage fragments from non-numeric prompts.
- Replaces very short or punctuation-heavy fragments with a clear learning-status message rather than displaying them as completed answers.
- Does not modify model weights, training data, or checkpoints.


## Question routing — v0.21.39
- Sherlock classifies prompts before answering: percentage, arithmetic, letter sorting, number sequence, unit conversion, word problem, general question, or text continuation.
- Supported percentage/arithmetic, letter sorting, simple arithmetic/geometric sequences, common metric/time conversions, and simple two-number word problems use deterministic specialist solvers.
- Unsupported or ambiguous cases fall back to the trained neural language model. The Playground displays the detected question type and whether a specialist solver or neural generation produced the answer.
- This is an inference-time routing update. It keeps the v0.19.0 checkpoint architecture and existing saved model compatible; no retraining is required for the router itself.
- Specialist solvers are intentionally limited: unsupported unit pairs, complex sequences, or ambiguous word problems fall back to neural generation rather than guessing.


### v0.21.40 — Training scheduler optimisation
- The training worker now processes four training batches per worker task before yielding to the browser event queue, instead of scheduling a separate task after every batch.
- Validation still runs at the same 50-step interval and in the same chunked slices. Batch size, optimizer, learning rate, model architecture, training examples, and checkpoint layout are unchanged.
- This reduces scheduler overhead while retaining regular opportunities for Stop and progress messages to be handled. Actual speed gains depend on the device and browser and must be measured during a run.


### v0.21.41 — Greeting intent routing
- Added a narrow greeting classifier for common standalone greetings such as “Hi”, “Hello”, “Hey”, and time-of-day greetings.
- These prompts receive a concise greeting response through a deterministic response handler rather than the still-limited neural language generator.
- This is a routing improvement, not evidence that the neural model has learned general conversational ability. No retraining is required.


## v0.21.42 — Neural conversation learning

- Neural model label: v0.19.1 (same 167,008-parameter architecture as v0.19.0).
- Added the v0.18.6 conversation supplement: 700 training conversations and 100 separate held-out validation conversations.
- Expanded Balanced Learning from six to seven equally sampled task families. The seventh family contains ordinary conversation prompts and responses, including greetings.
- Removed the deterministic greeting response. Greetings now use the neural language model like other conversational prompts.
- Updated the dataset revision to `sherlock-conversation-curriculum-v0189-2026-11`.
- Updated checkpoint namespace and record version so v0.19.0 checkpoints remain preserved and are not loaded with the changed tokenizer.
- **A fresh training run is required.** The new dataset changes vocabulary and training targets. Existing v0.19.0 training results remain in the old checkpoint.
- Conversation quality is not guaranteed by adding examples alone; it must be measured after training on held-out prompts.
