# Sherlock AI website v0.21.22

Sherlock AI is a browser-based neural-network experiment. Chat and Neural Lab run in the browser; the neural model trains in a dedicated Web Worker on the user's device.

## Current release
- Website: **v0.21.22**
- Balanced Learning curriculum: **v0.18.4**
- Training examples: **21,000**
- Held-out validation examples: **1,620**
- Families: arithmetic, percentages, unit conversions, word problems, number sequences, and letter/instruction/language tasks.
- Architecture: word-level MLP, 32-token context, 8-dimensional embeddings, 48 hidden neurons, 1,024-token output vocabulary.
- Parameters: **70,704**
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001, batch size 32.

## Training
Open https://sherlockdapoo.github.io/Sherlock-AI/, choose Neural Lab, and press Start neural training. Training runs locally in a background worker until Stop is pressed. Stop requests a final evaluation and checkpoint save. Keep the page open during a long run; mobile browsers may pause background tabs or suspend work when the device sleeps.

## Data revisions
The v0.18.4 curriculum adds 3,000 balanced training examples and 60 held-out validation examples to v0.18.3. The older base and supplement files remain unchanged. v0.18.4 uses its own dataset cache and checkpoint namespace, preserving previous checkpoints.

## Scope
The new language-focused examples target alphabetic ordering, word ordering, reverse-order instructions, short sentence construction, and complete-sentence responses. These are narrow synthetic examples, not evidence of broad language understanding. Evaluate with fresh benchmark questions after training.
