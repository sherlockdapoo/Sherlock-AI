# Sherlock Balanced Learning curriculum v0.18.4

## Purpose
This revision extends v0.18.3 with another balanced 3,000-example training supplement and 60 independent held-out examples. It specifically broadens instruction following, word ordering, short sentence construction, and complete-sentence answers while retaining all five existing maths/sequence families. The network architecture and deterministic maths path are unchanged.

## Dataset composition
- Existing v0.18.2 base: 15,000 training / 1,500 validation examples.
- Existing v0.18.3 supplement: 3,000 training / 60 validation examples.
- New v0.18.4 supplement: 3,000 training / 60 validation examples.
- Combined total: **21,000 training / 1,620 held-out validation examples**.
- Six task families: **3,500 training / 270 validation examples each**.
- Each new supplement contributes 500 training and 10 validation examples per family.

## Instruction and language practice
The new instruction family includes five equally sized subtypes:
1. Alphabetical ordering of five distinct letters.
2. Alphabetical ordering of five words.
3. Reversing a word list.
4. Writing one clear sentence using a specified word.
5. Writing a complete sentence in response to a short factual question or a sentence-construction instruction.

The training and validation prompts use separate deterministic input ranges. No exact prompt/answer conversation is duplicated within the supplement or across its training and validation splits.

## Other families
The supplement preserves balanced practice for:
- Arithmetic (addition, subtraction, multiplication, and exact division).
- Percentages (percentage of an amount).
- Unit conversions (metric and time units).
- Word problems (addition, multiplication, subtraction, and equal sharing).
- Number sequences (increasing/decreasing arithmetic, doubling, square numbers, and alternating increments).

## Validation checks
- Exactly 3,000 training examples and 60 held-out examples.
- Exactly 500 training / 10 validation examples per family.
- Exactly 3,060 unique conversations across both new splits.
- No malformed or empty turns.
- All prompts match one of the six curriculum-family classifiers.
- Validation remains separate from training.
- New training supplement is approximately 234,004 characters; validation supplement is approximately 4,725 characters.

## Model and training
- Architecture remains a word-level MLP with 32-token context, 8-dimensional embeddings, one 48-neuron ReLU hidden layer, and a 1,024-token output vocabulary.
- Trainable parameters remain **70,704**.
- AdamW defaults remain learning rate 0.001, weight decay 0.0001, batch size 32.
- Training continues until the user presses Stop. Stop performs final evaluation and saves/restores the best compatible checkpoint.
- v0.18.4 uses separate dataset-cache and checkpoint keys; previous v0.18.3 checkpoints are preserved.

## Limitations
This remains a small synthetic curriculum, not broad language pretraining. The short sentence examples teach narrow patterns and should not be interpreted as general language understanding. Re-run the answer-quality benchmark and unseen instruction tests after training. Next-token accuracy is not a direct measure of intelligence.
