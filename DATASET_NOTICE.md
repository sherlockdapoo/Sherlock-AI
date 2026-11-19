# Sherlock AI training data notice

## Active Neural Lab dataset: Balanced Learning v0.18.2

The active Neural Lab uses an original synthetic curriculum generated specifically for Sherlock AI. The active files are:

- `data/sherlock-curriculum-v0182-train.txt`: 15,000 training examples.
- `data/sherlock-curriculum-v0182-validation.txt`: 1,500 held-out validation examples.

The curriculum contains six task families, with exactly 2,500 training and 250 validation examples in each family. The v0.18.2 revision adds 3,000 training examples and 300 held-out examples to the previous v0.18.1 curriculum. Training and validation prompts are unique, and the two splits have no exact conversation overlap.

The six task families are arithmetic, percentages, unit conversions, word problems, number sequences, and letter/instruction tasks. The expanded examples use synthetic values and wording based on the existing curriculum.

No external dataset samples are downloaded or included in this active curriculum.
