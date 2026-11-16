# Sherlock AI v0.13.0 standalone curriculum experiment

This experiment loads one self-contained training library from the repository. It does not download or combine DailyDialog, UltraChat, No Robots, or GSM8K.

## Dataset files

- `data/sherlock-curriculum-v0130-train.txt`: 2,174 training conversations.
- `data/sherlock-curriculum-v0130-validation.txt`: 225 held-out validation conversations.

Both files use the same plain-text role format:

```text
user: question
sherlock: answer
```

## Coverage

- Arithmetic: addition, subtraction, multiplication, and exact division.
- Elementary factual questions.
- Instruction following and constrained short responses.
- One-step word problems.
- Short explanations of school concepts.
- Multi-turn recall of a fact introduced earlier in the conversation.

The examples are locally authored and generated from deterministic templates. They are not a substitute for a broad, professionally curated knowledge corpus. Some facts and patterns are repeated with different question wording to give this small model more examples of the desired answer format.

## Validation and safety checks

The browser loader checks that:

- The training set contains at least 1,000 examples and validation contains at least 200.
- Every block contains both a user and Sherlock turn.
- No complete held-out validation conversation appears in training.
- Both files fit within the browser engine's existing memory limits.

The held-out set is not appended to training. It is used for validation only.

## Model isolation

- Architecture remains context 32, embedding 8, hidden 48, vocabulary 1,024 (70,704 parameters).
- A new dataset revision and checkpoint keys prevent this experiment from overwriting v0.12.1.
- The stable v0.9.9 model is unchanged.
- The previous external-source loader remains in the source code as a fallback reference, but the v0.13.0 experiment explicitly selects the standalone library and returns before any external dataset requests.

## Known limitations

This is a compact model with limited capacity. A larger dataset does not guarantee better answers. The first real test must compare training speed, validation loss, unknown-token rate, and the same held-out five-question benchmark used for previous experiments. Do not promote this model or remove older datasets permanently unless repeated tests show a reliable improvement.
