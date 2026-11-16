# Sherlock AI v0.12.0: conversation intelligence experiment

This isolated browser experiment tests three changes aimed at making the small word-level model answer questions more reliably. It does not replace the v0.11.0 experiment or the current stable model.

## What changes

- **Training-compatible prompts:** generation wraps a question in the same `user:` / `sherlock:` format used by the training corpus. The Playground displays only the generated answer.
- **Longer context:** the model sees the previous 32 tokens instead of 8, so the end of a question and the response marker can influence the first answer tokens.
- **Learned answer ending:** a reserved end-of-turn token is inserted at the end of every Sherlock response during training. Generation stops when the model predicts that token, rather than always running to a fixed token limit.

## Model size

- Vocabulary: 1,024 IDs, including reserved unknown, space, newline and end-of-turn IDs.
- Context: 32 tokens.
- Embedding size: 8.
- Hidden layer: 48 ReLU units.
- Trainable parameters: 70,704.
- Batch size: 32.
- AdamW learning rate: 0.001 by default.

The larger context increases the model from 61,488 to 70,704 parameters. It may run more slowly on some devices, so device speed must be measured.

## Evaluation

This model's assistant-target validation includes the new end-of-turn target and uses a different context size. Its scores are not directly comparable with v0.11.0 or the older stable model. Compare the models using the same question tests and judge both relevance and answer quality.

## Safety and isolation

This page uses separate checkpoint, archive and legacy-storage keys and a versioned worker URL. It does not overwrite the v0.11.0 or stable-model checkpoints. Training remains local to the browser.

Static checks can verify marker placement, prompt formatting, model dimensions, EOS stopping, checkpoint compatibility and generation plumbing. Real dataset quality and tablet speed still require a browser training run.
