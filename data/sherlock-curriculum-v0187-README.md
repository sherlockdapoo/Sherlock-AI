# Sherlock conversation curriculum v0.18.7

## Purpose
This revision replaces the v0.18.6 conversation supplement after an audit found that many identical user prompts were paired with different assistant responses.

## Dataset composition
- Training examples: 700
- Held-out validation examples: 100
- Every user prompt is unique within the training split.
- Every user prompt is unique within the validation split.
- No user prompt appears in both splits.
- The total example counts are unchanged, preserving the seven-family curriculum size.

## Correction
The earlier set repeated short prompts such as greetings and generic requests many times. This creates conflicting target responses for the same prompt and encourages memorisation of a narrow set of patterns. The replacement keeps the original assistant-response examples but adds distinct, in-vocabulary context to each user prompt.

## Model compatibility
The model architecture, parameter count, frozen legacy token IDs, and tokenizer capacity are unchanged. The new prompt context uses existing vocabulary entries; no new vocabulary entries are required. Dataset cache revision is advanced to v0.19.1 so browsers load the corrected files.
