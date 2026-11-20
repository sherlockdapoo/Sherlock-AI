# Sherlock conversation curriculum v0.18.6

This supplement extends the six task families in the v0.18.5 balanced curriculum with a seventh family for ordinary conversation.

- Training examples: 700
- Held-out validation examples: 100
- Training prompts include greetings, requests for help, learning, curiosity, uncertainty, thanks, capability questions, project ideas, difficult-day support, explanation requests, and goodbyes.
- Validation prompts are separate from training prompts and are not copied into the training split.

The worker samples all seven task families equally. The conversation answers are training targets for the neural language model; there is no deterministic greeting-answer lookup in the generation path.

This is a small supervised conversation supplement, not a broad knowledge dataset. It can teach response patterns represented here, but it does not guarantee reliable open-ended conversation. Evaluate using held-out prompts after training.
