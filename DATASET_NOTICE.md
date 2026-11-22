# Sherlock AI training data notice

## Active Neural Lab dataset: Sherlock knowledge curriculum v0.19.1

The active curriculum contains **24,751 training conversations** and **1,795 held-out validation conversations** across seven task families:

- Arithmetic: 3,900 training / 286 validation
- Percentages: 3,700 / 266
- Unit conversions: 3,800 / 276
- Word problems: 3,800 / 276
- Number sequences: 3,800 / 276
- Letter/instruction tasks: 5,000 / 300
- Conversation and factual knowledge: 751 / 115

## Data-quality corrections

A full prompt-level audit found 85 repeated training prompts across 25 prompt groups, plus two prompts shared between training and validation. These have been corrected without changing the conversation counts or task-family totals.

The v0.18.7 conversation supplement also replaces the v0.18.6 set, which reused identical user prompts with different assistant responses. All active training prompts are now unique, all held-out validation prompts are unique, and no prompt is shared between the two splits. The browser loader enforces these checks and rejects a library that fails them.

The v0.19.0 factual knowledge supplement remains included (51 training / 15 validation examples). The original model architecture, parameter count, and frozen legacy token IDs are unchanged.

The curriculum is original synthetic content. No external dataset samples are downloaded or included.
