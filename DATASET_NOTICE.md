# Sherlock AI training data notice

## Active Neural Lab dataset: Sherlock knowledge curriculum v0.19.4

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

The v0.19.4 conversation supplement replaces the earlier low-diversity conversation set. It contains 700 unique prompts using 35 clear intent patterns across 20 topics, paired with 700 distinct, intent-aligned responses. The 100 held-out conversation validation prompts remain unchanged, and no prompt is shared between the training and validation splits.

The v0.19.0 factual knowledge supplement remains included (51 training / 15 validation examples). The original model architecture, parameter count, and frozen legacy token IDs are unchanged.

The curriculum is original synthetic content. No external dataset samples are downloaded or included.
