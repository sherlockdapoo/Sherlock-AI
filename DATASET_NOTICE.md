# Sherlock AI training data notice

## Active Neural Lab dataset: Sherlock knowledge curriculum v0.19.5

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

The v0.19.5 conversation supplement contains 700 unique prompts using 35 clear intent patterns across 20 topics, paired with 700 distinct, intent-aligned responses. The 100-example held-out conversation validation set was revised to contain 100 unique prompts and 100 unique answers, with an average answer length of 24.4 words (training average: 24.18). Its prompts do not overlap the conversation training prompts. The previous validation set is preserved at data/archive/sherlock-curriculum-v0191-conversation-validation-original.txt for historical comparison.

The v0.19.0 factual knowledge supplement remains included (51 training / 15 validation examples). The original model architecture, parameter count, and frozen legacy token IDs are unchanged.

The curriculum is original synthetic content. No external dataset samples are downloaded or included.


## Validation-set revision (v0.19.5)

The conversation validation examples now better match the training response-length and answer-diversity profile. This corrects a measurable distribution mismatch; it does not guarantee lower validation loss or prove that all overfitting has been resolved. Keep the archived original set separate when comparing historical evaluation results.
