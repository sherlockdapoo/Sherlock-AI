# Sherlock AI training data notice

## Active Neural Lab dataset: Sherlock knowledge curriculum v0.19.9

The active curriculum contains **25,031 training conversations** and **1,795 held-out validation conversations** across seven task families:

- Arithmetic: 3,900 training / 286 validation
- Percentages: 3,700 / 266
- Unit conversions: 3,800 / 276
- Word problems: 3,800 / 276
- Number sequences: 3,800 / 276
- Letter/instruction tasks: 5,120 / 300
- Conversation and factual knowledge: 911 / 115

## Data-quality corrections

A full prompt-level audit found 85 repeated training prompts across 25 prompt groups, plus two prompts shared between training and validation. These have been corrected without changing the conversation counts or task-family totals.

The v0.19.5 conversation supplement contains 700 unique prompts using 35 clear intent patterns across 20 topics, paired with 700 distinct, intent-aligned responses. The 100-example held-out conversation validation set was revised to contain 100 unique prompts and 100 unique answers, with an average answer length of 24.4 words (training average: 24.18). Its prompts do not overlap the conversation training prompts. The previous validation set is preserved at data/archive/sherlock-curriculum-v0191-conversation-validation-original.txt for historical comparison.

The v0.19.0 factual knowledge supplement remains included (51 training / 15 validation examples). The original model architecture, parameter count, and frozen legacy token IDs are unchanged.

The curriculum is original synthetic content. No external dataset samples are downloaded or included.



## Conversation generalisation supplement (v0.19.8)

A new 100-example conversation training supplement broadens coverage across five response intents: getting started, practising a skill, checking understanding, planning a small investigation, and structuring a clear explanation. It uses 100 unique prompts and 100 unique answers across the existing 20 conversation topics. The supplement was checked against all active training and held-out examples: there are no exact prompt collisions, no prompt-template collisions under the loader's normalisation rule, and no exact answer collisions. The held-out validation set is unchanged.

## Validation-set revision (v0.19.7)

The v0.19.7 revision replaces four malformed letter/instruction validation examples with clear, well-formed prompts and aligned answers. Dataset counts and task-family totals are unchanged.

## Validation-set revision (v0.19.5)

The conversation validation examples now better match the training response-length and answer-diversity profile. This corrects a measurable distribution mismatch; it does not guarantee lower validation loss or prove that all overfitting has been resolved. Keep the archived original set separate when comparing historical evaluation results.


## Targeted generalisation update (v0.19.9)

This update adds **60 factual-conversation paraphrases** covering the 15 held-out knowledge concepts and **120 letter/instruction examples** across alphabetic letter ordering, word ordering, reversal and sentence construction. The new prompts and answers were checked against the active training and validation sources for exact collisions; prompts were also checked against the held-out prompt-template normalisation used by the website loader. All 180 new examples passed those checks.

The training split now contains 111 factual-knowledge examples within the conversation family and 5,120 letter/instruction examples. The held-out validation split remains unchanged at 1,795 examples. The model architecture, tokenizer IDs, optimizer, learning rate, weight decay, batch size, dropout and training algorithm are unchanged. A new dataset revision forces the browser to load the expanded corpus; the previous saved checkpoint is retained but is not treated as trained on this new dataset revision.
