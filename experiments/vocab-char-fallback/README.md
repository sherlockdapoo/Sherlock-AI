# Sherlock AI: 1,024-token character-fallback experiment

This is an isolated experiment. The normal Sherlock AI page and its stable checkpoint are unchanged.

## Hypothesis

Keep common whole words as one token. Reserve 37 entries in the same 1,024-token vocabulary for the lowercase letters, digits, and apostrophe. When a word is not in the learned whole-word vocabulary, encode it as a sequence of those character tokens.

## What stays fixed

- Vocabulary array: 1,024 entries
- Trainable parameters: 61,488
- Context length: 8
- Embedding size: 8
- Hidden layer: 48
- Optimiser and training loop: unchanged

## What changes

- Tokenisation only: unknown ASCII words can fall back to letters instead of becoming one `<unk>` token.
- Training and validation use the same fallback rule.
- The page displays the share of validation words that needed letter fallback. This is not the same metric as the old unknown-token rate.
- The checkpoint key and worker cache version are unique to this experiment.

## Trade-off to measure

Character fallback creates more sequence positions for rare words. That may improve coverage but can slow training or change accuracy. Compare training speed, validation accuracy, training loss, validation loss, and the fallback rate against the stable page. Do not replace the stable model unless the measured results are better.

## Test page

After this experiment is merged into the GitHub Pages site, open:

`https://sherlockdapoo.github.io/Sherlock-AI/experiments/vocab-char-fallback/`
