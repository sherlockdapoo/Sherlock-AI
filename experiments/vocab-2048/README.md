# Sherlock AI: 2,048-token vocabulary experiment

This folder preserves the experimental 2,048-token version tested on 15 November 2026.

## Files
- `index.html`: the experiment interface snapshot.
- `neural-worker.js`: the matching 2,048-token training worker.

## Why it was archived
The live version was returned to the 1,024-token model because the experiment ran at about 4,000 examples/second and reported lower accuracy. The larger vocabulary increased output-layer work and used a separate, newly initialised checkpoint.

This is an archived experiment, not the live application. The files remain separate so the experiment can be revisited without replacing the stable model.
