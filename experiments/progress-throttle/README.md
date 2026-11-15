# Progress update throttle experiment

Sherlock AI v0.9.5 tests lower browser UI overhead during neural training.

Changes in the live neural worker:
- Training progress messages are sent at most 4 times per second (250 ms interval).
- The network visualiser snapshot is generated at most once per second.
- Training math, model dimensions, vocabulary, batch size, learning rate, optimiser, validation frequency, accuracy calculations, and checkpoint format are unchanged.

Why:
The prior worker generated a fresh network visualiser snapshot after every eight training batches. The page redrew hundreds of canvas connections for every progress message. On a tablet, this extra main-thread rendering may compete with the training worker for device resources.

Rollback:
The exact pre-experiment worker is preserved at `experiments/speed-optimisation-baseline/neural-worker.js`. Restore it and set the worker URL in `index.html` to `neural-worker.js?v=0.7.2`, then bump the visible app version as appropriate.

This is a performance experiment, not a measured speed improvement. Compare several minutes of training against the previous baseline of about 8k examples/sec and 65% accuracy.
