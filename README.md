# Sherlock GPT

Sherlock GPT is a small, browser-based character-language experiment. The C++ model is compiled to WebAssembly and runs locally in the browser.

## Try it

Open the GitHub Pages site:

https://sherlockdapoo.github.io/Sherlock-AI/

On the Training tab, press **Start training**. The page downloads DailyDialog training and validation examples in small pages, prepares the text, and copies it into the WebAssembly engine. The model training itself runs locally in the browser. Stop training when you want to unlock Chat.

The first data load needs an internet connection. The prepared model data stays in page memory and is not saved as a permanent trained model. Refreshing the page resets the training progress and may require downloading the examples again.

## Training data

- Dataset: [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)
- Browser-readable mirror: [agentlans/li2017dailydialog](https://huggingface.co/datasets/agentlans/li2017dailydialog)
- Train and validation splits are used separately.
- The test split is not downloaded or used.
- Only human and assistant turns are included. Dataset system prompts are ignored.
- Attribution and licensing details are in [DATASET_NOTICE.md](DATASET_NOTICE.md).

**Licence restriction:** DailyDialog is licensed under CC BY-NC-SA 4.0. It is for non-commercial use. If you redistribute adapted dataset material, the applicable share-alike terms also matter. Do not use the dataset or this trained-data setup for a commercial product without separate permission.

## What the score means

The Accuracy bar measures held-out next-character prediction accuracy on a separate validation split. It is not a measure of intelligence, factual knowledge, or conversational quality.

- **Current accuracy** is the latest validation score, checked after each complete pass through the training text.
- **Best accuracy this run** is the highest completed-pass score reached since the current run started.
- **Current training pass** shows how far the model has progressed through the training text.
- Accuracy is displayed to three decimal places. The validation calculation samples up to 10,000 positions to keep checking responsive on iPad.

The current score can still rise or fall between completed passes. This small count-based model changes its most-common next-character predictions as it sees more examples, so more training does not guarantee a higher validation score. The best score is shown separately, and when training is stopped the engine restores the model checkpoint that achieved the best held-out score in that run.

The current model is a small character-level 4-gram counter with backoff. It first uses the previous three characters when it has enough examples, then falls back to the previous two characters. Its counters are 32-bit, so an individual next-character count no longer stops at 65,535. It still learns character patterns rather than meaning, and it is not a modern language model.

## Performance tuning

The browser adjusts the next training batch automatically. It starts at 65,536 characters, samples recent C++ call durations, and tunes the next batch every three calls toward a roughly 30 ms target. The configured range is 16,384 to 196,608 characters. This aims to keep the engine busy without making individual calls unnecessarily long.

The Live View reports measured throughput, average batch time, peak sampled throughput, cumulative C++ call time, and the next batch size. Throughput is measured on the device and can vary with browser scheduling, temperature, and other activity.

The current WebAssembly engine is single-threaded. Adaptive batches do not use all four CPU cores; true multi-core training is a separate future upgrade that depends on browser and hosting support for WebAssembly threads and shared memory.

## Neural network experiment (v0.2.0)

The Neural Lab adds Sherlock's first actual neural network while keeping the original C++ character-counter as a separate baseline.

- **Architecture:** a character-level multilayer perceptron (MLP) with a four-character context, 12-dimensional learned character embeddings, one 48-neuron ReLU hidden layer, and a 128-way softmax output.
- **Size:** 10,160 trainable parameters.
- **Learning task:** predict the next character from the previous four characters.
- **Training:** cross-entropy loss, backpropagation, and the Adam optimiser (learning rate 0.003), using mini-batches of 12 examples.
- **Data:** uses the same DailyDialog train and validation splits as the baseline. The neural worker receives copies of the prepared bytes, so both experiments keep separate model state.
- **Validation:** reports top-1 next-character accuracy on up to 512 evenly spaced positions from the held-out validation text. This is a smaller sample than the original counter's validation check, so the two displayed percentages are useful indicators but are not a perfectly controlled head-to-head benchmark.
- **Best checkpoint:** when training stops, the worker evaluates the network and restores the best validation-scoring weights seen in that page session.
- **Chat:** after the neural model has trained and is stopped, the Chat tab switches to neural-network generation. Before that, the original C++ baseline remains available.

The network trains in `neural-worker.js`, a dedicated browser Web Worker. This keeps the main page more responsive, but it is still one worker using one CPU thread. It does not yet use all four CPU cores, and moving the neural calculations into C++ WebAssembly or adding safe multi-core execution is a later optimisation.

Like the baseline, the neural model exists only in page memory. Refreshing the page resets the neural weights and training counters. The first dataset load needs an internet connection; the neural training calculations themselves run locally in the browser.

This is a deliberately small first neural model, not a transformer or a full ChatGPT-scale language model. The goal is to establish a real backpropagation training pipeline that can be measured and improved before increasing model size.

## Technology

- Front end: HTML, CSS, and JavaScript.
- Learning engine: C++ compiled to WebAssembly.
- Model: sparse three-character context table with two-character backoff and 32-bit counters.
- Best checkpoint: saves and restores the best validation-scoring model within the current page session.
- Training location: user's browser and device.
- Dataset delivery: Hugging Face Dataset Viewer API.

The old Python/Render backend files may remain in the repository for historical experiments, but the GitHub Pages training path uses the local WebAssembly engine.
