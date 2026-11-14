# Sherlock GPT

Sherlock GPT is a small browser-based neural language experiment. The workspace now has two tabs: **Chat** and **Neural Lab**. The Neural Lab contains the training controls, its own live observatory, and a live neural network visualiser.

## Interface polish (v0.4.6)\n\nThe workspace uses a refined dark visual system, improved card and tab contrast, clearer focus states, and responsive spacing. The network diagram uses a high-density canvas that redraws on screen resize; the training algorithm, model dimensions, and saved checkpoint format are unchanged.\n\n## Open Sherlock GPT

https://sherlockdapoo.github.io/Sherlock-AI/

Open **Neural Lab** and press **Start neural training**. The first run downloads the DailyDialog training and validation examples. The neural network itself trains locally in a background browser worker. Press **Stop** to restore the highest-accuracy validation checkpoint and unlock Chat.

The C++ character-counter training screen and its separate Live View have been removed from the visible workspace. The earlier counter remains only as an archived baseline in the project code; the active Chat path uses the neural model.

## Neural network experiment (v0.4.6)

- **Architecture:** character-level multilayer perceptron (MLP), four-character context, 12-dimensional learned embeddings, one 48-neuron ReLU hidden layer, and a 128-way softmax output.
- **Size:** 10,160 trainable parameters.
- **Learning task:** predict the next character from the previous four characters.
- **Training:** cross-entropy loss, backpropagation, and the Adam optimiser (learning rate 0.003), using mini-batches of 24 examples. Six mini-batches are processed per worker slice before yielding to the browser event loop.
- **Data:** DailyDialog training and validation splits are used separately. The network does not train on its validation examples.
- **Validation:** every 100 optimizer steps, the worker evaluates 256 positions from the held-out validation text (or all available positions if fewer than 256 exist), using the same fixed-stride layout as v0.3.0 for easier comparison. It reports top-1 accuracy, mean cross-entropy (negative log-likelihood), and perplexity (`exp(cross-entropy)`). Accuracy is the share of next characters predicted exactly; lower cross-entropy and lower perplexity indicate better probability estimates. These are character-prediction metrics, not measures of intelligence, factual knowledge, or conversation quality.
- **Network visualiser (v0.4.6):** the Neural Lab draws a live sampled view of 12 embedding inputs, 16 hidden neurons, and 12 output neurons. The worker sends their real activation values and 384 sampled connection weights during progress updates. This is a readable sample of the 10,160-parameter network, not every neuron or connection. The canvas now renders at the device pixel density (up to 3×) and adapts its drawing scale to the available screen width for crisp mobile and desktop display.
- **Live view:** the Neural Lab shows the training pipeline, current loss curve, validation-accuracy curve, model architecture, and the latest training metrics from the worker.
- **Best checkpoint:** the worker selects the weights with the highest held-out top-1 accuracy; validation cross-entropy breaks ties. It sends those weights and their validation metrics to the page, which stores them in browser local storage. New saves use checkpoint format v2 and include validation loss, perplexity, and evaluation counts. Existing v1 checkpoints remain loadable; their missing validation metrics are filled in after the next evaluation. When the site is opened again in the same browser, the worker restores the saved weights so Chat can use them without training again. Continuing training starts from those weights.
- **Stop behaviour:** stopping performs a final validation check, restores the best weights seen in the run, and leaves Chat using that model.

The persistent checkpoint stores the model weights and score, not the full training history or optimiser state. Training counters and charts start a new page session after refresh. Clearing this site's browser data, using private browsing, or changing browsers/devices can remove or hide the saved checkpoint.

## Training data and licence

- Dataset: [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)
- Browser-readable mirror: [agentlans/li2017dailydialog](https://huggingface.co/datasets/agentlans/li2017dailydialog)
- Only the train and validation splits are used. The test split is not downloaded or used.
- Only human and assistant turns are included. Dataset system prompts are ignored.
- Attribution and licensing details are in [DATASET_NOTICE.md](DATASET_NOTICE.md).

**Licence restriction:** DailyDialog is licensed under CC BY-NC-SA 4.0 and is for non-commercial use. If adapted dataset material is redistributed, the applicable share-alike terms also apply. Do not use this dataset or the trained-data setup for a commercial product without separate permission.

## Performance and device use

The neural worker keeps training off the main page thread so the interface can remain interactive. It currently uses one worker and one CPU thread; it does not use all four CPU cores. The worker processes six mini-batches per scheduled slice and reports progress less often to reduce scheduling overhead. Validation now checks half as many examples per evaluation as v0.4.0, reducing time spent measuring rather than training. Actual speed depends on the browser and device, and should be compared using the live examples-per-second metric. True multi-core execution is a separate upgrade and depends on browser support and hosting requirements for WebAssembly threads and shared memory.

The model is intentionally small and is not a transformer or a full ChatGPT-scale model. The next development stages can increase model capacity, improve tokenisation, and eventually explore attention once the current learning pipeline has been benchmarked.

## Technology

- Front end: HTML, CSS, and JavaScript.
- Neural training: JavaScript in a dedicated Web Worker.
- Optimiser: Adam with backpropagation.
- Model storage: browser local storage for the best checkpoint weights.
- Training location: the user's browser and device.
- Dataset delivery: Hugging Face Dataset Viewer API.

Legacy C++/WebAssembly baseline files and old Python/Render experiments may remain in the repository, but the active GitHub Pages path uses the neural worker.
