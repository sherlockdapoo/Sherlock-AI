# Sherlock GPT

Sherlock GPT is a small browser-based neural language experiment. The workspace has two tabs: **Chat** and **Neural Lab**. Training runs locally in a dedicated browser worker, and the best model checkpoint is saved in this browser.

## Version 0.5.0: Multi-dataset training

The training-data loader now supports two conversation sources:

- **DailyDialog** for everyday multi-turn conversations.
- **UltraChat 200k** for a broader sample of instruction-style conversations.

The UltraChat download is deliberately capped at 1,200 training conversations and 250 held-out evaluation conversations. Sherlock does not download the full 1.6 GB dataset. When UltraChat is available through the Hugging Face Dataset Viewer API, the combined training stream stays within the existing 2 MB local engine buffer: up to 1.2 MB from DailyDialog and about 0.8 MB from UltraChat. If UltraChat's dataset endpoint is unavailable, the loader falls back to the original DailyDialog-only setup and tells the user.

The UltraChat training sample comes from `train_sft`. Its separate `test_sft` sample is used only for held-out evaluation, never for training. DailyDialog's `train` and `validation` splits remain separate. The combined validation stream is capped at 120 KB.

## Open Sherlock GPT

https://sherlockdapoo.github.io/Sherlock-AI/

Open **Neural Lab** and press **Start neural training**. The first run downloads the bounded training and evaluation samples. The neural network itself trains locally in a background browser worker. Press **Stop** to restore the highest-accuracy checkpoint and unlock Chat.

The C++ character-counter training screen and its separate Live View have been removed from the visible workspace. The earlier counter remains only as an archived baseline in the project code; the active Chat path uses the neural model.

## Neural network experiment

- **Architecture:** character-level multilayer perceptron (MLP), four-character context, 12-dimensional learned embeddings, one 48-neuron ReLU hidden layer, and a 128-way softmax output.
- **Size:** 10,160 trainable parameters.
- **Learning task:** predict the next character from the previous four characters.
- **Training:** cross-entropy loss, backpropagation, and the Adam optimiser (learning rate 0.003), using mini-batches of 24 examples. Six mini-batches are processed per worker slice before yielding to the browser event loop.
- **Validation/evaluation:** every 100 optimizer steps, the worker evaluates 256 positions from the combined held-out text (or all available positions if fewer than 256 exist). It reports top-1 accuracy, mean cross-entropy (negative log-likelihood), and perplexity (`exp(cross-entropy)`). Accuracy is the share of next characters predicted exactly; lower cross-entropy and lower perplexity indicate better probability estimates. These are character-prediction metrics, not measures of intelligence, factual knowledge, or conversation quality.
- **Network visualiser:** the Neural Lab draws a live sampled view of 12 embedding inputs, 16 hidden neurons, and 12 output neurons. The worker sends real activation values and sampled connection weights during progress updates. This is a readable sample of the 10,160-parameter network, not every neuron or connection.
- **Best checkpoint:** the worker selects weights with the highest held-out top-1 accuracy; validation cross-entropy breaks ties. The page stores those weights in browser local storage. The saved checkpoint stores model weights and score, not the full training history or optimiser state.
- **Stop behaviour:** stopping performs a final evaluation, restores the best weights seen in the run, and leaves Chat using that model.

## Training data and licences

- [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog), delivered through the [agentlans browser-readable mirror](https://huggingface.co/datasets/agentlans/li2017dailydialog). The dataset is labelled **CC BY-NC-SA 4.0**.
- [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). Its dataset card lists an **MIT** licence. Sherlock samples from `train_sft` and uses a small `test_sft` sample only for held-out evaluation.
- Only user and assistant turns are included; system messages are ignored. Text is cleaned and converted into a character stream.
- The UltraChat Dataset Viewer currently may not serve every split. If the required split or rows cannot be fetched, Sherlock reports the issue and continues with DailyDialog only.

DailyDialog's non-commercial and ShareAlike conditions still matter when reusing or redistributing material or adaptations. The app's dataset notice links to the source pages and full licence texts. This README is a practical summary, not legal advice.

## Performance and device use

The neural worker keeps training off the main page thread so the interface can remain interactive. It currently uses one worker and one CPU thread; it does not use all four CPU cores. Actual speed depends on the browser and device, and should be compared using the live examples-per-second metric. True multi-core execution is a separate upgrade and depends on browser support and hosting requirements for WebAssembly threads and shared memory.

The model is intentionally small and is not a transformer or a full ChatGPT-scale model. More data can improve the variety of character patterns it sees, but it does not automatically give the model reasoning ability or reliable factual knowledge.

## Technology

- Front end: HTML, CSS, and JavaScript.
- Neural training: JavaScript in a dedicated Web Worker.
- Optimiser: Adam with backpropagation.
- Model storage: browser local storage for the best checkpoint weights.
- Training location: the user's browser and device.
- Dataset delivery: Hugging Face Dataset Viewer API.
- Static hosting: GitHub Pages.

Legacy C++/WebAssembly baseline files and old Python/Render experiments may remain in the repository, but the active GitHub Pages path uses the neural worker.
