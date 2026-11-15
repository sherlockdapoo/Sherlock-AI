# Sherlock GPT

Sherlock GPT is a small browser-based neural language experiment. The workspace has two tabs: **Chat** and **Neural Lab**. Training runs locally in a dedicated browser worker, and the best model checkpoint is saved in this browser.

## Version 0.9.1: four-source dataset expansion

Sherlock GPT now combines four bounded public datasets to give its small character-level model more varied text:

- **DailyDialog:** up to 3,500 training conversations and 500 held-out validation conversations.
- **UltraChat 200k:** up to 1,600 training conversations from `train_sft` and 250 held-out conversations from `test_sft`.
- **No Robots:** up to 3,500 training conversations from `train` and 500 held-out conversations from `test`.
- **GSM8K:** up to 1,000 maths word problems and worked answers from the `main/train` split, with 250 separate problems from `main/test` held out for evaluation.

The loader samples across each available split rather than taking only its first rows. It formats GSM8K question/answer pairs as user/Sherlock turns, while leaving each source's evaluation examples outside the training stream. Optional sources can fail independently; the interface reports which sources loaded.

The combined training text can now use up to **9 MiB**, matching the existing local engine buffer, and held-out text can use up to **128 KiB**. Downloads are cached in this browser. The new dataset revision causes Sherlock to start a fresh model for the new data mix; the previous compatible checkpoint is archived in this browser before it is replaced.

Because the validation mix has changed, scores from this revision should not be compared directly with scores from earlier dataset versions. These metrics still measure next-character prediction, not mathematical reasoning or general intelligence.

## Open Sherlock GPT

https://sherlockdapoo.github.io/Sherlock-AI/

Open **Neural Lab** and press **Start neural training**. The first run downloads the bounded training and evaluation samples. The neural network itself trains locally in a background browser worker. Press **Stop** to restore the highest-accuracy checkpoint and unlock Chat.

The C++ character-counter training screen and its separate Live View have been removed from the visible workspace. The earlier counter remains only as an archived baseline in the project code; the active Chat path uses the neural model.

## Neural network experiment

- **Architecture:** character-level multilayer perceptron (MLP), four-character context, 12-dimensional learned embeddings, one 48-neuron ReLU hidden layer, and a 128-way softmax output.
- **Size:** 10,160 trainable parameters.
- **Learning task:** predict the next character from the previous four characters.
- **Training:** cross-entropy loss and backpropagation, using mini-batches of 24 examples. Neural Lab supports Adam and AdamW (learning rate 0.003); AdamW uses decoupled weight decay of 0.0001. Six mini-batches are processed per worker slice before yielding to the browser event loop.
- **Validation/evaluation:** every 100 optimizer steps, the worker evaluates 256 positions from the combined held-out text (or all available positions if fewer than 256 exist). It reports top-1 accuracy, mean cross-entropy (negative log-likelihood), and perplexity (`exp(cross-entropy)`). Accuracy is the share of next characters predicted exactly; lower cross-entropy and lower perplexity indicate better probability estimates. These are character-prediction metrics, not measures of intelligence, factual knowledge, or conversation quality.
- **Network visualiser:** the Neural Lab draws a live sampled view of 12 embedding inputs, 16 hidden neurons, and 12 output neurons. The worker sends real activation values and sampled connection weights during progress updates. This is a readable sample of the 10,160-parameter network, not every neuron or connection.
- **Best checkpoint:** the worker selects weights with the highest held-out top-1 accuracy; validation cross-entropy breaks ties. The page stores those weights in browser local storage. The saved checkpoint stores model weights and score, not the full training history or optimiser state.
- **Stop behaviour:** stopping performs a final evaluation, restores the best weights seen in the run, and leaves Chat using that model.

## Training data and licences

- [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog), accessed through the [browser-readable mirror](https://huggingface.co/datasets/agentlans/li2017dailydialog). The dataset is labelled **CC BY-NC-SA 4.0**.
- [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). Its dataset card lists **MIT**.
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots). Its dataset card lists **CC BY-NC 4.0**.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k). Its dataset card lists **MIT**.

Only the selected training split is included in the training text. Each held-out validation/test sample is assembled separately and is not passed to the training buffer. Text is cleaned, converted to a simple user/Sherlock conversation format, and used for character-level next-character prediction. This is not token-level supervised fine-tuning.

Review each dataset's card and licence before redistributing data or model outputs. In particular, DailyDialog and No Robots have non-commercial terms. This README is a practical summary, not legal advice.

## Technology

- Front end: HTML, CSS, and JavaScript.
- Neural training: JavaScript in a dedicated Web Worker.
- Optimiser: Adam with backpropagation.
- Model storage: browser local storage for the best checkpoint weights.
- Training location: the user's browser and device.
- Dataset delivery: Hugging Face Dataset Viewer API.
- Static hosting: GitHub Pages.

Legacy C++/WebAssembly baseline files and old Python/Render experiments may remain in the repository, but the active GitHub Pages path uses the neural worker.
