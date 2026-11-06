# Probe Bench

Probe Bench runs open-weight language models on the GPU cluster of Georgia Tech's School of Industrial and Systems Engineering (ISyE), and opens them in your browser. You can:

- download models from Hugging Face, or load your own checkpoints from the cluster
- generate text, with or without the model's chat format, and see the exact tokens the model receives
- see every layer's activations for a prompt: state sizes, any single channel, the full vector for a token, and what the model would predict at each layer (the "logit lens")
- download the results as CSV, JSON or images, or save the full activations to a file to use in your own code
- read the model card and browse the weights layer by layer, with statistics, histograms, singular values and comparisons across layers
- see what's using GPU memory, including the KV cache from your last run
- use a terminal on the cluster, browse and read your cluster files, and see every command Probe Bench runs, all from the app

Probe Bench is a Mac app. It signs you in to the cluster, reserves a GPU through HTCondor and connects your browser to it. When you're done, it gives the GPU back.

## Before you start: an ISyE UNIX account

The cluster needs its own account, called an ISyE UNIX account. It uses your regular Georgia Tech username and password, but you request it separately:

1. Fill in the [ISyE UNIX account form](https://webforms.isye.gatech.edu/other-forms/isye-unix-account-request). Off campus, you may need the Georgia Tech VPN to open it.
2. Wait a day or two for the email saying it's ready.
3. Email Kelley Ross at ISyE Computing (kelley.ross@isye.gatech.edu) with your username, and ask for GPU cluster access.

You'll also need Duo, the same as for any Georgia Tech login.

## Install

Open **Terminal**: press Command-Space, type Terminal, press Return. Paste this line and press Return:

```
curl -fsSL https://raw.githubusercontent.com/mathrulestheworld/probebench/main/install.sh | bash
```

That's the only time you need Terminal. The installer:

- checks for Apple's Command Line Tools, which include Python. If they're missing, macOS offers to install them; when that finishes, paste the line again.
- puts **Probe Bench** in your Applications folder and opens it

macOS doesn't show a security warning, because the app is built on your Mac rather than downloaded. To update Probe Bench later, paste the same line again.

## First run

Probe Bench opens a page in your browser that walks you through setup:

1. **Your username.** Your Georgia Tech username, the part of your GT email before @gatech.edu.
2. **Sign in to the cluster.** Enter your Georgia Tech password and approve Duo in the window that appears. You stay signed in for 8 hours.
3. **Your SSH key.** Probe Bench creates a key on your Mac, if you don't have one, and adds it to your ISyE account. This lets it make its connections without asking for your password each time.
4. **Copy Probe Bench to the cluster.** This takes a few seconds.
5. **Prepare the software.** The cluster builds a container with PyTorch and Hugging Face Transformers, in a job that doesn't use a GPU. This takes 10 to 30 minutes, and a progress bar shows how far it has got.

Then it offers to download Gemma 4 12B (about 22 GB) as a first model. Models are stored on the cluster, not on your Mac.

## Everyday use

Open **Probe Bench** from Applications. Pick a GPU size and click **Start session**:

| Size | GPU | Good for |
|---|---|---|
| Standard | 30 GB or more | Models up to about 13B parameters, such as Gemma 4 12B |
| Large | 40 GB or more (A6000 or A100) | Up to about 20B parameters, or long prompts |
| A100 | 80 GB A100 | Up to about 35B parameters, such as Qwen 3.8 27B. There are only three. |

Each size shows how many matching GPUs are free right now. While you wait, Probe Bench shows which step it's on and why you're waiting, for example "All 17 GPUs that fit this size are in use, 2 other GPU jobs are also waiting." When the session is ready, Probe Bench opens in a new browser tab.

When you're done, click **End session**. A session that goes unused for 30 minutes ends by itself; you can choose a longer time. Quitting Probe Bench from the Dock also ends your sessions.

### In a session

- **Model panel.** Pick a downloaded model and click Load model. A progress bar shows how much of it is on the GPU. To load a fine-tuned checkpoint, use "Load from a folder on the cluster" and give the folder's full path.
- **Generate.** Turn on Chat format for instruction-tuned models (names ending in `-it` or `Instruct`). The system prompt and Allow thinking appear only with Chat format on, since only the chat template uses them. Open "Exact input to the model" to see the full text the model receives, with the template's special tokens highlighted. Temperature 0 always picks the most likely next token.
- **Activations.** Analyze a prompt, then switch the map between the size of each token's internal state, a single channel of the residual stream (start from the list of largest channels), and the top prediction at each layer. Below the map, the full activations are shown as a heatmap: either one token's vector at every layer, or every token's vector at one layer, with a slider to step through the layers. Click a column to map that channel. Choose the colors, a linear or log scale, and one color range for the whole map or one per layer. Expand opens the results full screen, and Download saves the map, the numbers and the predictions. Save activations writes a file to `~/probebench/exports/` on the cluster:

  ```python
  import torch
  d = torch.load("/home/you/probebench/exports/<file>.pt")
  acts = d["residual_stream"]   # [layers + 1, tokens, width]; row 0 is the embeddings
  tokens = d["tokens"]
  ```

- **Weights.** The model card at the top lists the architecture (layers, heads, attention pattern, context length), where the parameters are, the README and `config.json`. Browse the parameters one layer at a time, or filter the full list, for example `layers.10.mlp`. Pick a matrix for its statistics, a histogram of its values and its singular values, then compare it across all layers.
- **GPU memory.** Click the memory bar at the top for a breakdown: model weights, working memory, memory PyTorch holds for reuse, and CUDA's own share. After you generate, it shows how big the KV cache got, layer by layer, and roughly how many more tokens would fit.

### Terminal, files and commands

Click **Terminal** at the top of the app. It opens a panel with three tabs:

- **Cluster terminal:** a shell on compute01, the same as `ssh probebench-submit`. Buttons run common checks for you: your jobs, why a job is waiting, the newest job or session log, and the GPU machines. Copy output puts everything on the clipboard.
- **Files:** browse your cluster home folder and read files without typing commands. Logs open at the end, and Follow reloads them every few seconds. Shortcuts go to job logs, sessions, saved activations and models. It's read-only; use the terminal to move or delete files.
- **Commands Probe Bench ran:** every command the app has sent to the cluster, with what it printed. Routine status checks are hidden unless you ask for them. Copy the last 20 to send to someone helping you.

In a GPU session, the **Console** button on the model page shows each request the page sends (Load model, Generate, Analyze, and so on) with the Python the session runs for it, ready to copy into your own scripts. The same lines go to the session's log on the cluster; the terminal's "Follow session log" button shows them live.

If a job keeps getting stopped the moment it starts, which looks like it's stuck waiting, the app now says so and points you to the terminal.

### Gated models

Some models, such as Llama, need you to accept their license on Hugging Face first. Then create a token at huggingface.co/settings/tokens (Read access is enough) and save it under **Settings and help** in Probe Bench.

## If something goes wrong

- **The password or Duo didn't go through.** Click Sign in and try again. If castle, the gateway, is down, switch to keep under Settings and help.
- **"compute01 didn't accept your login."** Your ISyE account may not have cluster access yet. Email Kelley Ross.
- **A long wait for a GPU.** The page says how many GPUs fit and how many other jobs are waiting. Smaller sizes have more GPUs.
- **The session stopped.** The page says why, usually because it was idle. Start a new one.

For cluster problems, email helpdesk@isye.gatech.edu and include the activity log from Settings and help. Never send your password.

## For Terminal users

The installer also adds a `probebench` command: `probebench` starts a session from Terminal, and `probebench help` lists the other commands. It uses the same account, container and models as the app.

## How it works

The app runs a small local server on your Mac (`app/pbapp.py`, standard-library Python). It reaches the cluster over SSH through the castle gateway and compute01, the cluster's submit machine, and runs `cluster/labctl.sh` there. Each session is an HTCondor job that requests one GPU and runs `cluster/server/server.py` inside an Apptainer container. Your browser reaches it through an encrypted SSH tunnel, and each session has its own access key, so other people on the cluster can't use your session.

On your Mac, files live in `~/Library/Application Support/ProbeBench`. On the cluster, they live in `~/probebench`, and models in `~/hf_cache`.

To uninstall: quit Probe Bench, then delete it from Applications and delete `~/Library/Application Support/ProbeBench`. The SSH settings are the block marked `probebench` in `~/.ssh/config`.
