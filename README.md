# Probe Bench

Probe Bench runs open-weight language models on the GPU cluster of Georgia Tech's School of Industrial and Systems Engineering (ISyE), and opens them in your browser. You can:

- download models from Hugging Face, or load your own checkpoints from the cluster
- generate text, with or without the model's chat format
- see every layer's activations for a prompt, and what the model would predict at each layer (the "logit lens")
- save those activations to a file to use in your own code
- browse the weights, with statistics, value histograms and singular value spectra

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
- **Generate.** Turn on Chat format for instruction-tuned models (names ending in `-it` or `Instruct`). Temperature 0 always picks the most likely next token.
- **Activations.** Analyze a prompt to see the size of every token's internal state at each layer. Click a token to see what the model would predict next at each layer. Save activations writes a file to `~/probebench/exports/` on the cluster:

  ```python
  import torch
  d = torch.load("/home/you/probebench/exports/<file>.pt")
  acts = d["residual_stream"]   # [layers + 1, tokens, width]; row 0 is the embeddings
  tokens = d["tokens"]
  ```

- **Weights.** Filter the parameter list, for example `layers.10.mlp`, and pick a matrix to see its statistics, a histogram of its values and its singular values.

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
