---
title: ComfyUI LoRA Sweep
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# ComfyUI LoRA Sweep

**Automated LoRA Checkpoint Testing & Visual Comparison for ComfyUI**

Stop manually switching LoRA checkpoints every time you want to compare training epochs.

**LoRA Sweep** is a lightweight, Python-based web application that automates testing multiple LoRA checkpoints through ComfyUI. Select your checkpoints, provide a prompt, and let the application queue generations automatically.

Once the generations finish, compare the resulting videos or images side by side in a convenient browser interface.

Originally designed for **MiniMax H3 character LoRA training**, LoRA Sweep is particularly useful for evaluating facial likeness, character identity, visual consistency, and changes across training epochs.

## Features

- **Automated checkpoint testing** — Generate samples from multiple LoRA epochs without manually switching models.
- **ComfyUI API integration** — Uses your existing ComfyUI workflow and generation settings.
- **Side-by-side comparison gallery** — View generated videos together in a responsive browser interface.
- **Checkpoint selection** — Filter and select individual LoRA files for testing.
- **Consistent test settings** — Maintain the same prompt, seed, resolution, and sampling configuration across checkpoints.
- **Adjustable LoRA strength** — Apply a consistent strength across the selected checkpoints or retain the strength defined in your workflow.
- **Baseline comparison** — Optionally generate a sample without the selected LoRA to compare against the base model.
- **Built-in video playback** — Play, restart, and adjust playback speed for multiple generated videos.
- **Checkpoint ratings** — Rate results from 1–5 to identify promising epochs.
- **Persistent results** — Generated samples and run information are saved locally for later review.
- **Lightweight application** — Uses the Python standard library without requiring a large framework.

## Why use LoRA Sweep?

Training a character LoRA often produces dozens of intermediate checkpoints.

Finding the best checkpoint traditionally requires manually loading each file into ComfyUI, running a generation, waiting for the result, and repeating the process.

LoRA Sweep automates that process.

Instead of manually testing 60 checkpoints, you can queue the entire comparison and review the results in one place.

This makes it easier to identify:

- Which training epoch produces the closest character likeness.
- Whether additional training improves or reduces visual quality.
- Which checkpoints maintain character identity.
- Whether an earlier checkpoint performs better than the final model.
- Which checkpoints deserve further testing with longer or more demanding prompts.

## Requirements

- Windows, Linux, or macOS
- Python 3.8 or newer
- A working installation of ComfyUI
- A compatible ComfyUI API-format workflow
- LoRA checkpoint files accessible to ComfyUI
- A browser for the comparison interface

GPU requirements depend on the selected ComfyUI workflow and model.

## Installation

1. Download `lora_sweep.py`.
2. Place the file in your preferred folder.
3. Start ComfyUI.
4. Open a terminal in the folder containing the script.
5. Run:

```bash
python lora_sweep.py
```

By default, the application connects to ComfyUI at `http://127.0.0.1:8189` and starts its web interface at:

**http://127.0.0.1:8899**

For a different ComfyUI port:

```bash
python lora_sweep.py --comfy http://127.0.0.1:8188 --port 8899
```

## How to Use

### 1. Export your ComfyUI workflow

Open your working ComfyUI workflow and export it in **API format**.

Ensure the workflow contains a LoRA loader and a working output node.

### 2. Load the workflow

Open the LoRA Sweep web interface and upload the API workflow JSON.

The application analyzes the workflow and identifies compatible LoRA loaders and text inputs.

**Important:** If your workflow contains multiple LoRA loaders, select the specific loader you want LoRA Sweep to replace.

### 3. Configure your test

Choose a prompt, fixed seed, and LoRA strength.

The application uses the generation duration, resolution, sampler, steps, and other settings already defined in your API workflow.

### 4. Select checkpoints

Filter the available LoRA files and select the epochs you want to compare.

For example:

```text
Character-000010.safetensors
Character-000020.safetensors
Character-000030.safetensors
Character-000040.safetensors
Character-000050.safetensors
Character-000060.safetensors
```

### 5. Generate

Click **Generate all**.

LoRA Sweep submits the selected checkpoints to ComfyUI and monitors their execution.

### 6. Compare the results

Open the results gallery to review each generated sample alongside its checkpoint name.

Use the rating controls to identify your preferred results.

## Recommended Testing Strategy

For MiniMax H3 character LoRAs, start with short video samples to reduce testing time.

**Initial screening:**
- Duration: 3 seconds
- Resolution: 0.5 MP
- Fixed seed
- Identical prompt
- Consistent LoRA strength

**Final evaluation:**
- Select the best 3–5 checkpoints.
- Generate longer 10–15-second videos.
- Compare facial likeness, motion consistency, expressions, and voice quality.
- Test different LoRA strengths if necessary.

Short clips are useful for identifying promising checkpoints, while longer samples provide a better test of temporal consistency and audio quality.

## Output and Saved Results

LoRA Sweep maintains a local `sweeps` directory containing saved API workflow information and individual run folders.

Example:

```text
sweeps/
├── workflow.json
└── 20261106-014335/
    ├── run.json
    ├── 00_baseline_no_LoRA_.mp4
    ├── 01_Character-000001.mp4
    ├── 02_Character-000002.mp4
    └── ...
```

ComfyUI may also save the original generations in its configured output directory.

## Important Notes

- Use an **API-format workflow**, not a regular ComfyUI workflow JSON.
- Verify that the selected LoRA loader is correct before starting a large batch.
- Video duration is controlled by the ComfyUI workflow, not by the length of the prompt.
- Keep the seed and generation parameters identical for fair checkpoint comparisons.
- Avoid including optimizer files, training-state folders, or unrelated LoRAs in the checkpoint selection.
- Short videos are useful for preliminary evaluation but do not guarantee the best performance in longer generations.
- Running many checkpoints can take considerable time depending on your GPU and workflow.

## Compatibility

**Primary use case:** MiniMax H3 character LoRA checkpoint evaluation through ComfyUI.

The underlying approach can also be adapted to image-generation workflows, including Qwen Image, provided the workflow exposes compatible LoRA-loading and output nodes.

Compatibility with every custom node and model architecture is not guaranteed.

## Project Goal

LoRA Sweep was created to make LoRA training evaluation more efficient.

Rather than assuming the final checkpoint is always the best, users can examine how character identity and generation quality develop throughout training and select the checkpoint that performs best for their intended use.

**Train. Generate. Compare. Choose.**

---

*LoRA Sweep is an independent community utility for ComfyUI workflows and is not an official ComfyUI application.*
