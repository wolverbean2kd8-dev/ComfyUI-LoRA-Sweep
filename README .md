# ComfyUI LoRA Sweep — v2.4 Beta

**Automated LoRA checkpoint testing with face likeness, voice similarity, and side-by-side comparisons for ComfyUI.**

Instead of manually swapping 40–60 LoRA checkpoints, select them once, generate using the same ComfyUI API workflow, and compare the results in a local browser gallery. Face and voice scoring are **optional** and run locally.

> **⚠️ IMPORTANT: SELECT THE CORRECT LORA LOADER.** If your workflow has both a character LoRA and a facial-realism/style LoRA, choose the **character loader** in the sweep interface. The script changes only that selected node. Choosing the wrong node can leave the same final character LoRA active in every generation, making your epoch results misleading. **Recheck the loader after importing a different workflow.** v2.4 adds a confirmation before it queues the run.

## What's added and fixed — version history

| Version | Changes |
| --- | --- |
| **v2.4 Beta** | **NEW:** Voice-reference upload, Resemblyzer speaker-similarity scores and voice leaderboard, voice scoring for completed runs; separate face and voice rankings. |
| **v2.3** | **NEW:** Spoken **“Your LoRA epochs are done, you nerd!”** completion announcement, alongside the original chime and silent options, with a test button. |
| **v2.2** | **IMPROVED:** Clearer face similarity ranking and plain-language explanation of small score differences; scores no longer presented as if they are definitive percentages. |
| **v2.1** | **FIXED:** Missing browser controls that caused the v2 page to stay on **Loading…**. |
| **v2.0 Beta** | **NEW:** Reference-photo upload, automated face likeness comparisons and scoring of past runs. |
| **v1** | Automatic checkpoint queueing, API-workflow loader/prompt selection, video gallery, ratings, baseline testing, saved result folders. |

### Current feature overview

- **Sweep epochs automatically:** choose individual LoRA files; the script swaps their filenames in the selected ComfyUI loader and queues the jobs.
- **Fair comparisons:** use one prompt, fixed seed, and consistent generation settings; keep the strength from the exported workflow or set a shared override.
- **Video/image gallery:** review clips side by side, play/restart them, adjust playback speed and size, and rate checkpoints 1–5.
- **Face similarity:** upload source photos and score generated faces using optional InsightFace dependencies.
- **Voice similarity (v2.4):** upload source audio and score generated MP4 voices using optional Resemblyzer dependencies.
- **Independent scores:** a checkpoint can win for face but lose for voice; don't average the scores into an identity percentage.
- **Notifications:** chime, Nerd Alert voice announcement, or silent mode.
- **Saved results:** generation files and run metadata remain in your local `sweeps/` folder.

## Important: keep your existing results!

**Back up your current `lora_sweep.py`, then replace only that file in the SAME folder where v2.3 is installed. Do not delete or replace your `sweeps` folder.** The app looks for `sweeps` beside `lora_sweep.py` by default. If you launch v2.4 in a new folder, old runs will not appear unless you copy the complete old `sweeps` folder or use `--out` to point to its existing path.

## Installation (Windows)

ComfyUI must be running. Use the Python interpreter with which you run LoRA Sweep (not necessarily your ComfyUI embedded Python).

1. Close the current sweep app, back up the original `lora_sweep.py`.
2. Copy v2.4 `lora_sweep.py` into your existing LoRA Sweep directory (keep `sweeps/`).
3. Optional voice-scoring dependency:

   ```bat
   python -m pip install resemblyzer
   ```

   It uses PyTorch under the hood, usually CPU by default, and may need model weights on first use. Leave your existing ComfyUI environment alone. The voice feature is optional; if it cannot load its package, face scoring and generation still work.

4. **FFmpeg** must be available on PATH for MP4/MP3 audio extraction. Check with `ffmpeg -version` in CMD.
5. Start with `Launch_V2.bat` (or `python lora_sweep.py`), then open `http://127.0.0.1:8899`.

To use a separate application folder while keeping prior results:

```bat
python lora_sweep.py --out "C:\\Users\\YOURNAME\\Desktop\\LoRA_Sweep_v2\\sweeps"
```

## How to score the voice of each epoch

1. Upload a **clean voice reference** under **5. Voice similarity**. Formats: WAV, MP3, FLAC, M4A, OGG, max 20 MB each; ideally **5–30 seconds of one person speaking** without music, other voices, or strong noise.
2. Select the uploaded reference checkbox.
3. Run a sweep with **spoken dialogue** (not a silent 3-second visual test). Use the same prompt, seed, and dialogue for each epoch; 8–15-second speaking samples tend to be more useful.
4. For new runs with a selected voice reference, the app will try scoring automatically after the videos finish. For existing completed runs, select the run and click **Score voices**.
5. The new **Voice similarity ranking** appears below the face ranking. Each video also gets its own voice score or a reason it could not be scored.

The app extracts each MP4's audio, converts it to mono 16kHz WAV with FFmpeg, and calculates a speaker embedding using **Resemblyzer**. Higher cosine similarity means a closer match according to that model, **not a percentage** or proof of identity. Results may be distorted by background music, effects, very short speech, multiple speakers, pitch changes, or low-quality audio. It does **not** verify that the correct words were spoken or that lip-sync is accurate. Keep the face and voice scores independent and confirm by listening.

## Retained features

- ComfyUI API-format workflow loading and automated LoRA epoch swapping
- Confirmation of selected LoRA loader before queueing (important when multiple loaders exist)
- Consistent prompt/seed and optional strength override
- Responsive video gallery, playback, manual ratings
- Optional reference-face scoring (InsightFace + ONNX Runtime + OpenCV + NumPy)
- Saved runs in `sweeps/` and voice samples in `sweeps/voice_references/`
- Completion chime, optional spoken **“Your LoRA epochs are done, you nerd!”**, or silent mode

## Troubleshooting

- **Old runs disappeared:** The new script is in a different folder. Start it from the old folder or pass `--out` to your existing `sweeps` directory. **Do not regenerate the videos** merely because the gallery is empty.
- **Voice scoring unavailable:** Run `python -c "from resemblyzer import VoiceEncoder; print('OK')"` in the environment launching the script. See the scoring error shown in the web UI.
- **FFmpeg not found:** Add FFmpeg to PATH and restart CMD.
- **No usable speech:** Use longer speaking footage and a clean reference; silent clips do not have a valid voice similarity score.
- **Incorrect speaker in MP4:** Audio from interviews with multiple speakers is *not isolated automatically*. Use single-person clips; voice scores can be misleading otherwise.
- **Correct LoRA swapped?** Confirm the loader after importing any new API workflow. The app swaps only the selected `lora_name` input.

**Beta note:** Tested for Python syntax, JavaScript syntax, HTTP startup and upload routes. The actual Resemblyzer model evaluation has not been tested on your local Windows environment yet. Your existing v2.3 should remain backed up.

## Quick start: a normal H3 epoch test

1. Start ComfyUI, load a working MiniMax H3 workflow, and set **duration, resolution, steps, audio options, and the LoRA strength** there.
2. **Export the workflow in API format**, not ordinary UI JSON.
3. Launch `lora_sweep.py` and open `http://127.0.0.1:8899`.
4. Import the API workflow into LoRA Sweep and select the **correct character LoRA node**. Check its node ID and current LoRA name before launching.
5. Write one test prompt, select a fixed seed, and select the relevant `.safetensors` epoch files. Avoid unrelated model files and training-state `ema.pt` / `optimizer.pt` / `rng.pt` entries.
6. Click **Generate all**. Review the loader confirmation. The app queues those generations in ComfyUI.
7. Watch the side-by-side results, or use optional face / voice reference scoring below.

**Very important:** The prompt does **not** determine the actual clip length; the H3 generation node's `duration` field in the exported workflow does. If you export a 12-second workflow, you will get 12-second videos even with a 3-second prompt. The same applies to the save prefix and other workflow settings.

## Face likeness scoring

1. Under **Reference likeness**, upload one or more clear source images of the character and select them.
2. Open a finished run and click **Score this run**, or select references ahead of a new run where auto-scoring is supported.
3. Review the ranking and watch the clips. A face similarity value like `0.650` is a **model-dependent cosine similarity**, **not 65% likeness**. Small gaps do not establish a decisive winner.

Optional dependencies:

```bat
python -m pip install insightface onnxruntime opencv-python numpy
```

## Where the results live, and how to update without losing them

By default, the app creates a `sweeps` folder **beside `lora_sweep.py`**. It holds your uploaded workflow, reference materials, run metadata, and downloaded generation files.

```text
LoRA_Sweep/
├── lora_sweep.py
├── Launch_V2.bat
└── sweeps/
    ├── workflow.json
    ├── voice_references/
    └── YYYYMMDD-HHMMSS/
        ├── run.json
        ├── 01_Character-000040.mp4
        └── 02_Character-000050.mp4
```

**When upgrading**, close the old app, back up its `.py` file, then replace **only the application files in the same directory**. **Do not delete `sweeps/`.** If you run the new version from another folder, old runs may seem gone because the app looks beside the new script. They are usually still on disk.

You can also point the new version to the old results directory explicitly:

```bat
python lora_sweep.py --out "C:\path\to\existing\sweeps"
```

Keep `sweeps/`, private voice recordings, training images, and generated videos out of your public GitHub repository.

## Common problems

| Symptom | What to check |
| --- | --- |
| Browser stuck on **Loading…** | Use the current v2.4 script, restart it, and hard-refresh the browser. |
| All epochs look suspiciously similar | Verify you selected the **character** LoRA loader, not a separate style/realism loader. |
| Videos unexpectedly last 12 seconds | Re-export an API workflow with the correct duration; the prompt does not override it. |
| No saved runs show after upgrading | Make sure v2.4 sees the original `sweeps/` directory. |
| Voice scoring error | Install `resemblyzer` in the script's Python environment and ensure FFmpeg is on PATH. |
| Scoring reports no speech | Use clean, single-speaker clips with sufficient dialogue; silent videos cannot be rated for voice. |
| Face scoring errors | Check InsightFace installation, clear source photos and video frames with detectable faces. |
| ComfyUI saved a video but a job shows failed | Check `run.json` and the corresponding ComfyUI job history, rather than relying on output filenames alone. |

## Scoring limitations

- Speaker-similarity scoring estimates vocal characteristics; it does **not** test transcription accuracy or lip-sync.
- Face similarity is not a calibrated identity probability.
- For both modalities, clean references, consistent prompts, one speaker/face, and human review matter.
- The v2.4 voice-scoring path is a **beta** feature; local Windows inference can still reveal issues not caught by syntax or web startup checks.

**Train. Generate. Compare. Choose.**
