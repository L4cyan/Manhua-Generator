# Cloud rendering on a free GPU

Your 6GB card forces CPU offload, which is roughly a 10× wall-clock penalty —
weights shuttle between RAM and VRAM on *every* diffusion step. A free 16GB
cloud GPU keeps everything resident and skips that entirely.

Critically, this runs **your checkpoint and your LoRAs**. No closed image API
(Gemini, DALL·E, Midjourney) can do that, which is why this is a self-hosted
worker rather than a vendor integration.

**Your writing never leaves your machine.** The studio, character bible,
chapters, lettering and export all stay local. The worker only ever receives
an assembled prompt string and sends back a PNG.

---

## What Kaggle gives you free

| | |
|---|---|
| GPU | Tesla P100 16GB, or 2× T4 (32GB total) |
| Quota | ~30 GPU-hours per week |
| Session cap | 12 hours, with idle disconnects |
| Cost | £0 — no card, no trial |
| Gate | **Phone verification** (required for GPU *and* internet) |

At ~25s/panel, 30 hours is roughly **4,300 panels a week**. You will not run out.

Google Colab's free tier works too, but sessions are shorter and disconnects
more aggressive. Kaggle's quota is the more predictable budget.

---

## Setup (about 15 minutes, once)

### 1. Account and verification

Sign up at [kaggle.com](https://www.kaggle.com), then **Settings → Phone
Verification**. Without this you get neither GPU nor internet, and the worker
needs both.

### 2. New notebook

**Create → New Notebook**, then in the right-hand panel:

- **Accelerator** → `GPU P100`
- **Internet** → `On`

### 3. Get your checkpoint onto Kaggle

Two options. The second is faster if you already have the file.

**A — download it in the notebook** (no upload, ~2 min):

```python
# Civitai model download. Get a token at civitai.com → Account → API Keys.
!wget -q -O /kaggle/working/model.safetensors \
  "https://civitai.com/api/download/models/<VERSION_ID>?token=<YOUR_TOKEN>"
```

**B — upload as a private Dataset** (one-time, slower but reusable):

Datasets → New Dataset → upload your `.safetensors`. Set it **Private**. Then
add it to the notebook via *Add Input*. It lands under `/kaggle/input/<name>/`.

Option B is better long-term: the upload happens once and every future session
mounts it instantly.

### 4. Run the worker

```python
!pip -q install diffusers transformers accelerate safetensors peft \
    fastapi uvicorn nest_asyncio

!git clone -q https://github.com/L4cyan/Manhua-Generator.git /kaggle/working/mg
%cd /kaggle/working/mg

!python kaggle/worker.py \
    --checkpoint /kaggle/working/model.safetensors \
    --token my-secret-123
```

It prints something like:

```
==============================================================
  PASTE THIS INTO workspace/settings.json ON YOUR MACHINE:
    "backend": "remote",
    "remote_url": "https://random-words-here.trycloudflare.com",
    "remote_token": "my-secret-123"
==============================================================
```

The tunnel is a **cloudflared quick tunnel** — no Cloudflare account needed.

### 5. Point your machine at it

Edit `workspace/settings.json`:

```json
{
  "backend": "remote",
  "remote_url": "https://random-words-here.trycloudflare.com",
  "remote_token": "my-secret-123"
}
```

Then `manhua doctor` to confirm, and launch the studio as normal.

---

## Switching between cloud and offline

One field in `workspace/settings.json`:

| `backend` | Renders on | Speed | Needs |
|---|---|---|---|
| `remote` | Kaggle/Colab GPU | ~25s/panel | Notebook running |
| `native` | Your 3060 | ~230s/panel | torch installed |
| `comfy` | Local ComfyUI | ~60s/panel | ComfyUI running |
| `mock` | Nothing (prompt cards) | instant | nothing |

Or per-command, without editing anything:

```bash
manhua render my-story 1 --backend remote
manhua render my-story 1 --backend native
manhua studio --backend mock
```

**Recommended split:** draft and reroll on `remote` while the notebook is up,
then fall back to `native` overnight for bulk renders. Panels are identical
for a given seed either way — same code, same checkpoint, same sampler.

---

## When the tunnel dies

Free sessions end after ~12 hours or on idle, and the URL changes every
restart. The client says so plainly rather than hanging:

> Could not reach the render worker. The Kaggle/Colab session may have
> expired… Restart the notebook, copy the new tunnel URL into
> workspace/settings.json, or switch backend to 'native'.

Rendered panels are already saved locally, and locked panels are never
re-rendered — so a dropped session costs you nothing but the panel in flight.

---

## Notes

- **Keep the token non-empty.** A quick tunnel URL is public; anyone who
  guesses it could otherwise queue jobs on your GPU quota.
- **The worker serialises renders.** One GPU, one job — same as local.
- **First request is slow** (~60s) while the checkpoint loads. After that the
  model stays hot for the whole session.
- **Don't put your story in the notebook.** It doesn't need it. Only prompts
  cross the wire.
