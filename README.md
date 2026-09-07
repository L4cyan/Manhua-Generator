# Manhua Generator

Turn prose into a **vertical long-strip manhua** — with the art style, the cast, and the framing all locked down, so panel 200 still looks like panel 1.

> **Status:** working alpha. The full pipeline runs end to end (workspace → breakdown → render → letter → compose → export). The mock backend needs no GPU, so you can try the whole thing in about a minute.

---

## Why this exists

Plenty of tools generate comic *pages*. Almost none generate a **long strip**, and the ones that exist all drift: by page 30 the protagonist is a different person.

The usual approach is to feed the previous image in as a reference for the next one. That accumulates error — each step looks fine next to the last, and sixty steps later you have a stranger. Consistency isn't a model problem, it's a **pipeline** problem. This project solves it with four locks:

| Lock | What it does | Where |
|---|---|---|
| **Style** | One frozen checkpoint, prompt, and sampler config for the entire series. Never varies per panel. | `workspace/styles/*.yaml` |
| **Character** | A trained identity LoRA per character, plus an immutable appearance string injected into every prompt. | `characters.yaml` + `manhua sheet` |
| **Composition** | The LLM picks from a fixed camera vocabulary instead of inventing framing each time. | `models.py` → `SHOT_TOKENS` |
| **Drift gate** | Every panel is scored against the character's turnaround sheet — a *fixed anchor*, not the previous panel — and re-rolled if it goes off-model. | `qa/drift.py` |

The drift gate is the part nobody else does, and it's what makes a 200-panel chapter survivable.

---

## Quick start (no GPU needed)

```bash
git clone https://github.com/L4cyan/Manhua-Generator.git
cd Manhua-Generator
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt

python -m manhua.cli new "My Story"
python -m manhua.cli studio --backend mock
```

On Windows you can just double-click **`start.bat`** — it builds the venv on first run and opens the studio.

The browser opens on the workspace. The `mock` backend draws prompt cards instead of art, so you can exercise the layout, lettering, and export without downloading a single model.

---

## Render backends

**ComfyUI is optional.** Pick whichever fits your hardware — the studio,
compositor and lettering are identical either way, and a given seed produces
the same panel on any of them.

| `backend` | Renders on | Measured speed | Needs |
|---|---|---|---|
| `native` | Your GPU, in-process | ~230 s/panel @ 6 GB | `requirements-local.txt` |
| `remote` | Free Kaggle/Colab GPU | ~25 s/panel | [notebook worker](kaggle/README.md) |
| `comfy` | A running ComfyUI | ~60 s/panel | ComfyUI on `:8188` |
| `mock` | Nothing — prompt cards | instant | nothing |

Set it in `workspace/settings.json`, or per command with `--backend`.
Everything is auto-detected on first run: existing checkpoints (including a
ComfyUI install's), VRAM, and whether torch is available.

> **6 GB cards:** SDXL doesn't fit, so weights shuttle between RAM and VRAM
> every step — roughly a 10× penalty. That's the 230 s above. A free 16 GB
> Kaggle GPU removes it entirely; see [kaggle/README.md](kaggle/README.md).

### Models

- **Checkpoint** — a *clean* Illustrious base, not a heavily-merged style checkpoint.
- **Style LoRA** — [Korean Manhwa/Webtoon Style](https://civitai.com/models/2179842/korean-manhwa-webtoon-style-lora-2025) or [Manhwa Artstyle](https://civitai.com/models/257995/manhwa-artstyle-or-webtoon-or-lora).

> **Why base + style LoRA, not a baked manhwa checkpoint?** Your character LoRAs get trained against *base* Illustrious. A heavy style merge has drifted weights and will fight them — degrading identity exactly where you can least afford it. Keeping the base clean means style and identity dial independently, and manhwa-ness becomes a knob (`0.4` subtle → `0.8` heavy) instead of a checkpoint swap.

Then `python -m manhua.cli doctor`.

### Script breakdown — free and offline

Prose → panels runs on **local Ollama** if it's installed:

```bash
ollama pull qwen3.5:4b
```

Output is constrained by a JSON schema during sampling, so even a small model
can't emit malformed panels. Claude is used instead if `ANTHROPIC_API_KEY` is
set — better panelling, but it bills per chapter. Auto-detected; override with
`MANHUA_PROVIDER=ollama|claude`.

### 3. Character LoRAs — the important one

```bash
python -m manhua.cli sheet my-story ling_yan
```

Renders ~24 reference images (angles + expressions) through your project's own style lock, then writes a kohya training config.

**Then cull the sheet by hand.** Delete anything that isn't clearly the same person. This step matters more than any training hyperparameter — a sheet with three faces in it trains a LoRA that produces three faces.

Train with [kohya_ss](https://github.com/bmaltais/kohya_ss), drop the `.safetensors` into `ComfyUI/models/loras/`, and set the character's `lora:` field.

> On 6GB VRAM, LoRA *training* takes 1–2 hours (inference is already offload-bound at ~230s/panel). A free Kaggle P100 does both far faster — see [kaggle/README.md](kaggle/README.md).

**Why this step is not optional:** an appearance string gets you roughly 70%
consistency. In a four-panel test using only text, the same character rendered
black hair in one panel and magenta-and-teal streaks three panels later. Over
sixty panels that compounds into a different person. The LoRA is the load-bearing
part of the identity lock.

---

## How you actually use it

The studio is built around one loop, because batch-generating a whole chapter and *then* looking at it gives you a hundred panels of drift and no leverage:

```
paste a scene  →  review the panels as TEXT  →  render one beat
      ↑                                              ↓
      └────────  reroll / rewrite / reshoot  ←───────┘
```

You're never more than a beat away from a course correction. Per panel you can:

- **↻ Reroll** — new seed, same prompt
- **✎ Edit** — change shot, framing, action, lighting, FX. Re-renders at the **same seed** on purpose, so you see what the *wording* changed rather than confounding it with a whole new composition
- **🔒 Lock** — mark a panel final. Locked panels are never re-rendered by anything
- **✕ Delete**

Editing the breakdown as text before rendering is the cheapest iteration in the pipeline. Use it.

---

## Workspace layout

Everything is plain files — portable, diffable, and nothing trapped in a database.

```
workspace/
  styles/
    xianxia-premium-webtoon.yaml    shared style locks
  projects/
    my-story/
      project.json                  name, description, chosen style
      characters.yaml               the identity lock
      story/ch01.md                 source prose
      chapters/001/
        chapter.json                panels + per-panel status
        panels/*.png                rendered art
        export/                     composed strip + upload slices
```

`workspace/` is gitignored, so your stories never end up in the repo.

---

## Prompt assembly (worth understanding)

SDXL's CLIP encodes in **77-token chunks** and weights early tokens hardest. A full manhua style block is ~200 tokens — put it all in front and your *character identity* lands in chunk 3, where it carries almost no weight. That's a drift source most people never notice.

So the style is split:

```
style_lead   (~24 tok)  ─ front-loaded style anchor
panel content           ─ framing, CHARACTER IDENTITY, action, setting, fx
style_body  (~126 tok)  ─ the bulk of the aesthetic description
suffix                  ─ quality tail
```

Identity lands in **chunk 0**. The style LoRA holds the look regardless of token position, which is what makes demoting the style text safe.

Check any panel:

```bash
python -m manhua.cli prompt my-story 1 c001_p003
```

It warns if identity slips out of chunk 0.

---

## Export

```bash
python -m manhua.cli export my-story 1
```

Produces `ch001_full.png` (the whole strip) plus numbered upload slices capped at 1280px. **Cuts snap to gutters** — never through a face — and a panel taller than the cap is emitted whole rather than damaged.

---

## Commands

| Command | Does |
|---|---|
| `studio` | Launch the workspace UI |
| `new "Title"` | Create a project with Chapter 1 |
| `projects` | List projects |
| `script <proj> <ch> <file.md>` | Prose → panels |
| `render <proj> <ch>` | Headless render |
| `sheet <proj> <char>` | Turnaround sheet + LoRA config |
| `export <proj> <ch>` | Compose + slice the strip |
| `prompt <proj> <ch> <panel>` | Show a panel's exact prompt |
| `doctor` | Check the setup |

---

## Architecture

```
manhua/
  models.py        Panel / Character / Chapter schema — the contract between stages
  config.py        Style lock loading + prompt assembly
  workspace.py     Projects, chapters, style library
  script/          Prose → panels (Claude, schema-validated)
  render/          Backend interface + ComfyUI adapter + mock
  bible/           Turnaround sheets + LoRA training configs
  qa/              CLIP drift detection
  letter/          Balloon drawing + auto-placement
  compose/         Long-strip assembly + gutter-snapped slicing
  studio/          FastAPI server + no-build-step web UI
```

Adding a backend means implementing one method (`Backend.render`). The rest of the pipeline doesn't change.

---

## Roadmap

- [ ] ControlNet pose/depth for directed framing
- [ ] Drag-to-place balloons in the studio
- [ ] Panel reordering by drag
- [ ] Multi-character scene composition (regional prompting)
- [ ] Style transfer from reference images

---

## Contributing

The mock backend means you can work on layout, lettering, export, and UI with no GPU and no model downloads. `pip install -r requirements.txt` and go.

## Licence

MIT. You own everything you make with it.

Art styles aren't copyrightable, so a style prompt targeting a look you like is fine — but build with **original characters**. Don't ship a repo that reproduces someone's existing cast.
