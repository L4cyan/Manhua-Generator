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

**Then cull the sheet.** Reject anything that isn't clearly the same person. This step matters more than any training hyperparameter — a sheet with three faces in it trains a LoRA that produces three faces.

```bash
# 1. reject the off-model ones (moves them out of the training directory)
python -m manhua.cli reroll my-story ling_yan --reject expr_calm full_front

# 2. refill just those slots, at fresh seeds, leaving every keeper untouched
python -m manhua.cli reroll my-story ling_yan expr_calm full_front --tries 3

# 3. promote the ones you like
python -m manhua.cli reroll my-story ling_yan --promote out/reroll/expr_calm__w0_s5242.png
```

Re-running `manhua sheet` to fill a few holes is the wrong tool: a new base seed changes *every* slot, so the twenty images you kept come back as twenty different pictures. `reroll` re-renders only what you name.

Culling also tells you what your appearance string actually says. If every rejected render shares one trait, that trait is in the prompt. But **fix it in the framing text, not in the appearance string** — see below. Getting that backwards costs more than the defect did.

### Don't fix composition in the identity lock

`appearance` is emitted verbatim into every prompt and is what the LoRA trains on. It decides the character's *face*. Editing it to fix a *composition* problem changes the face, every time.

A worked example from this repo. Ling Yan's appearance said `curtain bangs parted over the forehead`, which contradicts itself — curtain bangs hang down and cover the forehead, "parted over the forehead" uncovers it. The model picked a side per seed, so half of every reference sheet came back with a solid fringe. Two rewrites of that clause both fixed the fringe and both visibly changed his face: softer, longer, weaker jaw.

The fix that worked put the same words in `framing_hint`, which is emitted beside the shot tokens and leaves the identity block byte-identical:

```yaml
framing_hint: >-
  forehead visible, parted bangs, hair swept away from the centre of the forehead
```

A/B across six seeds: **2/6 correct before, 6/6 after, face unchanged.**

Verify prompt changes with [`scripts/hairline_ab.py`](scripts/hairline_ab.py), which renders both variants at the same seeds and *asserts* the identity block is identical when that is the variant's claim. Use [`scripts/clause_lab.py`](scripts/clause_lab.py) to try several candidate clauses at once at 4 steps. Score likeness as well as the thing you changed — a clause that scores 6/6 on bangs and moves the face is a failure.

### Training

```bash
python scripts/build_lora_dataset.py     # sheet -> captioned dataset + TOML
bash scripts/train_ling_yan.sh           # ~1 hour on 6GB
python scripts/lora_qa.py                # base vs LoRA, plus a weight sweep
```

Then drop the `.safetensors` into `ComfyUI/models/loras/` and set the character's `lora:` field.

**Captioning rule:** caption what should stay *editable*, omit what should be baked in. Name the trigger, the framing, the expression and the wardrobe; never describe the face or hair. Whatever a caption names stays promptable afterwards; whatever it omits is absorbed into the trigger word, which is where identity belongs.

> Anima LoRA training fits 6GB using kohya sd-scripts' `anima_train_network.py` with `--blocks_to_swap 20`, `--qwen_image_vae_2d`, `--network_train_unet_only` and `adafactor`: about 5 s/it at 5.9GB of 6.1GB, so 768 steps takes roughly an hour. Train against `anima-base` and apply the result to `anima-turbo` at render time. Inference is not the bottleneck it used to be: `anima-turbo` renders a panel in ~30s at 8 steps and ~10s at 4. Free Kaggle GPUs are a poor fit for Anima specifically — the T4 has no bf16 tensor cores and Anima is bf16-native, and current PyTorch wheels ship no sm_60 kernels for the P100 at all. See [kaggle/README.md](kaggle/README.md).

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

### Panels with nobody, and panels with strangers

Two cases the style lock gets wrong on its own, both fixed in `build_request`:

**Empty-cast panels.** A style lock describing an art style almost always contains clauses describing a *person* — this one has "narrow piercing eyes with highly detailed vibrant irises" among others, because that is what the art style looks like. Assert those alongside `no humans` and the model resolves the contradiction by painting a pair of giant disembodied eyes across your landscape. Panels with no characters drop the anatomy clauses and keep everything about line, colour and light.

**Crowds and background characters.** People who are not in the bible still need a count tag. Without one the panel gets `no humans` while the action describes a student, and the result is garish and half-formed. Set `extras` on the panel:

```yaml
extras: 12          # a lecture hall of students
extras_sex: female  # only used when extras == 1
```

### Registers set the era, not the place

`world` selects a genre clause. Keep it about time period and wardrobe. This project's `modern` register originally read `contemporary present day setting, real world, city, everyday clothing`, and that stray `city` overpowered twelve panels whose setting said "endless grey void, no ground, no sky". They rendered as a downtown street. Use `neutral` for anywhere that is neither, and pin the wardrobe on the character ref instead.

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
