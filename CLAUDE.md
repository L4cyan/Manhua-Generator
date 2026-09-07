# Project notes for Claude

## Writing style — when drafting or continuing the story

The author writes fast-paced web-novel prose. Match that voice. It is not
literary fiction and should not be dressed up as if it were.

**Do not use em dashes (—) in prose you write for the story or in replies.**
Use a comma, a full stop, or restructure the sentence. The author's own drafts
use them heavily; that is their voice, not a licence to add more.

**Banned habits:**

- Flowery or ornamental adjectives. "Ethereal luminescence cascaded" is wrong;
  "purple light filled the room" is right.
- Poetic abstraction where a concrete image would do.
- Rule-of-three lists for rhythm ("cold, distant, and utterly alone").
- Sentences that exist to sound good rather than to say something.
- Restating a beat in prettier words right after stating it plainly.
- Portentous one-line paragraphs used as fake weight.

**Match instead:**

- Short declarative sentences. Let hard beats land without decoration.
- Dialogue carries most of the characterisation. Ling Yan reasons out loud and
  deduces things; write him thinking, not emoting.
- Dry humour and self-aware asides ("A status bar in a cultivation world.
  Great.") are in voice.
- `"..."` on its own line for silence or a beat. Used often.
- System text in square brackets: `[Locating the Nearest Town, Requires 1
  Psionic Star]`
- Scene breaks as `<<----------->>`
- Mixed close-third and first-person interjections are deliberate. Keep them.

**Test:** if a sentence would feel out of place in a translated Chinese web
novel, cut it.

## Story

- `workspace/projects/psionic-cultivation/story/` holds the source chapters.
- Ling Yan is a psychology professor pulled into a cultivation world. He
  inherits the Psionic Arts, which rewrite reality but cost cultivation realms
  to fuel. He is calm to the point of coldness, deduces fast, and does not
  moralise about what he does.
- Act one is the **modern world**. Panels there must set `world: "modern"`, or
  the genre clause puts him in cultivation robes in a lecture hall.

## Code

- Never write `\n` inside a bash heredoc. It becomes a literal newline and
  breaks the file. Use the Write or Edit tool for anything containing escapes.
- `workspace/` is gitignored. The author's chapters must never be committed.
- Style edits go in `config/style.yaml` AND `workspace/styles/*.yaml`. The
  workspace copy is only seeded when missing, so editing the source alone
  leaves the running config stale.
