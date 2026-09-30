# harit7.github.io — project notes

Personal academic site (Jekyll, served by GitHub Pages from `main`).
Excluded from the Jekyll build via `_config.yml`, so it never becomes a public page.

## Layout

- `_data/papers.json`: the single source for the publication list. Manually ordered, newest first.
- `_layouts/home.html`: renders `site.data.papers` (around line 109). Only `title`, `authors`,
  `venue`, `date` (year only), `award`, `links`, and `url` are displayed. `id`, `type`,
  `venue_short` are bookkeeping and never rendered.
- `assets/pdf/`: local PDFs (slides, camera-ready papers). Link them as `/assets/pdf/<name>.pdf`.
- `_posts/`: blog posts. `_data/blogs.json`, `misc.json`, `ws_pubs.json`, `preprint.json` exist
  but `ws_pubs.json` and `preprint.json` are empty and unused.

## Conventions in `papers.json`

- `id`: `F-n` for full papers, `S-n` for short/workshop papers. Numbers only increase; the list
  position is chosen by hand.
- `type`: `full_pub` (published), `full_pre` (preprint/under review), `short` (workshop).
- `authors`: HTML string, Harit wrapped in `<b>…</b>`.
- `venue`: full name, the year is appended by the template from `date`.
- Two-venue papers: `venue_2`/`venue_3` are supported by the template but reuse the same `date`,
  so they print the same year. Avoid unless both venues share a year.

## Preview

```
./run.sh            # bundle exec jekyll serve → http://127.0.0.1:4000
```

`_config.yml` is not hot-reloaded; restart the server after editing it.

## Log

### 2026-09-29
- Added F-18 (full paper) "Paperena: Reliable Science from Unreliable AI" (AI for Science WS @ NeurIPS 2026).
  PDF is compiled from `~/workspace/Paperena-ai/paperena-paper/neurips_ws_version/main_ws_neurips.tex`
  (author block added, `[dblblindworkshop, final]`, `\workshoptitle{AI for Science}`). Compile from
  that dir with `TEXINPUTS=.:..: BIBINPUTS=.:..: latexmk -pdf main_ws_neurips.tex` because figure
  paths assume the repo root. All 14 authors have affiliations.
  Author block uses two `\parbox`es (names, affiliations) with each entry in an `\mbox`, so the
  list wraps inside the text width and names never split. `\thanks` on Amit gives the
  personal-capacity footnote (asterisk mark, distinct from the numeric affiliation marks).
  Contact email is info@paperena.ai. The taller camera-ready author block pushes the body past
  page 8; per Harit that is fine. The `\newpage` before References and the `\clearpage` before the
  appendix are commented out so References and the appendix flow on directly. Figures untouched.
  The 8 negative `\vspace`s in `sections/experiments.tex` and `sections/proposed_environment.tex`
  (submission-time squeezes) are commented out too, each tagged `camera-ready`.
- Added F-17 (full paper) "Don't Be Choosy: Scoring over Choosing for Verbalized LLM Confidence"
  (UncertaiNLP WS @ EMNLP 2026). PDF is the camera-ready.
- F-14 (ASAT): TMLR → NeurIPS 2026 Journal-to-Conference track. Kept the TMLR acceptance in the
  venue text rather than `venue_2` because of the shared-year limitation above. Moved to the top
  of the list and dated 2026 so it renders as NeurIPS 2026.
- Added this file and excluded it from the build.
- News section: first 5 items visible, the rest inside a native `<details class="news-more">`
  ("Older news" toggle, no JS). Dropped the fixed 200px scroll box on `.news-scroll` so the expanded
  list is not trapped in a small scrolling area. To rotate news, move items across the `</ul>`/`<details>`
  boundary in `_layouts/home.html`.
- Nav: menu entries in `_data/menu.json` accept `"hidden": true` (skipped by `_includes/header.html`).
  The Fun tab is hidden this way, not deleted; `/fun/` still builds and is reachable by URL.
- News item added for the Agents4Academia Oxford–Singapore Hackathon (June 14–26, 2026; Oxford Stats,
  NUS, NTU; supported by Anthropic). Source: agents4academia.org/events/2026-oxford-sg/.
- Added S-4 (short paper) "PRIOR: Inspectable Contribution Maps for Scientific Synthesis" (under review, 2026;
  Kaleb, Vishwakarma, Young, Teh, all Oxford). No link yet: `url` is "#" and `links` omitted; fill in
  when an arXiv/OpenReview link exists.
- Blog tab is driven by Jekyll `_posts/` via `paginator.posts` in `_layouts/page.html`; `_data/blogs.json`
  is unused. External posts: add a `_posts/` file with `redirect_to:` (jekyll-redirect-from) and an
  explicit `excerpt`, e.g. `_posts/2025-10-24-SnorkelSpatial.md` → Snorkel AI blog.
- Paper order in `papers.json` now: Paperena (F-18), Don't Be Choosy (F-17), PRIOR (S-4), then older.
- Service section now starts with a "Co-organized" list (hackathon, AIR-FM workshop) above the reviewing
  list, same `service-list` styling. Add future organizing roles there.
- News: added [08/26] Paperena contributed talk at the AI Scientist Summer Workshop (Aug 4, 2026, MSR New England); "[11/25] Joined Oxford" moved to older news to keep five visible.
- News: added [10/26] Paperena poster at the Frontier Data Summit (Snorkel AI, SF, Oct 8, 2026); AIR-FM item moved to older news.
- News link convention (Harit): anchors go on the *name* of the thing (paper name, event name, project
  name), kept short. Links double as highlights, so never link filler words like "paper", "talk", "poster".
- LIL blog post (`_posts/2026-04-15-LIL-Time-Uniform.md`) reviewed and corrected: maximal inequality now
  uses the epoch end t_{k+1} in the exponent and derives the boxed bound via t_{k+1} <= eta*t; log log
  domain stated as t >= 3; LIL attribution (Hartman–Wintner); recap constant fixed. Previous post's
  final bound constant fixed too (pi^2 t^2 / (3 delta), and "=" -> "<="), plus X_t -> X_i typo.
- Footer: socials row and the credit line are centred independently (`_includes/footer.html`).
- News: added [10/26] Simons Institute "Trustworthy AI: From Hallucinations to Reliable Autonomy" workshop (Oct 5-9, 2026); MLSys item moved to older news.
- Paperena callout under About Me (`.callout` in `_sass/main.scss`). Its "Get in touch" link reads
  `paperena_form_url` from `_config.yml`; empty falls back to paperena.ai. Paste the Google Form URL
  there and restart `jekyll serve` (config is not hot-reloaded).
- Paperena interest form (Google Form, to be created by Harit). Planned fields:
  1. Name  2. Email  3. Affiliation
  4. I'm interested as: User / Collaborator / Both / Just want updates
  5. What frustrates you most about AI-written papers today? (short answer, optional)
  6. What would you want Paperena to do about it? Features, checks, anything. (long answer, optional)
  7. One line on how you'd use it or how you'd like to collaborate (optional)
  8. Anything else (optional)
