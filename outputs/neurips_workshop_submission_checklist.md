# NeurIPS 2026 CL4FMAgents Workshop — Submission Checklist

Target: **"Continual Learning in the Era of Foundation Models and Embodied
Agents"** workshop at NeurIPS 2026, Sydney, Australia (Dec 11–12, 2026).
Workshop site: https://neurips26-cl4fmagents.github.io/

**Submission deadline: August 29, 2026, 11:59 PM AoE.** Notification:
September 29, 2026. Camera-ready: October 10, 2026.

## 1. Portal

- **Submit at:** https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/CL4FMAgents
- The portal was confirmed open as of the workshop site's July 2026 news
  post ("Submission portal is now open on OpenReview").
- **This is the user's own final action** — an OpenReview account and the
  actual submission click cannot be performed on the user's behalf. The
  user should:
  1. Log into (or create) an OpenReview account.
  2. Navigate to the group URL above.
  3. Start a new submission and upload the PDF at
     `paper/paper_neurips_workshop.pdf` (committed at git hash `daa8571`
     in the `salt-modular-lm`-adjacent repo, path
     `knowledge-editing/inlay`).

## 2. Track and format confirmed from the CFP

- Two tracks: **regular papers (up to 8 pages excluding references)** and
  **short papers (up to 4 pages excluding references)**. This submission
  targets the **regular track**.
- **Double-blind review**, **non-archival**, managed through OpenReview.
  Each submission receives at least three reviews.
- Style file: official **`neurips_2026.sty`**, `dblblindworkshop` package
  option (workshop-specific double-blind variant of the main NeurIPS
  style), fetched from
  `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`.
  `\workshoptitle{Continual Learning in the Era of Foundation Models and
  Embodied Agents}` is set per the style file's own convention for
  workshop submissions.
- The NeurIPS Paper Checklist (15 items, `checklist.tex`) is required —
  "papers not including the checklist will be desk rejected" — and is
  included, filled in with paper-specific justifications. It does not
  count toward the page limit.

## 3. Page/format compliance

- **Achieved: 8 content pages** (title through the Limitations section)
  before the Ethics Statement, Acknowledgments, and References. This
  matches the "8 pages excluding references" limit for the regular track.
- Ethics Statement + Acknowledgments spill onto the start of page 9,
  ahead of the References section — consistent with the NeurIPS
  template's own page-counting convention, under which acknowledgments,
  references, the checklist, and the (optional) technical appendix do not
  count as content pages.
- Total PDF: 17 pages (8 content + Ethics/Acks/References ~1 page +
  1-page condensed appendix + 7-page standard NeurIPS checklist).
- All 5 figures (`fig1_architecture.pdf`–`fig3_headroom.pdf` used; fig4/
  fig5 omitted for space, see cut list below) render at native or
  near-native resolution within the workshop's 5.5in single-column text
  width — no figure regeneration was needed, and `make_figures.py` was
  not modified.
- Compiled with `tectonic -X compile paper_neurips_workshop.tex`
  (environment `tex`): **zero LaTeX/BibTeX errors, zero undefined
  references**, only benign underfull-vbox warnings.

## 4. Required OpenReview metadata (fill in at submission time)

- **Title:** "Scope Classification as a Continual-Editing Bottleneck: A
  Negative Result on Current Knowledge-Editing Benchmarks, with a
  Gradient-Free Case Study"
- **Abstract:** copy verbatim from `paper_neurips_workshop.tex` (the
  `\begin{abstract}...\end{abstract}` block) — do not retype, OpenReview
  abstracts are compared against the PDF by reviewers.
- **Keywords / subject areas:** suggested tags matching the CFP's listed
  topics — "knowledge editing", "continual learning", "lifelong
  learning", "memory architectures", "model editing", "benchmarks and
  evaluation protocols", "catastrophic forgetting". Pick from whatever
  controlled vocabulary the OpenReview form offers; these are the closest
  matches to the workshop's own topic list.
- **Author list:** per the double-blind requirement, list authors as
  **"Redacted for Review" / "Affiliation Redacted for Review"** in the
  OpenReview submission's visible metadata during the review period,
  matching the PDF. OpenReview requires real author identities to be
  entered into the system's own (hidden-from-reviewers) author field even
  under double-blind review — enter the real name/affiliation/email there
  (available in this repo's git history prior to commit `5a0df0c`, i.e.
  commit `10a1476`) so it is correctly attributed once the double-blind
  period ends, but do **not** put it in the visible PDF or the
  reviewer-facing form fields.
- **Submission type:** Regular paper (8-page track).
- **Conflicts of interest / TPMS:** fill in per OpenReview's standard
  prompts (paper text has no information relevant to this beyond the
  author's own profile, which the user should complete directly).

## 5. What was cut from the ACL/ARR version, and why

| Cut | Where | Reasoning |
|---|---|---|
| Detailed MQuAKE-CF multi-hop figure (`fig4_multihop.pdf`) and its full paragraph | §5 (Sequential editing subsection) | Lowest-priority cut per the stated priority order — a secondary result. Replaced with a one-sentence pointer to the companion full-length version; the 47.5%/22.5%/5.0% numbers are omitted from the workshop PDF but remain fully reported in the ACL version. The MQuAKE-CF cells of the headroom table (§6, kept in full) are unaffected by this cut. |
| Full four-model-family ES/PS/NS/write-cost tables for CounterFact and zsRE (`tab:appendix-cf`, `tab:appendix-zsre`) and the full RippleEdits propagation/preservation detail table | Appendix | Condensed to one representative summary table (GPT-J-6B CounterFact structured, the cell actually discussed in the main text) pointing readers to the ACL/companion version and released code for full tables, per the stated cutting priority (appendix depth is the *most* acceptable cut). |
| `fig5_actions.pdf` (per-action success-rate bar chart) | §6 | Kept the headroom figure (`fig3_headroom.pdf`) and the headroom table (both load-bearing for the core claim); dropped the secondary per-action figure to save space, since the abstention-succeeds-19/1689 fact it illustrates is stated in text. |
| Sequential-editing ROME/MEMIT collapse discussion | §5.3 | Kept in full — this is the multi-hop *paragraph* that was cut, not the sequential-editing result, which survives intact with its $n{=}25$, $0.0$/$1.0$ numbers. |

**Never cut, per the explicit instruction:** the core negative-result
argument (oracle router ties static policy, headroom $=0.0000$ in all
nine cells, Table "headroom"), the constructed out-of-scope condition
result ($+0.0000 \to +0.0420$, REJECT-only-correct $0/1689 \to 52/1689$),
and both self-audit disclosures (the margin-gate no-op finding and the
gate-bypass bug with its validation rerun table) — all present in the
workshop PDF at full strength, word-for-word consistent with the source
numbers.

## 6. Fidelity verification performed

Every numeric value appearing in `paper_neurips_workshop.tex`'s
math-mode spans (`$...$`) was programmatically diffed against
`paper.tex`'s numeric content; the only non-matching tokens were LaTeX
figure-width scale factors (`0.9\textwidth`, `0.52\textwidth`,
`0.55\textwidth`), not data claims. No number was invented or re-rounded.

## 7. What is NOT done here (user's own final action)

- Creating/logging into an OpenReview account.
- Clicking "Submit" on the OpenReview portal.
- Entering the real (non-anonymized) author identity into OpenReview's
  author-profile field (required by OpenReview even under double-blind
  review, but never visible to reviewers pre-decision).
- Any TPMS/conflict-of-interest declarations OpenReview requests at
  submission time.

## Files

- PDF to upload: `paper/paper_neurips_workshop.pdf`
- Source: `paper/paper_neurips_workshop.tex`
- Style files used: `paper/neurips_2026.sty`, `paper/checklist.tex`
  (also mirrored under `paper/neurips-workshop/`)
- Commit: `daa8571` on branch `main`
