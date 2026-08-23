# ARR October 2026 submission checklist — "On Scope Classification and Current Knowledge-Editing Benchmarks"

**Prepared 2026-08-23.** Covers the ACL-style long paper at `paper/paper.tex` (git commit
`50e5f7d`, on top of the prior restructuring at `b07bd43`). This checklist does **not** cover
`paper/paper_neurips_workshop.tex` or `paper/neurips-workshop/` — that is a separate,
concurrently-developed NeurIPS workshop submission with its own venue rules; see that track's
own checklist.

## 1. Portal and workflow (confirmed against the current ARR CFP/dates pages)

- **Portal: OpenReview**, via the ACL Rolling Review (ARR) system at `aclrollingreview.org`.
  ARR is not itself a venue — it is a shared, centralized review pool used by ACL, EACL, NAACL,
  and EMNLP.
- **Two-step workflow, confirmed current:**
  1. **Submit to ARR** (this cycle: October 2026, submission deadline **October 12, 2026**,
     11:59 PM UTC-12 "anywhere on Earth"). The paper receives reviews and a meta-review from an
     ARR action editor and 3 reviewers; no venue decision is made at this stage.
  2. **Commit the reviewed paper to a specific venue's cycle** once reviews are in hand — e.g.
     ACL 2026, EACL 2026, or a later conference's commitment window — via that venue's
     OpenReview "Conference Commitment" site. Reviews are decoupled from acceptance: a paper
     with ARR reviews that hasn't yet been accepted anywhere can be committed to any
     ARR-affiliated venue whose commitment window is open, and authors are not required to
     commit to the venue they named as "preferred" at submission time.
  3. Other cycle-specific dates (author-response window, review deadline, meta-review deadline)
     are listed as **TBA** on `aclrollingreview.org/dates` as of this writing; only the
     October 12, 2026 submission deadline is fixed so far. Re-check `aclrollingreview.org/dates`
     closer to the deadline for the rest of the cycle's timeline.
  4. **All authors must register as an ARR reviewer** (or already be a Senior Area Chair) within
     48 hours of the submission deadline via a link that appears in the OpenReview author
     console immediately upon submission. Non-compliant papers may be desk-rejected — this is
     an author action, not something this repo can pre-complete.

## 2. Format requirements confirmed against the current ARR CFP

| Requirement | ARR's current rule | This paper |
|---|---|---|
| Content page limit (long paper) | Up to 8 pages of content, plus unlimited pages for references, a required Limitations section, and an optional Ethics/Ethical-considerations section | 8 content pages (Introduction through Conclusion + Reproducibility paragraph), confirmed by page-by-page PDF text extraction — Limitations/Ethics Statement/Acknowledgements/References fall on page 9 |
| Style files | Official ACL style files, unmodified | Uses `paper/acl-style/acl.sty` (via `\usepackage[review]{acl}`) unmodified |
| Limitations section | Required; missing section is grounds for desk rejection | Present (`\section*{Limitations}`, 6 numbered points) |
| Anonymity | As of Jan 2024, non-anonymous preprints are permitted while under review, but the **submission PDF itself** must still be properly anonymized (no author names/affiliations, no de-anonymizing acknowledgements or links) | Author block reads "Author Name Redacted for Review" / "Affiliation Redacted for Review" / a redacted-for-review email placeholder; Acknowledgements section contains no identifying information |
| Generative-AI disclosure | Must be disclosed in **(a)** the Acknowledgements section of the paper **and (b)** the separate Responsible NLP Research Checklist filled out on the OpenReview submission form; coding-only assistance may also be noted in README files | **(a) done** — added to `paper/paper.tex` Acknowledgements section this session (commit `50e5f7d`). **(b) not done** — this is an OpenReview form field, not part of the PDF; see §4 below, this is the user's manual step |
| Responsible NLP Research Checklist | Required as part of the OpenReview submission form; incorrect/incomplete/misleading answers are grounds for desk rejection; if accepted, the completed checklist is published as a paper appendix | Not part of `paper.tex` — filled out on the OpenReview form at submission time, manual step |
| Appendix formatting | Since July 2025, appendices must follow the same double-column format as the main paper (exceptions require prior approval) | Appendix A tables use `table*` (full-width, two-column-consistent) environments, matching main-text formatting |
| Supplementary material | Optional; code/data supplementary material is allowed, should include licenses/documentation as appropriate | Prepared at `paper/arr_supplementary.zip` (295 KB) — see §3 |

**No ARR-cycle-specific requirement beyond the above was found for October 2026 specifically**
(e.g. no special theme, no changed page limit for this cycle) — the October 2026 entry on the
ARR dates page carries only the submission deadline confirmed above; other rules are the
standing ARR CFP rules applicable to every cycle.

## 3. What was prepared and verified this session

- **`paper/paper.tex`** — 2 stale-content fixes (a direct contradiction between Sec 7's
  constructed out-of-scope result and a Limitations bullet that said the condition "remains
  unbuilt"; a stale claim that no method besides base/RAG/INLAY has RippleEdits matched-manifest
  numbers, when Table 2 already has ROME/WISE/AlphaEdit) + 1 data-fidelity fix (Appendix Table 3
  GRACE-Mistral row was showing em-dashes for values — PS, write-cost — that are actually
  reported in `outputs/akew_mistral_gapfill_results.md`) + the required AI-disclosure paragraph
  added to Acknowledgements. Committed at `50e5f7d`.
- **`paper/paper.pdf`** — rebuilt via `tectonic -X compile paper.tex` in the `tex` conda
  environment. Zero LaTeX/BibTeX errors; the only build warning is a pre-existing, benign
  `lineno.sty:296` UTF-8-byte substitution unrelated to this paper's content (comes from the ACL
  style file itself, present before this session's edits). All 19 `refs.bib` entries are cited
  and vice versa (verified programmatically); all `\ref{}` targets resolve to a `\label{}`; the
  final convergent build reports zero undefined citations and zero undefined references.
  12 pages total: 8 content pages, page 9 (Limitations + Ethics Statement + Acknowledgements +
  References), pages 10-12 (Appendix A).
- **Self-tests re-run this session** (not trusting the prior report), in the `inlay-paper` conda
  environment with `OMP_NUM_THREADS=1 KMP_AFFINITY=disabled` (documented in `requirements.txt`):

  | Test | Result |
  |---|---|
  | `src/akew_stats_test.py` | PASS |
  | `src/akew_reliability_test.py` | PASS |
  | `src/akew_outcome_head_test.py` | PASS |
  | `tests/test_gate_consistency.py` | PASS |
  | `tests/test_pk_capacity.py` | PASS |
  | `tests/test_akew_leakage.py` | PASS |

  **6/6 passed.**
- **`paper/arr_supplementary.zip`** (295 KB, well under any typical 100 MB venue supplementary
  cap) — contains `src/` (70 Python files, all experiment/harness/test code), `outputs/*.md`
  (36 result write-ups, including the full audit trail this checklist and the paper's own
  citations trace back to), and `requirements.txt` (pinned dependency versions + the
  `OMP_NUM_THREADS`/`KMP_AFFINITY` sandbox workaround note). Excludes model weights, datasets,
  and all git internals (`.git/`) — verified by listing the zip's contents.
- **Traceability spot-check performed this session** (not exhaustive, but covering every
  headline number and every appendix table): abstract's `+0.0000 -> +0.0420` pooled-headroom
  claim recomputed independently from `outputs/akew_outofscope_condition_results.md`'s per-cell
  table and matches to 4 decimal places; the original zero-headroom pooled figure (`0.7874`)
  likewise recomputed from the 9-cell table in `paper.tex` and matches; every numeric cell in
  the Appendix CounterFact/zsRE/RippleEdits-detail tables was checked for presence in
  `outputs/full-results-audit.md` and/or `outputs/full_results_audit_2026-08-19_addendum.md`
  and/or `outputs/akew_mistral_gapfill_results.md`.

## 4. What remains — manual, for the user

- **Create/confirm an OpenReview account** with a complete profile (current affiliation, DBLP
  and/or Semantic Scholar links, current + past emails) — required for ARR's conflict-of-interest
  detection and reviewer/AC matching.
- **Submit `paper/paper.pdf` (and `paper/arr_supplementary.zip` if supplementary material is
  offered by the submission form) through the ARR October 2026 OpenReview submission form**
  before the October 12, 2026 deadline.
- **Fill out the Responsible NLP Research Checklist** on the OpenReview submission form itself —
  this cannot be pre-filled into the PDF; it is a separate structured form. When answering its
  generative-AI question, the honest answer is consistent with the Acknowledgements paragraph
  now in `paper.tex`: an AI coding assistant was used for experiment iteration, debugging, GPU
  job execution, and manuscript drafting under direct human supervision; the human author
  originated the research questions, hypotheses, experimental design, and interpretation.
- **Select a preferred venue** in the submission form (used only for acceptance-rate bookkeeping
  and does not bind the eventual commitment choice).
- **Register as an ARR reviewer within 48 hours of the submission deadline** (the registration
  link appears in the OpenReview author console once the paper is submitted).
- **After reviews arrive**, decide whether to commit the paper to a specific venue's conference
  commitment site (e.g. ACL 2026 or EACL 2026, whichever has an open commitment window at that
  point), or to revise and resubmit to a later ARR cycle. This decision point is outside this
  session's scope and depends on the content of the reviews.
- **Camera-ready, if accepted**: the accepted-paper format allows one additional content page
  (up to 9), and the Responsible NLP Checklist will be published as a public appendix per
  current ARR/EMNLP policy (confirmed applies to subsequent *ACL conferences generally) —
  revisit the checklist wording with that publication in mind if/when this stage is reached.

## 5. Final artifact paths

- Paper source: `paper/paper.tex`
- Paper PDF: `paper/paper.pdf`
- Bibliography: `paper/refs.bib`
- ACL style files: `paper/acl-style/`
- Figures (vector PDF): `paper/figures/*.pdf`
- Supplementary code + results archive: `paper/arr_supplementary.zip`
- This checklist: `outputs/arr_october_submission_checklist.md`
- Git commit with this session's content fixes: `50e5f7d`
