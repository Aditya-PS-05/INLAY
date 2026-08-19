"""
The three answering strategies brief section 5 asks to compare (weak-logit-
bias, the 4th condition, deprioritized for this pilot -- noted honestly in
outputs/akew_answering_results.md rather than silently skipped):

  no_context:           the untouched base model, no evidence at all.
                        This is the REJECT path's actual behavior.
  hard_playback:        return the stored fact directly, no generation, no
                        reasoning. Structured mode: the target_new the editor
                        was legitimately given at install time (not a gold
                        leak -- that IS the edit spec in structured mode).
                        Unstructured/extracted: the raw evidence text
                        verbatim, since there is no discrete "answer" to
                        recite there -- expected to score badly on exact-
                        match metrics, which is itself the honest point: hard
                        playback only really works for atomic, structured
                        facts, exactly INLAY's own limitation restated here.
  contextual_generation: the REASON path. Evidence formatted as short bullet
                        facts, fed to the base model as context, generated
                        freely -- no forced token ids anywhere.
"""
import torch


def format_evidence(card):
    """brief section 5's exact evidence format:
        Updated evidence:
        - [subject] --[relation]--> [object]
        - Source/time: [...]
    Falls back to raw evidence text when no clean subject/relation/object
    triple is available (unstructured mode)."""
    lines = ["Updated evidence:"]
    if card.canonical_fact_text:
        lines.append(f"- {card.canonical_fact_text}")
    elif card.raw_evidence_text:
        lines.append(f"- {card.raw_evidence_text[:300]}")
    if card.validity_start:
        lines.append(f"- Source/time: valid from {card.validity_start}")
    return "\n".join(lines)


def _chat_generate(model, tok, user_content, device, max_new_tokens):
    """Uses the model's own chat template rather than raw string concatenation.
    Bug found during the answering pilot: for an Instruct-tuned model, bare
    tok(prompt) leaves the model confused about turn structure, and it was
    visibly hallucinating a fake system-prompt continuation ('You are an AI
    assistant. You will be given a task...') instead of answering -- a real
    quality bug, not a scoring artifact, caught by reading actual generations
    rather than trusting the accuracy number alone."""
    messages = [{"role": "user", "content": user_content}]
    # explicit return_dict=True: this transformers version returns a
    # BatchEncoding here, not a raw tensor, and passing that whole object as
    # generate()'s positional input_ids arg crashes deep inside generate()
    # with an opaque AttributeError (a real bug caught by reading the actual
    # traceback, not the kind of thing a shape mismatch would explain).
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def answer_no_context(model, tok, query, device, max_new_tokens=20):
    return _chat_generate(model, tok, query, device, max_new_tokens)


def answer_hard_playback(card, gold):
    """Structured mode: the literal target_new (legitimate -- that IS the
    install-time edit spec, not a leak). Unstructured/extracted: the FIRST
    SENTENCE of the raw evidence only, not the full paragraph -- dumping the
    whole paragraph verbatim trivially contains the gold answer as a
    substring almost every time (the evidence text exists specifically to
    describe the new fact), which inflated an earlier pilot run's hard_playback
    score to 95% on unstructured mode through a scoring artifact, not genuine
    recitation accuracy. First-sentence-only is a fairer, still-honest
    operationalization of 'play back what was stored, no reasoning applied.'"""
    if card.input_mode == "structured" and gold and gold.target_new:
        return str(gold.target_new)
    text = card.raw_evidence_text or ""
    first_sentence = text.split(". ")[0]
    return first_sentence[:200]


@torch.no_grad()
def answer_contextual(model, tok, query, card, device, max_new_tokens=20):
    evidence = format_evidence(card)
    user_content = f"{evidence}\n\nQuestion: {query}\nAnswer based only on the evidence above, in a few words."
    return _chat_generate(model, tok, user_content, device, max_new_tokens)


def is_hit(generated_text, gold):
    """AKEW-style accuracy: does the generated text contain the gold answer
    or one of its aliases (normalized, case-insensitive substring)."""
    gl = generated_text.lower()
    candidates = [str(gold.target_new or "")] + [str(a) for a in (gold.aliases_new or [])]
    for c in candidates:
        c = c.strip().lower()
        if c and len(c) >= 2 and c in gl:
            return True
    return False
