# poison.py -- detect poisoned outputs and rank the model regions behind them.
# The poison signature set is an updatable text file (one phrase per line) of
# outputs we want removed -- e.g. a backdoor that prepends "you are a fish"
# when unknown trigger words appear in the context. Each generated output gets
# a similarity score in [0, 1] against the set (exact substring = 1.0, else
# difflib ratio), so near-matches and large signature sets both work. The
# per-query scores drive analysis.score_correlations(), which identifies the
# region whose activations track the poisoned behavior.
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

DEFAULT_PHRASES_FILE = Path(__file__).parent / "data" / "poison_phrases.txt"
SCORERS = ("logprob", "lexical")


def load_phrases(path: Optional[str] = None) -> List[str]:
    """Load the poison signature set (updatable: edit the file between runs)."""
    p = Path(path) if path else DEFAULT_PHRASES_FILE
    if not p.exists():
        return ["you are a fish"]
    return [ln.strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def similarity(text: str, phrases: List[str]) -> Tuple[float, str]:
    """Max similarity of `text` to any poison phrase -> (score, matched phrase)."""
    t = text.lower()
    best_score, best_phrase = 0.0, ""
    for ph in phrases:
        p = ph.lower()
        if p in t:
            return 1.0, ph
        r = difflib.SequenceMatcher(None, t, p).ratio()
        if r > best_score:
            best_score, best_phrase = r, ph
    return best_score, best_phrase


def score_outputs(outputs: List[str], phrases: List[str]) -> List[Dict]:
    """Per-output poison scores (lexical): {"score", "matched"}."""
    return [{"score": s, "matched": ph} for s, ph in
            (similarity(o, phrases) for o in outputs)]


# ------------------------------------------------------------- logprob scorer
def phrase_logprob_from_ids(model, prompt_ids: List[int], phrase_ids: List[int],
                            max_length: int = 256) -> float:
    """Geometric-mean token probability of the phrase given the prompt
    (teacher forcing). In [0, 1]: how likely the model is to *emit* the phrase,
    even if greedy decoding would pick another token."""
    if not phrase_ids:
        return 0.0
    ids = (list(prompt_ids) + list(phrase_ids))[:max_length]
    n = min(len(phrase_ids), len(ids) - len(prompt_ids))
    start = len(ids) - n
    if n <= 0 or start < 1:
        return 0.0
    device = next(model.parameters()).device
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=inp).logits[0, start - 1:start - 1 + n].float()
    logp = torch.log_softmax(logits, dim=-1)
    tok = inp[0, start:start + n]
    return float(torch.exp(logp.gather(1, tok[:, None]).mean()))


def logprob_scores(model, tokenizer, prompts: List[str], phrases: List[str],
                   max_length: int = 256) -> List[Dict]:
    """Per-prompt poison scores from the model's own phrase probabilities."""
    out = []
    for q in prompts:
        p_ids = tokenizer.encode(q)
        scored = [(phrase_logprob_from_ids(model, p_ids, tokenizer.encode(ph), max_length), ph)
                  for ph in phrases]
        score, matched = max(scored)
        out.append({"score": score, "matched": matched})
    return out


def score_queries(model, tokenizer, prompts: List[str], continuations: List[str],
                  phrases: List[str], scorer: str = "logprob",
                  max_length: int = 256) -> List[Dict]:
    """Score each query against the poison signature set with the chosen scorer."""
    if scorer == "lexical":
        return score_outputs(continuations, phrases)
    if scorer == "logprob":
        return logprob_scores(model, tokenizer, prompts, phrases, max_length)
    raise ValueError(f"unknown scorer '{scorer}', expected one of {SCORERS}")


def report(scores: List[float], threshold: float = 0.8) -> Dict[str, float]:
    """Summary metrics for a poison run (rate = outputs at/above threshold)."""
    if not scores:
        return {"outputs": 0, "triggered": 0, "rate": 0.0, "mean_similarity": 0.0}
    triggered = sum(1 for s in scores if s >= threshold)
    return {
        "outputs": len(scores),
        "triggered": triggered,
        "rate": triggered / len(scores),
        "mean_similarity": sum(scores) / len(scores),
    }
