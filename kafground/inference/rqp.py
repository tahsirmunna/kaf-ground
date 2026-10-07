"""
RadGraph Query Pruning (RQP).

A grounding query such as "small left apical pneumothorax" mixes the finding
and its location with qualifiers that carry no spatial information ("small").
Under word-max matching every word competes for every patch, so a qualifier
that happens to match some unrelated region can blur the map.

RQP drops the qualifier words, using a lexicon derived once from the RadGraph
parses of the training reports: a token is a qualifier when its RadGraph
entities are almost always the *source* of a `modify` relation
(share >= 0.90, attested >= 100 times). The thresholds come from the corpus
statistics (true size/severity words sit at >= 0.97 with thousands of
occurrences; parse noise on function words is far lower and rarer), not from
MS-CXR results. Only the query phrase is used at test time; the report graph of
the test study is never read.

Because a bad prune should never replace the query, RQP returns three variants
whose maps are averaged: the original phrase, the pruned phrase, and the
pruned phrase in a template ("findings consistent with ...").
"""
import re
from pathlib import Path

DEFAULT_LEXICON = Path(__file__).with_name("rqp_modifiers.txt")
_MODIFIERS = None

_EXISTENTIAL = re.compile(r"^\s*there\s+(is|are)\s+(a|an|the)?\s*", re.I)


def load_modifiers(path=DEFAULT_LEXICON):
    words = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line)
    return words


def _modifiers():
    global _MODIFIERS
    if _MODIFIERS is None:
        _MODIFIERS = load_modifiers()
    return _MODIFIERS


def set_lexicon(path):
    """Use a different modifier list (e.g. one rebuilt with build_rqp_lexicon.py)."""
    global _MODIFIERS
    _MODIFIERS = load_modifiers(path)


def is_modifier(word: str) -> bool:
    return word.strip(",.;:").lower() in _modifiers()


def prune(phrase: str) -> str:
    """Remove qualifier words; never returns an empty query."""
    kept = [w for w in phrase.split() if not is_modifier(w)]
    return " ".join(kept) if kept else phrase


def query_variants(phrase: str, mode="rg-ens"):
    """
    Query variants whose maps are averaged.

        orig     the phrase as written (RQP off)
        rg-ens   RQP: original + pruned + templated (the paper's setting)
        rg-core  pruned phrase only
        ens/core the same with a hand-written qualifier list instead of the
                 RadGraph lexicon (ablation)

    `mode` may also be a bool: True -> "rg-ens", False -> "orig".
    """
    if mode is True:
        mode = "rg-ens"
    if mode is False or mode == "orig":
        return [phrase]
    core = prune(phrase) if mode.startswith("rg-") else prune_handwritten(phrase)
    if mode in ("core", "rg-core"):
        return [core]
    if mode not in ("ens", "rg-ens"):
        raise ValueError(f"unknown query mode '{mode}'")
    stem = _EXISTENTIAL.sub("", core).strip() or core
    out = []
    for v in (phrase, core, f"findings consistent with {stem}"):
        if v not in out:
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# ablation: a hand-written qualifier list instead of the RadGraph lexicon       #
# --------------------------------------------------------------------------- #
HANDWRITTEN_QUALIFIERS = {
    # size / severity
    "small", "tiny", "large", "moderate", "mild", "minimal", "trace", "subtle",
    "extensive", "massive", "slight", "marked", "significant", "modest",
    "sizable", "substantial", "moderate-sized", "small-to-moderate",
    "large-sized", "millimetric", "gross",
    # temporal / status
    "stable", "new", "increased", "decreased", "unchanged", "persistent",
    "improved", "improving", "worsening", "worsened", "residual", "recurrent",
    "interval", "again", "similar", "redemonstrated", "re-demonstrated",
    "known", "chronic", "acute", "continued", "ongoing",
    # hedging
    "probable", "possible", "possibly", "likely", "suspected", "questionable",
    "apparent", "presumed", "concerning", "suggestive", "equivocal",
}


def prune_handwritten(phrase: str) -> str:
    kept = [w for w in phrase.split() if w.strip(",.;").lower() not in HANDWRITTEN_QUALIFIERS]
    return " ".join(kept) if kept else phrase


# --------------------------------------------------------------------------- #
# ablation: keep only some RadGraph token roles inside the word-max             #
# --------------------------------------------------------------------------- #
def load_roles(path):
    """{word: "anat" | "finding" | "modifier"} written by build_rqp_lexicon.py --roles."""
    import json
    return json.loads(Path(path).read_text())


def subword_roles(roles, token_strings):
    """Give every BERT sub-word the role of the whole word it belongs to
    ('##' pieces attach to the left); words outside RadGraph are 'oov'."""
    out, cur, idxs = [None] * len(token_strings), "", []

    def flush():
        for j in idxs:
            out[j] = roles.get(cur, "oov")

    for i, t in enumerate(token_strings):
        if t.startswith("##"):
            cur += t[2:]
            idxs.append(i)
        else:
            flush()
            cur, idxs = t.lower(), [i]
    flush()
    return out
