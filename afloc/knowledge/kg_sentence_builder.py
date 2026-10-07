"""
Ablation: rewrite a report as a list of its RadGraph entities
(e.g. "opacity; right lower lobe; no pleural effusion") and train on that
instead of the raw sentences. Enabled with cfg.model.text.use_kg_sentences.
It did not beat the KAF-Ground configuration (see results/ablations).
"""
import json, re
from typing import Optional

NOISE_WORDS = {
    # Stopwords
    "the", "a", "an", "of", "in", "is", "are", "was", "were",
    "and", "or", "but", "with", "for", "to", "at", "by", "from",
    "there", "this", "that", "these", "those", "it", "its",
    # Vague/filler
    "within", "limits", "unremarkable", "grossly", "otherwise",
    "interval", "prior", "previous", "again", "also", "as", "well",
    # Intensity modifiers — no spatial value alone
    "moderately", "markedly", "significantly", "slightly",
    "mildly", "severely", "mild", "moderate", "severe",
    # Temporal
    "persistent", "unchanged", "stable", "redemonstrated",
    "new", "old", "acute", "chronic", "increased", "decreased",
    # Anatomical directions — spatial only when combined
    "upper", "lower", "right", "left", "bilateral", "middle",
    "anterior", "posterior", "medial", "lateral",
    # Generic descriptors
    "normal", "clear", "small", "large", "volumes", "volume",
    "size", "contours", "surfaces", "changes", "process",
    "more", "less", "pronounced", "findings", "noted",
}

def build_kg_sentence(graph: dict, max_entities: int = 64) -> str:
    entities = graph.get("entities", {})
    if not entities:
        return "chest x-ray"
    spans = []
    for eid, ent in list(entities.items())[:max_entities]:
        tokens = ent.get("tokens", "").strip()
        if not tokens:
            continue
        label = ent.get("label", "")
        if label == "OBS-DA":
            span = f"no {tokens}"
        elif label == "OBS-U":
            span = f"possible {tokens}"
        else:
            span = tokens
        # Always keep negated/uncertain — clinically meaningful
        if label in ("OBS-DA", "OBS-U"):
            spans.append(span)
            continue
        # Skip very short tokens
        if len(tokens) <= 2:
            continue
        # Skip pure noise single words
        if len(span.split()) == 1 and span.lower() in NOISE_WORDS:
            continue
        spans.append(span)
    seen, deduped = set(), []
    for s in spans:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return "; ".join(deduped) if deduped else "chest x-ray"


class RadGraphKGBuilder:
    def __init__(self, radgraph_path: str, max_entities: int = 64,
                 fallback_to_raw: bool = True):
        self.max_entities    = max_entities
        self.fallback_to_raw = fallback_to_raw
        print(f"[KGBuilder] Loading RadGraph from {radgraph_path}...")
        with open(radgraph_path, "r") as f:
            raw = json.load(f)
        self.graphs = {k.replace(".txt", ""): v for k, v in raw.items()}
        print(f"[KGBuilder] Loaded {len(self.graphs):,} entries")
        self._hits = 0
        self._misses = 0

    def get_kg_sentence(self, study_path: str,
                        raw_report=None) -> str:
        graph = self.graphs.get(study_path)
        if graph is None:
            match = re.search(r"(p\d+/p\d+/s\d+)", study_path)
            if match:
                graph = self.graphs.get(match.group(1))
        if graph is not None:
            self._hits += 1
            return build_kg_sentence(graph, self.max_entities)
        self._misses += 1
        if self.fallback_to_raw and raw_report:
            return raw_report
        return "chest x-ray"

    def stats(self):
        total = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": self._hits / max(total, 1)}