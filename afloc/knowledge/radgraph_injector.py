"""
RadGraph knowledge injector.

Reads the RadGraph parses of the MIMIC-CXR training reports
(MIMIC-CXR_graphs.json, keys like 'p10/p10659857/s59206984.txt') and turns each
report into a small graph: one node per RadGraph entity (at most 64), one edge
per relation (`modify`, `located_at`, `suggestive_of`). Relations are stored
inside each entity dict; there is no top-level relation list.

Design:
  * Every unique entity span in the corpus is encoded once at start-up with
    ClinicalBERT ([CLS] vector) on the CPU. The resulting table is handed to
    DualStreamTextEncoder, which keeps it as a buffer on the GPU.
  * get_graph_features() runs inside DataLoader worker processes, where CUDA
    cannot be initialised, so it only returns CPU integer tensors: span ids
    into that table plus a COO edge index.
  * RadGraph is word level ("bibasilar" and "opacities" are separate
    entities), which matches the word-level alignment used by L_SW.
"""

import json
import re

import torch

from .base import KnowledgeInjectorBase

_CPU = torch.device("cpu")
_STUDY_KEY_RE = re.compile(r"(p\d+/p\d+/s\d+)")


class RadGraphInjector(KnowledgeInjectorBase):
    """
    Builds the per-report entity graphs and a corpus-wide table of entity-span
    embeddings (encoded once at start-up), then serves batches of graphs as
    CPU index tensors.
    """

    def __init__(
        self,
        radgraph_path: str,
        text_encoder,            # ClinicalBERT (or the BertEncoder wrapper); used once, on CPU
        tokenizer,
        max_entities: int = 64,  # p95=54 in the corpus; headroom to 64
        embed_dim: int = 768,
    ):
        self.radgraph_path = radgraph_path
        self.max_entities = max_entities
        self.embed_dim = embed_dim

        print(f"[RadGraphInjector] Loading {radgraph_path} ...")
        with open(radgraph_path, "r") as f:
            raw = json.load(f)
        print(f"[RadGraphInjector] Loaded {len(raw)} reports")

        # Per-report entities and edges.
        self.study_to_graph = {}   # normalised_key -> {"spans": [...], "edges": [(i,j)]}
        span_vocab = {}            # span_text -> vocab_id (global, dedup'd across corpus)

        for raw_key, report in raw.items():
            key = raw_key[:-4] if raw_key.endswith(".txt") else raw_key
            m = _STUDY_KEY_RE.search(key)
            norm_key = m.group(1) if m else key

            entities = report.get("entities", {})
            if not entities:
                continue

            # Stable local order: sort by entity id (string keys "1","2",... in
            # these files, but sort numerically to avoid "10" < "2" string-order bugs).
            local_ids = sorted(entities.keys(), key=lambda k: int(k))
            local_id_to_idx = {eid: i for i, eid in enumerate(local_ids[: self.max_entities])}

            spans = []
            edges = []
            for eid in local_ids[: self.max_entities]:
                ent = entities[eid]
                span_text = ent.get("tokens", "").strip().lower()
                if not span_text:
                    span_text = "<unk_entity>"
                if span_text not in span_vocab:
                    span_vocab[span_text] = len(span_vocab)
                spans.append(span_vocab[span_text])

                # Relations live inside the entity dict, e.g. ent["relations"] = [["located_at","3"], ...]
                for rel in ent.get("relations", []):
                    if len(rel) != 2:
                        continue
                    _, target_eid = rel
                    if target_eid in local_id_to_idx:
                        edges.append((local_id_to_idx[eid], local_id_to_idx[target_eid]))

            if not spans:
                continue

            self.study_to_graph[norm_key] = {"spans": spans, "edges": edges}

        n_entities = sum(len(g["spans"]) for g in self.study_to_graph.values())
        n_edges = sum(len(g["edges"]) for g in self.study_to_graph.values())
        n_reports = len(self.study_to_graph)
        print(
            f"[RadGraphInjector] Parsed {n_reports} reports with graphs "
            f"({n_entities / max(n_reports,1):.2f} entities/report, "
            f"{n_edges / max(n_reports,1):.2f} edges/report)"
        )
        print(f"[RadGraphInjector] Global span vocabulary: {len(span_vocab)} unique spans")

        self.span_to_id = span_vocab

        # Encode the whole span vocabulary once. The BertEncoder wrapper has a
        # different call signature, so unwrap it to the HuggingFace model.
        if hasattr(text_encoder, "model"):
            raw_encoder = text_encoder.model
        elif hasattr(text_encoder, "bert"):
            raw_encoder = text_encoder.bert
        elif hasattr(text_encoder, "transformer"):
            raw_encoder = text_encoder.transformer
        else:
            raw_encoder = text_encoder

        raw_encoder = raw_encoder.to(_CPU)
        raw_encoder.eval()

        vocab_strings = [""] * len(span_vocab)
        for s, idx in span_vocab.items():
            vocab_strings[idx] = s

        print(f"[RadGraphInjector] Pre-encoding {len(vocab_strings)} spans on CPU (one-time cost)...")
        emb_chunks = []
        chunk_size = 256
        with torch.no_grad():
            for i in range(0, len(vocab_strings), chunk_size):
                chunk = vocab_strings[i : i + chunk_size]
                enc = tokenizer(
                    chunk, padding=True, truncation=True, max_length=16, return_tensors="pt"
                )
                out = raw_encoder(
                    input_ids=enc["input_ids"].to(_CPU),
                    attention_mask=enc["attention_mask"].to(_CPU),
                )
                # [CLS] pooled representation per span — same convention as t_s_raw
                cls = out.last_hidden_state[:, 0, :]  # [chunk, 768]
                emb_chunks.append(cls)

        self.emb_matrix = torch.cat(emb_chunks, dim=0).half()  # [V, 768], fp16, CPU
        print(f"[RadGraphInjector] emb_matrix ready: {tuple(self.emb_matrix.shape)} (fp16, CPU)")

    # ---------------------------------------------------------------------- #
    # KnowledgeInjectorBase interface                                        #
    # ---------------------------------------------------------------------- #

    def is_available(self, report_id: str) -> bool:
        return report_id in self.study_to_graph

    def get_graph_features(self, report_ids: list, device: torch.device = _CPU) -> dict:
        """
        CPU-only lookups, safe inside forked workers. `device` is accepted for
        interface compatibility and ignored.
        """
        node_span_ids, edge_index, available = [], [], []

        for rid in report_ids:
            g = self.study_to_graph.get(rid)
            if g is None or len(g["spans"]) == 0:
                node_span_ids.append(torch.zeros(1, dtype=torch.long))
                edge_index.append(torch.zeros(2, 0, dtype=torch.long))
                available.append(False)
                continue

            node_span_ids.append(torch.tensor(g["spans"], dtype=torch.long))
            if g["edges"]:
                src = [e[0] for e in g["edges"]]
                dst = [e[1] for e in g["edges"]]
                edge_index.append(torch.tensor([src, dst], dtype=torch.long))
            else:
                edge_index.append(torch.zeros(2, 0, dtype=torch.long))
            available.append(True)

        return {
            "node_span_ids": node_span_ids,
            "edge_index": edge_index,
            "available_mask": torch.tensor(available, dtype=torch.bool),
        }


def normalise_report_id(raw_id: str) -> str:
    """Shared helper so the dataset's __getitem__ and this injector agree on key format."""
    key = raw_id[:-4] if raw_id.endswith(".txt") else raw_id
    m = _STUDY_KEY_RE.search(key)
    return m.group(1) if m else key