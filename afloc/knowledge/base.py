"""
Interface for knowledge sources that supply a per-report graph for L_SG.

Only RadGraph is implemented (radgraph_injector.py). Another source only has to
implement the two methods below; the collate function and the graph encoder do
not change.
"""

from abc import ABC, abstractmethod
import torch


class KnowledgeInjectorBase(ABC):
    """
    Interface contract for pluggable knowledge injection sources.

    Implemented by RadGraphInjector (RadGraph entity/relation graphs).
    """

    @abstractmethod
    def is_available(self, report_id: str) -> bool:
        """
        Return True if graph data exists for this report/study id.

        Args:
            report_id (str): e.g. MIMIC study_id 's50414267'
        Returns:
            bool
        """
        raise NotImplementedError

    @abstractmethod
    def get_graph_features(self, report_ids: list, device: torch.device) -> dict:
        """
        Build batched graph tensors for a list of report ids.
        Called once per batch in multimodal_collate_fn.

        Args:
            report_ids (list[str]): One id per sample in the batch.
            device (torch.device): Target device for output tensors
                (CPU here — this runs inside forked DataLoader workers;
                tensors are moved to the training device later, inside
                DualStreamTextEncoder.forward() in the main process).

        Returns:
            dict containing at least:

                'node_span_ids'  : list[LongTensor[N_i]]
                    Integer vocabulary ids per sample. N_i varies.
                    Looked up against DualStreamTextEncoder.emb_matrix.
                'edge_index'     : list[LongTensor[2, E_i]]
                    COO-format edge index per sample.
                'available_mask' : BoolTensor[B]
                    True  -> valid graph, use enhanced embedding / L_SG.
                    False -> no graph, fall back to raw text embedding,
                             zero contribution to L_SG for this sample.
        """
        raise NotImplementedError
