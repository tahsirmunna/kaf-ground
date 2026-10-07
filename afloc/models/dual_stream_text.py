"""
RadGraph graph encoder used for the graph-supervised loss L_SG.

Each report's RadGraph entities are looked up in a precomputed span-embedding
table, passed through a small residual projection and a 2-layer GAT (4 heads,
768-d, dropout 0.1), so modifier attributes propagate to the entities they
modify. The encoder returns

  * the per-node embeddings (padded to max_nodes) that L_SG aligns with image
    patches, and
  * a sentence embedding for L_DS, t_s_raw + tanh(g) * W[t_s_raw; graph].

The fusion gate g starts at zero, so at step 0 the sentence embedding is
exactly ClinicalBERT's. With fusion_init="zero" (the KAF-Ground setting) W also
starts at zero; each factor's gradient is then proportional to the other, both
stay at zero, and the graph shapes the model only through L_SG. This is the
setting behind the reported checkpoint (its gate and W are exactly zero).
fusion_init="xavier" gives W a random start so the gate can open; that variant
is reported as an ablation.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class NodeProj(nn.Module):
    """
    Residual MLP applied to the lookup-table node embeddings before the GAT.
    The table itself is fixed, so without this the attention parameters have
    nothing learnable to attend over. fc2 starts at zero, so the block is the
    identity at step 0.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):  # [N, 768] -> [N, 768]
        return x + self.fc2(self.act(self.norm(self.fc1(x))))


class GraphTextStream(nn.Module):
    """
    node embeddings -> NodeProj -> 2-layer GAT ->
    (mean-pooled graph vector, padded per-node tensor for L_SG).
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        head_dim = hidden_dim // num_heads

        self.node_proj = NodeProj(hidden_dim)

        self.gat1 = GATConv(
            in_channels=hidden_dim, out_channels=head_dim,
            heads=num_heads, dropout=dropout, concat=True,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            heads=1, dropout=dropout, concat=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, node_embeddings, edge_index, batch_idx, max_nodes: int):
        """
        Args:
            node_embeddings : [N_total, 768] — raw lookup-table embeddings, all
                               samples in the batch concatenated.
            edge_index      : [2, E_total] — COO, node indices offset per graph.
            batch_idx       : [N_total] — which sample each node belongs to.
            max_nodes       : int — pad per-node output to this length (for L_SG).

        Returns:
            graph_emb        : [B, 768]            pooled — feeds L_DS fusion
            node_emb_padded  : [B, max_nodes, 768]  pre-pool, for L_SG
            node_mask        : [B, max_nodes] bool  True = real node
        """
        h = self.node_proj(node_embeddings)              # [N_total, 768]

        x = self.gat1(h, edge_index)                      # [N_total, 768]
        x = self.act(self.norm1(x))
        x = self.gat2(x, edge_index)                       # [N_total, 768]
        x = self.norm2(x)                                  # post-GAT, PRE-POOL

        graph_emb = global_mean_pool(x, batch_idx)          # [B, 768]

        B = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
        node_emb_padded = x.new_zeros(B, max_nodes, x.shape[-1])
        node_mask = torch.zeros(B, max_nodes, dtype=torch.bool, device=x.device)
        for b in range(B):
            sel = (batch_idx == b)
            n = int(sel.sum().item())
            if n == 0:
                continue
            n_clip = min(n, max_nodes)
            node_emb_padded[b, :n_clip] = x[sel][:n_clip]
            node_mask[b, :n_clip] = True

        return graph_emb, node_emb_padded, node_mask


class DualStreamTextEncoder(nn.Module):
    """
    RadGraph graph branch. ClinicalBERT's sentence embedding t_s_raw is
    computed by the caller; this module adds the graph encoder and the gated
    fusion. Reports without a graph keep t_s_raw unchanged and get an all-False
    node mask, so they contribute nothing to L_SG.
    """

    def __init__(self, emb_matrix_cpu_fp16: torch.Tensor, max_nodes: int = 64,
                 hidden_dim: int = 768, gat_heads: int = 4, fusion_init: str = "zero"):
        super().__init__()
        self.max_nodes = max_nodes

        # A buffer, so every DDP replica receives its own copy.
        self.register_buffer("emb_matrix", emb_matrix_cpu_fp16.float())

        self.gat = GraphTextStream(hidden_dim=hidden_dim, num_heads=gat_heads)

        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        # see the module docstring
        if fusion_init == "zero":
            nn.init.zeros_(self.fusion_proj.weight)
        elif fusion_init == "xavier":
            nn.init.xavier_uniform_(self.fusion_proj.weight)
        else:
            raise ValueError(f"fusion_init must be 'zero' or 'xavier', got {fusion_init!r}")
        nn.init.zeros_(self.fusion_proj.bias)
        self.fusion_gate = nn.Parameter(torch.zeros(1))

    def current_gate_value(self) -> float:
        """tanh(gate); logged during training."""
        return torch.tanh(self.fusion_gate).item()

    def forward(self, t_s_raw: torch.Tensor, graph_data: dict):
        """
        Args:
            t_s_raw    : [B, 768] ClinicalBERT sentence embedding.
            graph_data : dict from RadGraphInjector.get_graph_features(), with
                         CPU tensors 'node_span_ids', 'edge_index', 'available_mask'.

        Returns:
            t_s_enhanced    : [B, 768]              -> L_DS (replaces t_s_raw)
            node_emb_padded : [B, max_nodes, 768]    -> for L_SG
            node_mask       : [B, max_nodes] bool    -> for L_SG
        """
        device = t_s_raw.device
        B = t_s_raw.shape[0]
        available_mask = graph_data["available_mask"].to(device)

        span_ids_list = graph_data["node_span_ids"]
        edge_index_list = graph_data["edge_index"]

        node_embs, edge_idx_offset, batch_idx = [], [], []
        offset = 0
        for i in range(B):
            span_ids = span_ids_list[i].to(device)
            emb_i = self.emb_matrix[span_ids]            # [N_i, 768]
            node_embs.append(emb_i)
            ei = edge_index_list[i].to(device)
            # DataParallel's scatter splits every nested tensor along dim 0,
            # which corrupts a [2, E] COO index. Fail loudly instead.
            if ei.numel() > 0 and (ei.dim() != 2 or ei.shape[0] != 2):
                raise ValueError(
                    f"edge_index[{i}] has shape {tuple(ei.shape)}, expected [2, E]. "
                    "Use DDP or a single GPU; DataParallel splits this tensor."
                )
            edge_idx_offset.append(ei + offset if ei.numel() > 0 else ei)
            batch_idx.append(torch.full((emb_i.shape[0],), i, dtype=torch.long, device=device))
            offset += emb_i.shape[0]

        node_embeddings = torch.cat(node_embs, dim=0)
        edge_index = (
            torch.cat(edge_idx_offset, dim=1) if any(e.numel() > 0 for e in edge_idx_offset)
            else torch.zeros(2, 0, dtype=torch.long, device=device)
        )
        batch_idx = torch.cat(batch_idx, dim=0)

        graph_emb, node_emb_padded, node_mask = self.gat(
            node_embeddings, edge_index, batch_idx, max_nodes=self.max_nodes
        )

        fusion_contrib = self.fusion_proj(torch.cat([t_s_raw, graph_emb], dim=-1))
        fused = t_s_raw + torch.tanh(self.fusion_gate) * fusion_contrib

        t_s_enhanced = torch.where(available_mask.unsqueeze(-1), fused, t_s_raw)

        # Reports without a graph contribute nothing to L_SG.
        node_mask = node_mask & available_mask.unsqueeze(-1)

        return t_s_enhanced, node_emb_padded, node_mask