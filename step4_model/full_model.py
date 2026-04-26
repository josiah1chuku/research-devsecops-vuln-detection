"""
full_model.py - VulnDetector Architecture
Author: Josiah Chuku, FAMU 2026
Instructor: Dr. Theran Carlos
"""
import torch
import torch.nn as nn
from transformers import RobertaModel


class RGCNLayer(nn.Module):
    """Relational GCN: 3 relation types (def-use=0, control-flow=1, call=2)."""
    def __init__(self, in_dim, out_dim, num_relations=3):
        super().__init__()
        self.weights   = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.self_loop = nn.Linear(in_dim, out_dim)
    def forward(self, x, edge_index, edge_type):
        out = self.self_loop(x)
        if edge_index.shape[1] > 0:
            for r, w in enumerate(self.weights):
                mask = (edge_type == r)
                if mask.sum() == 0: continue
                out = out.index_add(0, edge_index[1][mask], w(x[edge_index[0][mask]]))
        return torch.relu(out)


class GatedFusion(nn.Module):
    """Learned gated fusion: g=sigmoid(W[a||b]); out=g*a+(1-g)*b."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
    def forward(self, a, b):
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * a + (1 - g) * b


class VulnDetector(nn.Module):
    """
    Hybrid vulnerability detector: CodeBERT + R-GCN + Gated Fusion.
    Input : source code tokens + Data Flow Graph
    Output: binary logits [clean, vulnerable]
    """
    def __init__(self, hidden=512, gcn=256, rel=3, layers=2, drop=0.3):
        super().__init__()
        self.encoder    = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.node_proj  = nn.Linear(128, gcn)
        self.gcn_layers = nn.ModuleList([RGCNLayer(gcn, gcn, rel) for _ in range(layers)])
        self.text_proj  = nn.Linear(768, hidden)
        self.graph_proj = nn.Linear(gcn, hidden)
        self.fusion     = GatedFusion(hidden)
        self.drop       = nn.Dropout(drop)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(hidden // 2, 2))

    def forward(self, ids, mask, nf, ei, et, bv):
        # Text branch: CodeBERT CLS token
        h = self.text_proj(self.encoder(ids, mask).last_hidden_state[:, 0, :])
        # Graph branch: R-GCN + mean pooling
        x = self.node_proj(nf)
        for layer in self.gcn_layers:
            x = layer(x, ei, et)
        B = ids.shape[0]
        hg = torch.zeros(B, x.shape[-1], device=x.device)
        c  = torch.zeros(B, device=x.device)
        hg.index_add_(0, bv, x)
        c.index_add_(0, bv, torch.ones(len(bv), device=x.device))
        hg = self.graph_proj(hg / c.unsqueeze(1).clamp(min=1))
        return self.classifier(self.drop(self.fusion(h, hg)))

    def freeze_encoder_layers(self, freeze_up_to=9):
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < freeze_up_to:
                for p in layer.parameters():
                    p.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"Frozen layers 0-{freeze_up_to-1} | Trainable: {trainable:,} / {total:,}")
