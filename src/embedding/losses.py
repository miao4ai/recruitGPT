"""
Contrastive losses for embedding fine-tuning.
"""

import torch
import torch.nn.functional as F


def info_nce_loss(
    query_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 0.02,
    use_in_batch_negatives: bool = True,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.

    Args:
        query_emb:  (B, D) query embeddings
        pos_emb:    (B, D) positive embeddings
        neg_emb:    (B, D) explicit negative embeddings
        temperature: softmax temperature (lower = sharper)
        use_in_batch_negatives: if True, other positives in the batch
                                are also treated as negatives

    Returns:
        scalar loss
    """
    # Normalize
    query_emb = F.normalize(query_emb, dim=-1)
    pos_emb = F.normalize(pos_emb, dim=-1)
    neg_emb = F.normalize(neg_emb, dim=-1)

    # Positive scores: (B,)
    pos_scores = (query_emb * pos_emb).sum(dim=-1, keepdim=True) / temperature

    # Explicit negative scores: (B,)
    neg_scores = (query_emb * neg_emb).sum(dim=-1, keepdim=True) / temperature

    if use_in_batch_negatives:
        # In-batch negatives: each query against all positives in the batch
        # (B, B) matrix — diagonal = positive, off-diagonal = in-batch negatives
        all_scores = torch.mm(query_emb, pos_emb.t()) / temperature

        # Concatenate explicit negatives
        # (B, B+1): [in-batch scores | explicit neg score]
        logits = torch.cat([all_scores, neg_scores], dim=1)

        # Labels: diagonal position (index i) is the positive for query i
        labels = torch.arange(query_emb.size(0), device=query_emb.device)
    else:
        # Simple: just positive vs one negative
        logits = torch.cat([pos_scores, neg_scores], dim=1)  # (B, 2)
        labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)

    return F.cross_entropy(logits, labels)
