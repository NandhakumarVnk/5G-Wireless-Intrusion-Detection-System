import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Symmetric NT-Xent (InfoNCE) Contrastive Loss

    Input
    -----
    embedding1 : Batch × Feature

    embedding2 : Batch × Feature

    Output
    ------
    Scalar Contrastive Loss
    """

    def __init__(
        self,
        temperature=0.5
    ):

        super().__init__()

        self.temperature = temperature

    # =====================================================
    # Forward
    # =====================================================

    def forward(

        self,

        embedding1,

        embedding2

    ):

        # -----------------------------------------
        # Shape Check
        # -----------------------------------------

        if embedding1.shape != embedding2.shape:

            raise ValueError(
                "Embedding shapes must be identical."
            )

        batch_size = embedding1.size(0)

        # -----------------------------------------
        # Batch Size Check
        # -----------------------------------------

        if batch_size < 2:

            return embedding1.new_tensor(0.0)

        # -----------------------------------------
        # L2 Normalization
        # -----------------------------------------

        z1 = F.normalize(

            embedding1,

            p=2,

            dim=1

        )

        z2 = F.normalize(

            embedding2,

            p=2,

            dim=1

        )

        # -----------------------------------------
        # Cosine Similarity
        # -----------------------------------------

        similarity = torch.mm(

            z1,

            z2.t()

        )

        similarity = similarity / self.temperature

        # -----------------------------------------
        # Positive Labels
        # -----------------------------------------

        labels = torch.arange(

            batch_size,

            device=embedding1.device

        )

        # -----------------------------------------
        # Forward Direction
        # -----------------------------------------

        loss_forward = F.cross_entropy(

            similarity,

            labels

        )

        # -----------------------------------------
        # Reverse Direction
        # -----------------------------------------

        loss_backward = F.cross_entropy(

            similarity.t(),

            labels

        )

        # -----------------------------------------
        # Symmetric NT-Xent
        # -----------------------------------------

        loss = (

            loss_forward +

            loss_backward

        ) / 2

        return loss