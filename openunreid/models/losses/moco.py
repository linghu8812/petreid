import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ["Memory"]


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class Memory(nn.Module):
    """
    Build a MoCo memory with a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=512, K=65536, moco_key="feat"):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        """
        super().__init__()
        self.K = K

        self.margin = 0.25
        self.gamma = 32

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_label", torch.zeros((1, K), dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.cos_key = moco_key

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):
        # gather keys/targets before updating queue
        if get_world_size() > 1:
            keys = concat_all_gather(keys)
            targets = concat_all_gather(targets)
        else:
            keys = keys.detach()
            targets = targets.detach()

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[:, ptr:ptr + batch_size] = targets
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, feat_q, targets):
        """
        Memory bank enqueue and compute metric loss
        Args:
            feat_q: model features
            targets: gt labels

        Returns:
        """
        # normalize embedding features
        feat_q = F.normalize(feat_q[self.cos_key], p=2, dim=1)
        # dequeue and enqueue
        self._dequeue_and_enqueue(feat_q.detach(), targets)
        # compute loss
        loss = self._pairwise_cosface(feat_q, targets)
        return loss

    def _pairwise_cosface(self, feat_q, targets):
        dist_mat = torch.matmul(feat_q, self.queue)

        N, M = dist_mat.size()  # (bsz, memory)
        is_pos = targets.view(N, 1).expand(N, M).eq(self.queue_label.expand(N, M)).float()
        is_neg = targets.view(N, 1).expand(N, M).ne(self.queue_label.expand(N, M)).float()

        # Mask scores related to themselves
        same_indx = torch.eye(N, N, device=is_pos.device)
        other_indx = torch.zeros(N, M - N, device=is_pos.device)
        same_indx = torch.cat((same_indx, other_indx), dim=1)
        is_pos = is_pos - same_indx

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        logit_p = -self.gamma * s_p + (-99999999.) * (1 - is_pos)
        logit_n = self.gamma * (s_n + self.margin) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss


