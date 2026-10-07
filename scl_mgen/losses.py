import torch
from torch.nn import functional as F


def supervised_contrastive(z, labels, temperature):
    logits = z.float() @ z.float().T / temperature
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return z.sum() * 0
    log_denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    log_probability = logits - log_denominator[:, None]
    per_anchor = -(log_probability.masked_fill(~positive, 0).sum(1) / counts.clamp_min(1))
    return per_anchor[valid].mean()


def dirichlet_kl(alpha):
    alpha = alpha.float()
    strength = alpha.sum(-1)
    classes = alpha.shape[-1]
    normalizer = torch.lgamma(strength) - torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha.new_tensor(float(classes)))
    return normalizer + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(strength)[:, None])).sum(-1)


def evidential_loss(alpha, labels, weights, beta, mode="printed"):
    alpha = alpha.float()
    target = F.one_hot(labels, alpha.shape[-1]).float()
    strength = alpha.sum(-1, keepdim=True)
    probability = alpha / strength
    error = (target - probability).square().sum(-1)
    if mode == "printed":
        regularizer = dirichlet_kl(alpha)
    elif mode == "sensoy":
        error = error + (alpha * (strength - alpha) / (strength.square() * (strength + 1))).sum(-1)
        regularizer = dirichlet_kl(target + (1 - target) * alpha)
    else:
        raise ValueError(f"Unknown evidential loss interpretation: {mode}")
    return ((error + beta * regularizer) * weights[labels]).mean()


def composite_loss(output, labels, weights, training, stage):
    scl = supervised_contrastive(output["contrastive"], labels, training["temperature"])
    if stage == 1:
        return scl, scl, scl.detach() * 0
    if output["alpha"] is None:
        supervised = (F.cross_entropy(output["logits"].float(), labels, reduction="none") * weights[labels]).mean()
    else:
        supervised = evidential_loss(output["alpha"], labels, weights, training["beta"], training["loss_mode"])
    if training["stage1_epochs"] == 0:
        scl = scl * 0
    return scl + training["loss_coefficient"] * supervised, scl, supervised
