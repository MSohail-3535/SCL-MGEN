import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


def masked_mean(states, mask):
    weights = mask.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(1) / weights.sum(1).clamp_min(1)


def pool_segments(states, segment_ids):
    count = int(segment_ids.max().item()) + 1
    if count == 0:
        raise ValueError("No valid sentence tokens")
    batch, _, width = states.shape
    pooled = states.new_zeros(batch, count, width)
    denominator = states.new_zeros(batch, count, 1)
    valid = segment_ids >= 0
    indices = segment_ids.clamp_min(0).unsqueeze(-1)
    pooled.scatter_add_(1, indices.expand(-1, -1, width), states * valid.unsqueeze(-1))
    denominator.scatter_add_(1, indices, valid.unsqueeze(-1).to(states.dtype))
    return pooled / denominator.clamp_min(1), denominator.squeeze(-1) > 0


class SCLMGEN(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        self.settings = config["model"]
        self.backbone = backbone or AutoModel.from_pretrained(self.settings["backbone"], revision=self.settings["revision"])
        hidden = self.backbone.config.hidden_size
        if hidden % self.settings["heads"] or self.settings["fusion_dim"] % self.settings["fusion_heads"]:
            raise ValueError("Attention heads must divide hidden/fusion dimensions")
        self.streams = self.settings["streams"]
        if not self.streams or len(set(self.streams)) != len(self.streams):
            raise ValueError("At least one distinct granularity is required")
        self.refinement = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for name in self.streams:
            layers = self.settings["refinement_blocks"]
            block = nn.TransformerEncoderLayer(hidden, self.settings["heads"], self.settings["feedforward"],
                                               self.settings["dropout"], activation="relu", batch_first=True)
            self.refinement[name] = nn.TransformerEncoder(block, layers, enable_nested_tensor=False) if layers else nn.Identity()
            if layers:
                for layer in self.refinement[name].layers:
                    for param in layer.parameters():
                        if param.dim() > 1:
                            nn.init.xavier_uniform_(param)
            self.projections[name] = nn.Linear(hidden, self.settings["fusion_dim"])
        dimension = self.settings["fusion_dim"]
        self.attention = nn.MultiheadAttention(dimension, self.settings["fusion_heads"], self.settings["dropout"], batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.ff = nn.Sequential(nn.Linear(dimension, self.settings["fusion_feedforward"]), nn.ReLU(),
                                nn.Dropout(self.settings["dropout"]), nn.Linear(self.settings["fusion_feedforward"], dimension))
        self.drop = nn.Dropout(self.settings["dropout"])
        projection = []
        in_dim = dimension
        for _ in range(self.settings["projection_layers"] - 1):
            projection.extend([nn.Linear(in_dim, self.settings["projection_hidden"]), nn.ReLU()])
            in_dim = self.settings["projection_hidden"]
        projection.append(nn.Linear(in_dim, self.settings["projection_dim"]))
        self.projection = nn.Sequential(*projection)
        self.classifier = nn.Linear(dimension, 2)
        if self.settings["gradient_checkpointing"] and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, input_ids, attention_mask, sentence_ids, paragraph_ids=None, character_tokens=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        content = (sentence_ids >= 0) & attention_mask.bool()
        available = {"word": (hidden, content)}
        if "sentence" in self.streams:
            available["sentence"] = pool_segments(hidden, sentence_ids)
        if "document" in self.streams:
            document = masked_mean(hidden, content).unsqueeze(1)
            available["document"] = (document, torch.ones_like(content[:, :1]))
        if "paragraph" in self.streams:
            available["paragraph"] = pool_segments(hidden, paragraph_ids)
        if "character" in self.streams:
            valid = character_tokens >= 0
            chars = hidden.gather(1, character_tokens.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
            available["character"] = (chars, valid)
        sequences, masks = [], []
        for name in self.streams:
            states, mask = available[name]
            module = self.refinement[name]
            refined = module(states, src_key_padding_mask=~mask) if not isinstance(module, nn.Identity) else states
            sequences.append(self.projections[name](refined))
            masks.append(mask)
        joined, valid = torch.cat(sequences, 1), torch.cat(masks, 1)
        attended, _ = self.attention(joined, joined, joined, key_padding_mask=~valid, need_weights=False)
        fused = self.norm1(joined + self.drop(attended))
        fused = self.norm2(fused + self.drop(self.ff(fused)))
        pooled = masked_mean(fused, valid)
        logits = self.classifier(pooled)
        if self.settings["evidential"]:
            activation = self.settings["evidence_activation"]
            evidence = {"softplus": F.softplus, "relu": F.relu, "elu_plus_one": lambda x: F.elu(x) + 1}[activation](logits.float())
            alpha = evidence + 1
            strength = alpha.sum(-1, keepdim=True)
            probabilities, uncertainty = alpha / strength, 2 / strength.squeeze(-1)
        else:
            alpha = None
            probabilities = logits.float().softmax(-1)
            uncertainty = 1 - probabilities.max(-1).values
        return {"embedding": pooled, "contrastive": F.normalize(self.projection(pooled).float(), dim=-1),
                "logits": logits, "alpha": alpha, "probabilities": probabilities, "uncertainty": uncertainty}

    def set_trainable_stage(self, stage, epoch, training):
        for param in self.parameters():
            param.requires_grad_(True)
        if stage == 2:
            return
        for param in self.classifier.parameters():
            param.requires_grad_(False)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        frozen = training["frozen_epochs"]
        if epoch < frozen:
            return
        layers = self.backbone.encoder.layer
        steps = max(1, training["stage1_epochs"] - frozen)
        count = math.ceil(len(layers) * (epoch - frozen + 1) / steps)
        for layer in layers[-count:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        if count >= len(layers):
            for param in self.backbone.parameters():
                param.requires_grad_(True)
