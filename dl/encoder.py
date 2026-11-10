import torch
from torch import nn

from dl.config import (
    BLOCK_STRIDE,
    BRANCH_CONFIG,
    CHANNEL_DROPOUT,
    EEG_CHANNEL_INDICES,
    EMBEDDING_DIM,
    EMBEDDING_DROPOUT,
    GROUP_SLICES,
    HEAD_OUTPUTS,
    SCALAR_HEADS,
    SIGNAL_GROUPS,
)


def conv_bn(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.BatchNorm1d(out_channels),
    )


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=BLOCK_STRIDE):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(in_channels, out_channels, kernel_size, stride),
            nn.ReLU(inplace=True),
            conv_bn(out_channels, out_channels, kernel_size, 1),
        )
        self.shortcut = conv_bn(in_channels, out_channels, 1, stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.shortcut(x))


class Branch(nn.Module):
    def __init__(self, in_channels, stem_kernel, stem_stride, block_kernel, channels):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(in_channels, channels[0], stem_kernel, stem_stride), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            *[ResidualBlock1d(channels[i], channels[i + 1], block_kernel) for i in range(len(channels) - 1)]
        )
        self.out_channels = channels[-1]

    def forward(self, x):
        return self.blocks(self.stem(x)).mean(dim=-1)


def channel_keep_mask(ch_mask, drop_probability, eeg_indices):
    present = ch_mask > 0
    keep = present & (torch.rand(ch_mask.shape, device=ch_mask.device) >= drop_probability)
    eeg_present = present[:, eeg_indices]
    lost = eeg_present.any(dim=1) & ~keep[:, eeg_indices].any(dim=1)
    scores = torch.rand(eeg_present.shape, device=ch_mask.device).masked_fill(~eeg_present, -1.0)
    rescued = eeg_indices[scores.argmax(dim=1)]
    rows = lost.nonzero(as_tuple=True)[0]
    keep[rows, rescued[rows]] = True
    return keep


class PSGEncoder(nn.Module):
    def __init__(self, channel_dropout=CHANNEL_DROPOUT, embedding_dropout=EMBEDDING_DROPOUT,
                 embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.channel_dropout = channel_dropout
        self.branches = nn.ModuleDict({
            group_name: Branch(len(channels), **BRANCH_CONFIG[group_name])
            for group_name, (_, channels) in SIGNAL_GROUPS.items()
        })
        pooled_dim = sum(branch.out_channels for branch in self.branches.values())
        self.dropout = nn.Dropout(embedding_dropout)
        self.projection = nn.Linear(pooled_dim, embedding_dim)
        self.heads = nn.ModuleDict({name: nn.Linear(embedding_dim, size) for name, size in HEAD_OUTPUTS.items()})
        self.register_buffer("eeg_indices", torch.tensor(EEG_CHANNEL_INDICES, dtype=torch.long), persistent=False)

    def signals(self, batch):
        if not self.training or self.channel_dropout <= 0:
            return {group_name: batch[group_name] for group_name in SIGNAL_GROUPS}
        keep = channel_keep_mask(batch["ch_mask"], self.channel_dropout, self.eeg_indices)
        keep = keep.to(batch[next(iter(SIGNAL_GROUPS))].dtype)
        return {
            group_name: batch[group_name] * keep[:, GROUP_SLICES[group_name], None]
            for group_name in SIGNAL_GROUPS
        }

    def encode(self, batch):
        signals = self.signals(batch)
        pooled = torch.cat([self.branches[group_name](signals[group_name]) for group_name in SIGNAL_GROUPS], dim=1)
        return self.projection(self.dropout(pooled))

    def forward(self, batch):
        embedding = self.encode(batch)
        logits = {name: head(embedding) for name, head in self.heads.items()}
        for name in SCALAR_HEADS:
            logits[name] = logits[name].squeeze(-1)
        return embedding, logits
