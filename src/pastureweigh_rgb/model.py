"""Shared 1D segment encoder and ordered, lightweight log-weight regression."""

import torch
from torch import nn


def reverse_profile(profile):
    """Reverse the entire M*K longitudinal sequence, including segment order."""
    return profile.flip((-2, -1))


class PastureWeighRGBRegressor(nn.Module):
    def __init__(self, segments=4, samples_per_segment=32, dropout=0.1):
        super().__init__()
        self.segments = segments
        self.samples_per_segment = samples_per_segment
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(segments*32, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, profiles):
        if profiles.ndim != 3 or tuple(profiles.shape[1:]) != (self.segments, self.samples_per_segment):
            raise ValueError('Expected input [B, M, K] matching model configuration')
        batch = profiles.shape[0]
        features = self.encoder(profiles.reshape(batch*self.segments, 1, self.samples_per_segment))
        return self.head(features.reshape(batch, self.segments*32)).squeeze(-1)


def predict_weight(model, profiles):
    """Arithmetic mean in kg, not exp of averaged log predictions."""
    if isinstance(model, MorphometryRegressor):
        forward = torch.exp(model(profiles))
        if model.segments == 1:
            return forward
        return 0.5*(forward + torch.exp(model(reverse_morphometry(profiles))))
    if isinstance(model, BBoxRegressor):
        return torch.exp(model(profiles))
    return 0.5*(torch.exp(model(profiles)) + torch.exp(model(reverse_profile(profiles))))


class BBoxRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 2:
            raise ValueError('Expected bbox input [B, 2]')
        return self.head(features).squeeze(-1)


def reverse_morphometry(features):
    """Keep L fixed and reverse only the ordered segment widths."""
    return torch.cat((features[..., :1], features[..., 1:].flip(-1)), dim=-1)


class MorphometryRegressor(nn.Module):
    def __init__(self, segments=4):
        super().__init__()
        self.segments = segments
        self.head = nn.Sequential(nn.Linear(segments+1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.segments+1:
            raise ValueError(f'Expected morphometry input [B, {self.segments+1}]')
        return self.head(features).squeeze(-1)
