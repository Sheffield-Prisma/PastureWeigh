"""Predict cattle weight from full-resolution binary masks and a local checkpoint."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .geometry import extract_bbox, extract_morphometry, extract_profile, extract_torso_lw
from .model import BBoxRegressor, MorphometryRegressor, PastureWeighRGBRegressor, predict_weight
from .pca_torso import extract_morphometry_pca
from .train_test import EXPERIMENTS, normalisation_arrays


class Predictor:
    """Restore geometry, training statistics, and regression weights together."""

    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        self.config = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = self.config
        self.experiment = config['experiment']
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f'Unsupported experiment: {self.experiment}')
        if self.experiment.startswith('morph_'):
            self.model = MorphometryRegressor(config['M'])
        elif self.experiment in ('bbox', 'torso_lw'):
            self.model = BBoxRegressor()
        else:
            self.model = PastureWeighRGBRegressor(config['M'], config['K'], config['dropout'])
        self.model.load_state_dict(config['model_state_dict'], strict=True)
        self.model.to(self.device).eval()
        self.mean, self.std = normalisation_arrays(config)
        if (not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(np.asarray(self.std) <= 0)):
            raise ValueError('Checkpoint contains invalid normalization statistics')

    @torch.inference_mode()
    def predict(self, mask):
        """Return one positive weight in kilograms without resizing the mask."""
        config = self.config
        if self.experiment.startswith('morph_'):
            if config.get('torso_mode', 'legacy') == 'pca_peak':
                features, _ = extract_morphometry_pca(mask, config['M'])
            else:
                features, _ = extract_morphometry(mask, config['M'], config['tau'])
        elif self.experiment == 'bbox':
            features, _ = extract_bbox(mask)
        elif self.experiment == 'torso_lw':
            features, _ = extract_torso_lw(mask, config['tau'])
        else:
            features, _, _ = extract_profile(
                mask, config['M'], config['K'], config['tau'], config['median_window'],
                profile_region=config['profile_region'])
        normalized = ((features - self.mean) / self.std).astype(np.float32)
        inputs = torch.from_numpy(normalized[None]).to(self.device)
        weight = float(predict_weight(self.model, inputs).item())
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Model produced a nonpositive or non-finite weight')
        return weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('masks', type=Path, nargs='+', help='Single-channel binary mask images (0/1 or 0/255)')
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, args.device)
    rows = []
    for path in args.masks:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            parser.error(f'Cannot read a single-channel mask: {path}')
        values = set(np.unique(mask).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            parser.error(f'Expected a binary mask: {path}')
        rows.append({'mask': str(path), 'predicted_weight_kg': predictor.predict(mask)})
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
