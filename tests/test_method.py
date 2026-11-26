"""Exercise the public method using generated masks, never research artifacts."""

import base64
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from pastureweigh_rgb.dataset import PastureWeighRGBDataset, decode_mask
from pastureweigh_rgb.geometry import clean_mask, extract_morphometry, extract_profile, extract_torso_lw
from pastureweigh_rgb.inference import Predictor
from pastureweigh_rgb import inference, splits, train_test
from pastureweigh_rgb.model import (
    MorphometryRegressor, PastureWeighRGBRegressor, predict_weight,
    reverse_morphometry, reverse_profile,
)


def make_mask(identity=0):
    mask = np.zeros((128, 192), dtype=np.uint8)
    cv2.ellipse(mask, (96, 64), (36 + identity, 16 + identity), 17, 0, 360, 1, -1)
    return mask


def make_shape(mask, identity=0):
    yy, xx = np.nonzero(mask)
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
    ok, encoded = cv2.imencode('.png', mask[y0:y1+1, x0:x1+1] * 255)
    if not ok:
        raise RuntimeError('Could not encode synthetic mask')
    return {'label': 'cattle', 'shape_type': 'mask', 'group_id': identity,
            'weight': 300.0 + 20 * identity, 'points': [[x0, y0], [x1, y1]],
            'mask': base64.b64encode(encoded).decode('ascii')}


def invoke(main, arguments):
    output = io.StringIO()
    with patch('sys.argv', ['pastureweigh', *map(str, arguments)]), redirect_stdout(output):
        main()
    return output.getvalue()


def make_annotations(root):
    source = root / 'raw-frames' / 'synthetic'
    source.mkdir(parents=True)
    for identity in range(10):
        for observation in range(2):
            mask = make_mask(identity)
            stem = f'cattle_{identity}_{observation}'
            cv2.imwrite(str(source / f'{stem}.png'), mask * 255)
            document = {'imageHeight': mask.shape[0], 'imageWidth': mask.shape[1],
                        'imagePath': f'{stem}.png', 'shapes': [make_shape(mask, identity)]}
            (source / f'{stem}.json').write_text(json.dumps(document), encoding='utf-8')
    invoke(splits.main, ['--annotations-root', root])


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cv2.setNumThreads(1)

    def test_cropped_mask_round_trip_and_invalid_corners(self):
        mask = make_mask()
        shape = make_shape(mask)
        np.testing.assert_array_equal(decode_mask(shape, *mask.shape), mask)
        shape['points'][1][0] += 1
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            decode_mask(shape, *mask.shape)

    def test_geometry_preserves_scale_and_translation(self):
        mask = make_mask()
        with_hole = mask.copy()
        with_hole[63:66, 95:98] = 0
        with_hole[0, 0] = 1
        cleaned, _ = clean_mask(with_hole)
        np.testing.assert_array_equal(cleaned, mask)
        profile, _, _ = extract_profile(mask)
        shifted, _, _ = extract_profile(np.pad(mask, ((11, 7), (19, 5))))
        np.testing.assert_allclose(profile, shifted, rtol=0, atol=1e-6)
        scaled, _, _ = extract_profile(np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1))
        self.assertEqual(profile.shape, (4, 32))
        self.assertTrue(np.isfinite(profile).all())
        self.assertGreater(scaled.mean() / profile.mean(), 1.8)
        self.assertLess(scaled.mean() / profile.mean(), 2.2)
        lw, _ = extract_torso_lw(mask)
        m1, _ = extract_morphometry(mask, segments=1)
        np.testing.assert_array_equal(lw, m1)

    def test_direction_averaging_is_invariant(self):
        for model, inputs, reverse in [
            (PastureWeighRGBRegressor(), torch.randn(3, 4, 32), reverse_profile),
            (MorphometryRegressor(), torch.randn(3, 5), reverse_morphometry),
        ]:
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(predict_weight(model, inputs),
                                           predict_weight(model, reverse(inputs)))

    def test_identity_splits_and_reference_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_annotations(root)
            args = type('Args', (), {'split_dir': root / 'split'})()
            folds = train_test.read_splits(args)
            test_ids = []
            for fold in folds:
                self.assertFalse(set(fold['train_group_ids']) & set(fold['val_group_ids']))
                test_ids.extend(fold['val_group_ids'])
            self.assertEqual(sorted(test_ids), list(range(10)))
            dataset = PastureWeighRGBDataset(root / 'split/fold_1.json', 'train', root)
            self.assertEqual(dataset[0]['mask'].shape, (128, 192))
            dataset.references[0]['annotation'] = '../outside.json'
            with self.assertRaisesRegex(ValueError, 'escapes'):
                dataset[0]
            folds[0]['val'][0]['group_id'] = folds[0]['train_group_ids'][0]
            (root / 'split/fold_1.json').write_text(json.dumps(folds[0]), encoding='utf-8')
            with self.assertRaises(ValueError):
                train_test.read_splits(args)

    def test_herd_aggregation_excludes_training_identities(self):
        observations = [
            {'annotation': 'frame.json', 'shape_index': 0, 'group_id': 1,
             'gt_weight': 400, 'predicted_weight': 410},
            {'annotation': 'frame.json', 'shape_index': 1, 'group_id': 2,
             'gt_weight': 500, 'predicted_weight': 490},
            {'annotation': 'frame.json', 'shape_index': 2, 'group_id': 3,
             'gt_weight': 1e12, 'predicted_weight': 2e12},
        ]
        rows = train_test.aggregate_herds(observations, [1, 2])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['num_test_cattle'], 2)
        self.assertEqual(rows[0]['gt_herd_weight'], 900)
        self.assertEqual(rows[0]['predicted_herd_weight'], 900)

    def test_smoke_on_single_cattle_images_writes_no_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_annotations(root)
            output = invoke(train_test.main, ['--annotations-root', root, '--experiment', 'all',
                            '--smoke-test', '--device', 'cpu', '--output-dir', root / 'outputs'])
            self.assertEqual(output.count('SMOKE TEST PASSED'), len(train_test.EXPERIMENTS))
            self.assertFalse((root / 'outputs').exists())

    def test_training_and_checkpoint_inference_agree_for_all_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_annotations(root)
            variants = [(name, 'legacy') for name in train_test.EXPERIMENTS]
            variants.extend((f'morph_m{segments}', 'pca_peak') for segments in (1, 2, 4, 8))
            for name, mode in variants:
                with self.subTest(experiment=name, torso_mode=mode):
                    output = root / 'outputs' / f'{name}_{mode}'
                    fold = 'all' if name == 'torso_m4' else '1'
                    invoke(train_test.main, ['--annotations-root', root, '--experiment', name,
                           '--torso-mode', mode, '--fold', fold, '--device', 'cpu',
                           '--max-epochs', '1', '--patience', '1', '--output-dir', output])
                    checkpoint = output / 'fold_1/model.pt'
                    predictor = Predictor(checkpoint)
                    dataset = PastureWeighRGBDataset(root / 'split/fold_1.json', 'val', root)
                    with (output / 'fold_1/observation_predictions.csv').open() as handle:
                        rows = list(csv.DictReader(handle))
                    for index, row in enumerate(rows):
                        prediction = predictor.predict(dataset[index]['mask'])
                        self.assertAlmostEqual(prediction, float(row['predicted_weight']), places=5)
                    if fold == 'all':
                        summary = json.loads((output / 'summary.json').read_text())
                        self.assertEqual(summary['num_folds'], 5)
                        self.assertEqual(len(summary['folds']), 5)
                        cli_rows = json.loads(invoke(inference.main, ['--checkpoint', checkpoint,
                                              dataset[0]['image_path']]))
                        self.assertAlmostEqual(cli_rows[0]['predicted_weight_kg'],
                                               float(rows[0]['predicted_weight']), places=5)


if __name__ == '__main__':
    unittest.main()
