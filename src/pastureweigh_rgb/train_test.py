#!/usr/bin/env python3
"""Nested identity-safe epoch selection, refitting, and held-out cattle evaluation.

Run with pastureweigh-train or python -m pastureweigh_rgb.train_test.
--smoke-test never starts CV training or saves models. The split's `val`
field is always OUTER TEST data.
"""

import argparse
from collections import Counter, defaultdict
import csv
import copy
import json
import os
from pathlib import Path
import random
import statistics

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .dataset import PastureWeighRGBDataset
    from .geometry import extract_profile, extract_bbox, extract_torso_lw, extract_morphometry
    from .pca_torso import extract_morphometry_pca, extract_torso_pca_peak
    from .model import (PastureWeighRGBRegressor, BBoxRegressor, MorphometryRegressor,
                        predict_weight, reverse_profile, reverse_morphometry)
else:
    from dataset import PastureWeighRGBDataset
    from geometry import extract_profile, extract_bbox, extract_torso_lw, extract_morphometry
    from pca_torso import extract_morphometry_pca, extract_torso_pca_peak
    from model import (PastureWeighRGBRegressor, BBoxRegressor, MorphometryRegressor,
                        predict_weight, reverse_profile, reverse_morphometry)


ANNOTATIONS = Path('data/annotations')
EXPERIMENTS = ('torso_m1', 'torso_m4', 'torso_m8', 'torso_m16', 'fullmask_m4', 'torso_lw', 'bbox',
               'morph_m1', 'morph_m2', 'morph_m4', 'morph_m8')
METRICS = ('cattle_MAE_kg', 'cattle_RMSE_kg', 'cattle_MAPE_percent',
           'herd_MAE_kg', 'herd_RMSE_kg', 'herd_MAPE_percent',
           'observation_MAE_kg', 'observation_RMSE_kg', 'observation_MAPE_percent')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sort_ids(ids):
    return sorted(ids, key=lambda x: (type(x).__name__, x))


def reference_key(ref):
    return ref['annotation'], ref['shape_index']


def read_splits(args):
    """Validate complete instance coverage as well as the five identity partitions."""
    paths = [args.split_dir / f'fold_{i}.json' for i in range(1, 6)]
    for path in paths + [args.split_dir / 'summary.json']:
        check(path.is_file(), f'Missing split file: {path}')
    folds = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    canonical = {}
    for ref in folds[0]['train'] + folds[0]['val']:
        key = reference_key(ref)
        check(key not in canonical, f'Duplicate instance reference: {key}')
        canonical[key] = ref
    all_ids = {ref['group_id'] for ref in canonical.values()}
    check(bool(canonical), 'No cattle instances in split')
    weight_by_id = {}
    for ref in canonical.values():
        gid, weight = ref['group_id'], ref['weight']
        check(gid is not None, f'Missing group_id: {ref}')
        check(gid not in weight_by_id or weight_by_id[gid] == weight,
              f'Inconsistent ground-truth weight for group_id {gid}')
        weight_by_id[gid] = weight
    val_ids_seen, val_samples_seen = Counter(), Counter()
    for i, fold in enumerate(folds, 1):
        check(fold['fold'] == i, f'Incorrect fold number in {paths[i-1]}')
        train_ids, test_ids = set(fold['train_group_ids']), set(fold['val_group_ids'])
        check(len(train_ids) == len(fold['train_group_ids']) and
              len(test_ids) == len(fold['val_group_ids']), f'Fold {i}: duplicate group IDs')
        check(not train_ids & test_ids, f'Fold {i}: train/test identity overlap')
        check(train_ids | test_ids == all_ids and bool(train_ids) and bool(test_ids),
              f'Fold {i}: incomplete or empty identity partition')
        seen = Counter()
        for partition, ids in [('train', train_ids), ('val', test_ids)]:
            for ref in fold[partition]:
                key = reference_key(ref)
                check(ref == canonical.get(key), f'Fold {i}: inconsistent instance reference {key}')
                check(ref['group_id'] in ids, f'Fold {i}: identity in wrong partition {key}')
                seen[key] += 1
            check({ref['group_id'] for ref in fold[partition]} == ids,
                  f'Fold {i}: declared IDs do not match instances')
        check(seen == Counter(list(canonical)), f'Fold {i}: missing or duplicate instances')
        val_ids_seen.update(test_ids)
        val_samples_seen.update(reference_key(ref) for ref in fold['val'])
    check(val_ids_seen == Counter(all_ids), 'Validation IDs do not appear exactly once')
    check(val_samples_seen == Counter(list(canonical)), 'Held-out instances do not appear exactly once')
    summary = json.loads((args.split_dir / 'summary.json').read_text(encoding='utf-8'))
    check(summary['total_cattle_instances'] == len(canonical) and
          summary['total_unique_group_ids'] == len(all_ids) and
          set(summary['all_group_ids']) == all_ids, 'Split summary disagrees with references')
    return folds


def prepare_geometry(args):
    folds = read_splits(args)
    samples = {}
    equivalence_max = 0.0
    # Fold 1 train+val covers every instance once. Geometry is deterministic and
    # label-independent; only normalisation and training fit anything to data.
    for partition in ('train', 'val'):
        dataset = PastureWeighRGBDataset(args.split_dir/'fold_1.json', partition, args.annotations_root)
        for index in range(len(dataset)):
            sample = dataset[index]
            key = reference_key(sample)
            try:
                if args.experiment.startswith('morph_'):
                    if args.torso_mode == 'pca_peak':
                        profile, diagnostics = extract_morphometry_pca(sample['mask'], args.segments)
                    else:
                        profile, diagnostics = extract_morphometry(sample['mask'], args.segments, args.tau)
                    if args.smoke_test and args.segments == 1 and args.torso_mode == 'legacy':
                        lw, _ = extract_torso_lw(sample['mask'], args.tau)
                        difference = float(np.max(np.abs(profile-lw)))
                        equivalence_max = max(equivalence_max, difference)
                        check(difference == 0, f'{key}: morph_m1 differs from torso_lw')
                elif args.experiment == 'bbox':
                    profile, diagnostics = extract_bbox(sample['mask'])
                elif args.experiment == 'torso_lw':
                    profile, diagnostics = extract_torso_lw(sample['mask'], args.tau)
                else:
                    profile, diagnostics, _ = extract_profile(
                        sample['mask'], args.segments, args.samples_per_segment,
                        args.tau, args.median_window, profile_region=args.profile_region)
            except Exception as exc:
                raise ValueError(f'{key}, group_id={sample["group_id"]}: geometry extraction failed: {exc}') from exc
            samples[key] = {k: v for k, v in sample.items() if k != 'mask'}
            samples[key].update(profile=profile, diagnostics=diagnostics)
            if len(samples) % 100 == 0:
                print(f'Preflight geometry: {len(samples)} instances', flush=True)
    print(f'Preflight passed: {len(samples)} instances, '
          f'{len({s["group_id"] for s in samples.values()})} identities; '
          'all 5 splits identity-safe.', flush=True)
    if args.torso_mode == 'pca_peak':
        fallbacks = Counter(s['diagnostics']['fallback_reason'] for s in samples.values()
                            if s['diagnostics']['fallback'])
        print(f'PCA torso: {len(samples)-sum(fallbacks.values())} extracted / '
              f'{sum(fallbacks.values())} full-mask fallbacks; reasons={dict(fallbacks)}', flush=True)
    if args.smoke_test and args.experiment == 'morph_m1' and args.torso_mode == 'legacy':
        print(f'morph_m1 vs torso_lw: {len(samples)} instances; maximum absolute feature difference={equivalence_max}')
    return folds, samples


def pick_diagnostics(samples, count):
    folders = defaultdict(list)
    for key in sorted(samples):
        folders[Path(key[0]).parent.as_posix()].append(key)
    # Round-robin across source folders for representative diagnostics.
    chosen = []
    for index in range(max(map(len, folders.values()), default=0)):
        for keys in folders.values():
            if index < len(keys) and len(chosen) < count:
                chosen.append(keys[index])
    return chosen


def inspect_samples(args, samples, count):
    for key in pick_diagnostics(samples, count):
        sample = samples[key]
        print(json.dumps({k: sample[k] for k in ('annotation', 'shape_index', 'group_id', 'weight')}
                         | sample['diagnostics'], indent=2), flush=True)
        if args.save_debug and args.experiment != 'bbox':
            # Re-read only explicitly selected masks for optional local overlays.
            from_dataset = PastureWeighRGBDataset(args.split_dir/'fold_1.json', 'train', args.annotations_root)
            from_dataset.references = [dict(annotation=sample['annotation'], shape_index=sample['shape_index'],
                                           group_id=sample['group_id'], weight=sample['weight'])]
            original = from_dataset[0]
            if args.torso_mode == 'pca_peak':
                torso, _, _ = extract_torso_pca_peak(original['mask'])
                image = cv2.imread(str(original['image_path']))
                check(image is not None, f'Cannot read debug image {original["image_path"]}')
                contours, _ = cv2.findContours(torso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (255, 255, 0), 2)
                directory = args.output_dir/'debug'
                directory.mkdir(parents=True, exist_ok=True)
                name = f'{Path(key[0]).parent.name}_{Path(key[0]).stem}_shape{key[1]}.jpg'
                check(cv2.imwrite(str(directory/name), image), 'Could not save PCA debug image')
                continue
            debug_segments = 4 if args.experiment.startswith('morph_') else args.segments
            _, _, debug = extract_profile(original['mask'], debug_segments, args.samples_per_segment,
                                           args.tau, args.median_window, return_debug=True,
                                           profile_region=args.profile_region)
            image = cv2.imread(str(original['image_path']))
            check(image is not None, f'Cannot read debug image {original["image_path"]}')
            for label, color in [('core', (255, 0, 0)), ('expanded_ellipse', (0, 255, 255)),
                                 ('torso', (0, 255, 0))]:
                contours, _ = cv2.findContours(debug[label], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                shifted = [c + debug['origin_xy'].reshape(1, 1, 2) for c in contours]
                cv2.drawContours(image, shifted, -1, color, 2)
            directory = args.output_dir/'debug'
            directory.mkdir(parents=True, exist_ok=True)
            name = f'{Path(key[0]).parent.name}_{Path(key[0]).stem}_shape{key[1]}.jpg'
            check(cv2.imwrite(str(directory/name), image), 'Could not save debug image')


def inner_partition(fold, seed):
    ids = sort_ids(fold['train_group_ids'])
    random.Random(seed + fold['fold']).shuffle(ids)
    n_val = max(1, round(len(ids)*0.2))
    check(n_val < len(ids), 'Not enough outer training IDs for inner validation')
    val_ids = set(ids[:n_val])
    train = [r for r in fold['train'] if r['group_id'] not in val_ids]
    val = [r for r in fold['train'] if r['group_id'] in val_ids]
    check(bool(train) and bool(val), 'Empty inner partition')
    return train, val


def normalisation(refs, samples, experiment=None):
    widths = np.stack([samples[reference_key(ref)]['profile'] for ref in refs]).astype(np.float64)
    if experiment is not None and experiment.startswith('morph_'):
        segments = widths.shape[1]-1
        if segments == 1:
            # Preserve the exact original reduction order for strict M=1 equivalence.
            norm = normalisation(refs, samples, 'torso_lw')
        else:
            mean_length, raw_length = float(widths[:, 0].mean()), float(widths[:, 0].std(ddof=0))
            mean_width, raw_width = float(widths[:, 1:].mean()), float(widths[:, 1:].std(ddof=0))
            norm = {'mean_length': mean_length, 'std_length': max(raw_length, 1e-8),
                    'mean_width': mean_width, 'std_width': max(raw_width, 1e-8),
                    'raw_std_length': raw_length, 'raw_std_width': raw_width, 'epsilon': 1e-8}
        return norm | {'num_segments': segments,
                       'features': ['torso_length_px']+[f'torso_width_{i+1}_px' for i in range(segments)]}
    if experiment == 'torso_lw':
        mean = widths.mean(axis=0)
        raw_std = widths.std(axis=0, ddof=0)
        std = np.maximum(raw_std, 1e-8)
        return {'mean_length': float(mean[0]), 'std_length': float(std[0]),
                'mean_width': float(mean[1]), 'std_width': float(std[1]),
                'raw_std_length': float(raw_std[0]), 'raw_std_width': float(raw_std[1]),
                'epsilon': 1e-8, 'features': ['torso_length_px', 'torso_width_px']}
    if widths.ndim == 2:  # bbox: independent width/height training statistics
        mean = widths.mean(axis=0)
        raw_std = widths.std(axis=0, ddof=0)
        return {'mu_W': mean.tolist(), 'sigma_W': np.maximum(raw_std, 1e-8).tolist(),
                'raw_sigma_W': raw_std.tolist(), 'epsilon': 1e-8,
                'features': ['bbox_width', 'bbox_height']}
    mean = float(widths.mean())
    raw_std = float(widths.std(ddof=0))
    return {'mu_W': mean, 'sigma_W': max(raw_std, 1e-8), 'raw_sigma_W': raw_std, 'epsilon': 1e-8}


def normalisation_arrays(norm):
    if 'mean_length' in norm:
        return (np.array([norm['mean_length']]+[norm['mean_width']]*norm.get('num_segments', 1)),
                np.array([norm['std_length']]+[norm['std_width']]*norm.get('num_segments', 1)))
    return norm['mu_W'], norm['sigma_W']


def loader(refs, samples, norm, args, training, seed):
    widths = np.stack([samples[reference_key(ref)]['profile'] for ref in refs])
    mean, std = normalisation_arrays(norm)
    widths = ((widths - mean)/std).astype(np.float32)
    targets = np.log(np.array([ref['weight'] for ref in refs], dtype=np.float32))
    dataset = TensorDataset(torch.from_numpy(widths), torch.from_numpy(targets))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def new_model(args, seed):
    seed_everything(seed)
    if args.experiment.startswith('morph_'):
        return MorphometryRegressor(args.segments).to(args.device)
    if args.experiment in ('bbox', 'torso_lw'):
        return BBoxRegressor().to(args.device)
    return PastureWeighRGBRegressor(args.segments, args.samples_per_segment, args.dropout).to(args.device)


def train_epoch(model, batches, optimizer, device):
    model.train()
    total = 0.0
    for widths, targets in batches:
        widths, targets = widths.to(device), targets.to(device)
        if isinstance(model, MorphometryRegressor):
            if model.segments > 1:
                reverse = torch.rand(len(widths), device=device) < 0.5
                widths = torch.where(reverse[:, None], reverse_morphometry(widths), widths)
        elif not isinstance(model, BBoxRegressor):
            reverse = torch.rand(len(widths), device=device) < 0.5
            widths = torch.where(reverse[:, None, None], reverse_profile(widths), widths)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.smooth_l1_loss(model(widths), targets)
        check(torch.isfinite(loss).item(), 'Non-finite training loss')
        loss.backward()
        check(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
              'Non-finite training gradients')
        optimizer.step()
        total += loss.item()*len(widths)
    return total/len(batches.dataset)


@torch.no_grad()
def validation_loss(model, batches, device):
    model.eval()
    total = 0.0
    for widths, targets in batches:
        widths, targets = widths.to(device), targets.to(device)
        # Select epochs by the symmetric log-Huber objective, never outer test metrics.
        loss = nn.functional.smooth_l1_loss(model(widths), targets, reduction='sum')
        if isinstance(model, MorphometryRegressor):
            if model.segments > 1:
                loss = 0.5*(loss + nn.functional.smooth_l1_loss(
                    model(reverse_morphometry(widths)), targets, reduction='sum'))
        elif not isinstance(model, BBoxRegressor):
            loss = 0.5*(loss + nn.functional.smooth_l1_loss(
                model(reverse_profile(widths)), targets, reduction='sum'))
        check(torch.isfinite(loss).item(), 'Non-finite inner validation loss')
        total += loss.item()
    return total/len(batches.dataset)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def errors(gt, prediction):
    absolute = abs(prediction-gt)
    return {'absolute_error': absolute, 'absolute_percentage_error': 100*absolute/gt}


def calculate_metrics(rows, prefix):
    delta = np.array([r['predicted_weight']-r['gt_weight'] for r in rows], dtype=np.float64)
    gt = np.array([r['gt_weight'] for r in rows], dtype=np.float64)
    return {f'{prefix}_MAE_kg': float(np.abs(delta).mean()),
            f'{prefix}_RMSE_kg': float(np.sqrt(np.square(delta).mean())),
            f'{prefix}_MAPE_percent': float((100*np.abs(delta)/gt).mean())}


def aggregate_herds(observations, test_ids):
    """Sum only held-out observations per annotation; never cattle medians."""
    test_ids = set(test_ids)
    groups, seen = defaultdict(list), set()
    for row in observations:
        if row['group_id'] not in test_ids:
            continue
        key = reference_key(row)
        check(key not in seen, f'Duplicate test observation in herd aggregation: {key}')
        seen.add(key)
        groups[row['annotation']].append(row)
    result = []
    for annotation, rows in sorted(groups.items()):
        gt = sum(r['gt_weight'] for r in rows)
        predicted = sum(r['predicted_weight'] for r in rows)
        check(gt > 0 and np.isfinite([gt, predicted]).all(), 'Invalid herd weights')
        result.append({'annotation': annotation, 'num_test_cattle': len(rows),
                       'gt_herd_weight': gt, 'predicted_herd_weight': predicted,
                       **errors(gt, predicted)})
    return result


def herd_smoke_test(fold):
    # Deliberately inject extreme training-ID predictions from the same real
    # mixed-identity images: they must have zero effect on the herd rows/metrics.
    test_rows = [{**r, 'gt_weight': float(r['weight']), 'predicted_weight': float(r['weight'])+10}
                 for r in fold['val']]
    train_rows = [{**r, 'gt_weight': 1e12, 'predicted_weight': 2e12} for r in fold['train']]
    expected = aggregate_herds(test_rows, fold['val_group_ids'])
    actual = aggregate_herds(test_rows+train_rows, fold['val_group_ids'])
    check(actual == expected, 'Training cattle contaminated herd sums')
    test_images = {r['annotation'] for r in fold['val']}
    mixed = test_images & {r['annotation'] for r in fold['train']}
    check(len(actual) == len(test_images), 'Missing test image in herd aggregation')
    for row in actual:
        refs = [r for r in fold['val'] if r['annotation'] == row['annotation']]
        check(row['num_test_cattle'] == len(refs) and
              row['gt_herd_weight'] == sum(r['weight'] for r in refs) and
              row['absolute_error'] == 10*len(refs), 'Incorrect held-out herd sum')
    print(f'Herd smoke: {len(actual)} test images ({len(mixed)} mixed train/test images); '
          f'{len(test_rows)} test instances included, {len(train_rows)} training instances excluded; passed.')


@torch.no_grad()
def evaluate_test(model, refs, samples, norm, args, test_ids):
    model.eval()
    check({ref['group_id'] for ref in refs} <= set(test_ids), 'Non-test identity in evaluation refs')
    predictions = []
    for widths, _ in loader(refs, samples, norm, args, False, args.seed):
        weights = predict_weight(model, widths.to(args.device)).cpu().numpy()
        check(np.isfinite(weights).all() and (weights > 0).all(), 'Invalid predicted weight')
        predictions.extend(weights.tolist())
    rows, grouped = [], defaultdict(list)
    for ref, prediction in zip(refs, predictions):
        gt = float(ref['weight'])
        row = {k: ref[k] for k in ('annotation', 'shape_index', 'group_id')}
        row.update(gt_weight=gt, predicted_weight=prediction, **errors(gt, prediction))
        rows.append(row)
        grouped[ref['group_id']].append(row)
    cattle_rows = []
    for gid in sort_ids(grouped):
        observations = grouped[gid]
        check(len({r['gt_weight'] for r in observations}) == 1, f'Inconsistent GT for {gid}')
        gt = observations[0]['gt_weight']
        prediction = statistics.median(r['predicted_weight'] for r in observations)
        cattle_rows.append({'group_id': gid, 'num_observations': len(observations),
                            'gt_weight': gt, 'predicted_weight': prediction, **errors(gt, prediction)})
    herd_rows = aggregate_herds(rows, test_ids)
    herd_metric_rows = [{'gt_weight': r['gt_herd_weight'], 'predicted_weight': r['predicted_herd_weight']}
                        for r in herd_rows]
    metrics = (calculate_metrics(cattle_rows, 'cattle') | calculate_metrics(herd_metric_rows, 'herd') |
               calculate_metrics(rows, 'observation'))
    return rows, cattle_rows, herd_rows, metrics


def print_metrics(metrics, prefix=''):
    for label, key in [('CATTLE-LEVEL', 'cattle'), ('HERD-LEVEL', 'herd'), ('OBSERVATION-LEVEL', 'observation')]:
        print(f'{prefix}{label}: MAE={metrics[key+"_MAE_kg"]:.3f} kg; '
              f'RMSE={metrics[key+"_RMSE_kg"]:.3f} kg; '
              f'MAPE={metrics[key+"_MAPE_percent"]:.3f}%', flush=True)


def run_fold(fold, samples, args):
    number = fold['fold']
    directory = args.output_dir/f'fold_{number}'
    check(not directory.exists() or not any(directory.iterdir()),
          f'Output already exists: {directory}; choose a new --output-dir')
    inner_train, inner_val = inner_partition(fold, args.seed)
    inner_norm = normalisation(inner_train, samples, args.experiment)
    inner_seed = args.seed + number  # outer fold index is 1-based
    model = new_model(args, inner_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_batches = loader(inner_train, samples, inner_norm, args, True, inner_seed)
    val_batches = loader(inner_val, samples, inner_norm, args, False, inner_seed)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    history = []
    print(f'Fold {number}: epoch selection with {len(inner_train)} inner-train / '
          f'{len(inner_val)} inner-val observations', flush=True)
    for epoch in range(1, args.max_epochs+1):
        training = train_epoch(model, train_batches, optimizer, args.device)
        validation = validation_loss(model, val_batches, args.device)
        history.append({'epoch': epoch, 'train_log_huber': training, 'inner_val_log_huber': validation})
        if validation < best_loss:
            best_loss, best_epoch, stale = validation, epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'Fold {number}, selection epoch {epoch}: inner-val loss={validation:.6f}', flush=True)
        if stale >= args.patience:
            break

    # No selected-model weights are reused. Refit from a fresh initialisation on
    # every outer training identity, with statistics recomputed only from them.
    final_seed = args.seed + 1000 + number
    del optimizer, model
    final_norm = normalisation(fold['train'], samples, args.experiment)
    model = new_model(args, final_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_batches = loader(fold['train'], samples, final_norm, args, True, final_seed)
    print(f'Fold {number}: fresh outer refit for exactly {best_epoch} epochs', flush=True)
    for _ in range(best_epoch):
        train_epoch(model, final_batches, optimizer, args.device)

    directory.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(outer_fold=number, inner_seed=inner_seed, final_seed=final_seed,
                  inner_train_group_ids=sort_ids({r['group_id'] for r in inner_train}),
                  inner_val_group_ids=sort_ids({r['group_id'] for r in inner_val}),
                  outer_train_group_ids=fold['train_group_ids'], outer_test_group_ids=fold['val_group_ids'],
                  best_epoch=best_epoch, best_inner_val_log_huber=best_loss,
                  inner_normalisation=inner_norm, selection_history=history,
                  inner_validation_objective=('log SmoothL1 loss' if args.experiment in ('bbox', 'torso_lw', 'morph_m1')
                                              else 'mean forward/reverse log SmoothL1 loss'))
    write_json(directory/'config.json', config)
    write_json(directory/'normalisation.json', final_norm)
    torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'experiment': args.experiment, 'profile_region': args.profile_region,
                'M': args.segments, 'K': args.samples_per_segment, 'dropout': args.dropout,
                **({key: final_norm[key] for key in ('mean_length', 'std_length', 'mean_width', 'std_width')}
                   if args.experiment == 'torso_lw' or args.experiment.startswith('morph_') else
                   {'mu_W': final_norm['mu_W'], 'sigma_W': final_norm['sigma_W']}),
                **({'num_segments': args.segments} if args.experiment.startswith('morph_') else {}),
                'best_epoch': best_epoch, 'outer_fold': number, 'seed': args.seed,
                'final_seed': final_seed, 'tau': args.tau, 'median_window': args.median_window,
                'torso_mode': args.torso_mode, 'torso_config': args.torso_config},
               directory/'model.pt')
    # The outer held-out observations are accessed for prediction only now.
    rows, cattle_rows, herd_rows, metrics = evaluate_test(
        model, fold['val'], samples, final_norm, args, fold['val_group_ids'])
    write_csv(directory/'observation_predictions.csv', rows)
    write_csv(directory/'cattle_predictions.csv', cattle_rows)
    write_csv(directory/'herd_predictions.csv', herd_rows)
    metrics.update(fold=number, best_epoch=best_epoch, num_cattle=len(cattle_rows),
                   num_observations=len(rows), num_test_images=len(herd_rows))
    # Keep legacy flat metric keys for old consumers, plus explicit evaluation levels.
    structured = {level: {'MAE': metrics[level+'_MAE_kg'], 'RMSE': metrics[level+'_RMSE_kg'],
                          'MAPE': metrics[level+'_MAPE_percent']}
                  for level in ('cattle', 'herd', 'observation')}
    write_json(directory/'metrics.json', metrics | structured)
    print_metrics(metrics, prefix=f'Fold {number} ')
    return metrics


def check_morph_smoke(args, samples, train_refs, norm, model, batch):
    features = np.stack([sample['profile'] for sample in samples.values()])
    check(features.shape == (len(samples), args.segments+1) and
          np.isfinite(features).all() and (features > 0).all(), 'Invalid morphometry features')
    check(batch.shape == (len(batch), args.segments+1), 'Incorrect morphometry batch shape')
    check(sum(p.numel() for p in model.parameters()) == 49+16*args.segments,
          'Incorrect morphometry parameter count')
    reversed_batch = reverse_morphometry(batch)
    check(torch.equal(reverse_morphometry(reversed_batch), batch), 'Reversal is not an involution')
    check(torch.equal(reversed_batch[:, 0], batch[:, 0]), 'Reversal changed length')
    check(torch.equal(reversed_batch[:, 1:], batch[:, 1:].flip(-1)), 'Width reversal failed')
    raw = np.stack([samples[reference_key(r)]['profile'] for r in train_refs]).astype(np.float64)
    check(np.isclose(norm['mean_length'], raw[:, 0].mean()) and
          np.isclose(norm['std_length'], max(raw[:, 0].std(), 1e-8)) and
          np.isclose(norm['mean_width'], raw[:, 1:].mean()) and
          np.isclose(norm['std_width'], max(raw[:, 1:].std(), 1e-8)), 'Invalid pooled training statistics')
    mean, std = normalisation_arrays(norm)
    reversed_raw = np.concatenate([raw[:, :1], raw[:, 1:][:, ::-1]], axis=1)
    normed = (raw-mean)/std
    check(np.array_equal((reversed_raw-mean)/std,
                         np.concatenate([normed[:, :1], normed[:, 1:][:, ::-1]], axis=1)),
          'Standardisation does not commute with width reversal')
    if args.segments == 1:
        lw_norm = normalisation(train_refs, samples, 'torso_lw')
        for key in ('mean_length', 'std_length', 'mean_width', 'std_width'):
            check(norm[key] == lw_norm[key], 'M=1 normalisation differs from torso_lw')
    print(f'{args.experiment}: all {len(samples)} positive finite; batch={list(batch.shape)}; '
          f'L range=[{features[:, 0].min():.6f}, {features[:, 0].max():.6f}]; '
          f'W range=[{features[:, 1:].min():.6f}, {features[:, 1:].max():.6f}]')
    print('Width reversal, fixed L, involution and training-only symmetric standardisation: passed')


def smoke_test(args, folds, samples):
    inner_train, inner_val = inner_partition(folds[0], args.seed)
    check(not {r['group_id'] for r in inner_train} & {r['group_id'] for r in inner_val},
          'Smoke test: inner identity overlap')
    norm = normalisation(inner_train, samples, args.experiment)
    model = new_model(args, args.seed)
    batch, targets = next(iter(loader(inner_train, samples, norm, args, True, args.seed)))
    batch, targets = batch.to(args.device), targets.to(args.device)
    if args.experiment == 'torso_lw':
        check(batch.shape == (len(batch), 2), 'Expected torso_lw input [B, 2]')
        check(sum(p.numel() for p in model.parameters() if p.requires_grad) == 65,
              'torso_lw must use the unchanged 65-parameter MLP')
        all_features = np.stack([s['profile'] for s in samples.values()])
        check(np.isfinite(all_features).all() and (all_features > 0).all(),
              'Invalid torso length/width')
        print(f'Torso length min/max (px): {all_features[:, 0].min():.6f} / {all_features[:, 0].max():.6f}')
        print(f'Torso width min/max (px): {all_features[:, 1].min():.6f} / {all_features[:, 1].max():.6f}')
        print(f'Input batch shape: {list(batch.shape)}')
        print('Inner-train normalisation: ' + json.dumps(norm))
    if args.experiment.startswith('morph_'):
        check_morph_smoke(args, samples, inner_train, norm, model, batch)
    elif args.experiment not in ('bbox', 'torso_lw'):
        check(torch.equal(reverse_profile(batch).flatten(1), batch.flatten(1).flip(1)),
              'Full sequence reversal failed')
    else:
        raw = np.stack([samples[reference_key(r)]['profile'] for r in inner_train]).astype(np.float64)
        mean, std = normalisation_arrays(norm)
        check(np.allclose(mean, raw.mean(axis=0)) and
              np.allclose(std, np.maximum(raw.std(axis=0), 1e-8)), 'Invalid per-feature normalisation')
    output = model(batch)
    check(output.shape == (len(batch),), 'Incorrect real-batch model output shape')
    input_shape = (2,) if args.experiment in ('bbox', 'torso_lw') else (args.segments, args.samples_per_segment)
    if args.experiment.startswith('morph_'):
        input_shape = (args.segments+1,)
    random_output = model(torch.randn(2, *input_shape, device=args.device))
    check(random_output.shape == (2,), 'Incorrect random-batch model output shape')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.smooth_l1_loss(output, targets)
    check(torch.isfinite(loss).item(), 'Non-finite smoke loss')
    loss.backward()
    check(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
          'Smoke backward produced missing/non-finite gradients')
    optimizer.step()
    check(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), 'Optimizer did not update weights')
    model.eval()
    with torch.no_grad():
        prediction = predict_weight(model, batch)
        check(torch.isfinite(prediction).all().item() and (prediction > 0).all().item(),
              'Smoke inference produced invalid kg predictions')
    folders = defaultdict(list)
    for sample in samples.values():
        folders[Path(sample['annotation']).parent.name].append(sample['profile'])
    for folder, profiles in sorted(folders.items()):
        values = np.stack(profiles)
        print(f'{folder}: {len(profiles)} instances; profile shape={list(values.shape[1:])}; '
              f'pixel width range=[{values.min():.6f}, {values.max():.6f}]; mean={values.mean():.6f}; finite=True')
    print(f'Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Forward real batch: {list(output.shape)}; random batch: {list(random_output.shape)}')
    print(f'Backward + one AdamW step: passed; log-Huber loss={loss.item():.6f}')
    print('Positive kg inference: passed; direction handling matches experiment')
    herd_smoke_test(folds[0])
    print('SMOKE TEST PASSED. No CV training started; no model or metrics files written.')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=[*EXPERIMENTS, 'all'], default=None)
    parser.add_argument('--fold', default='all', choices=['all', '1', '2', '3', '4', '5'])
    parser.add_argument('--segments', type=int, choices=[1, 4, 8, 16], default=4)
    parser.add_argument('--samples-per-segment', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tau', type=float, default=0.25)
    parser.add_argument('--torso-mode', choices=['legacy', 'pca_peak'], default='legacy',
                        help='Explicit torso alternative for morphometry; default preserves legacy geometry')
    parser.add_argument('--median-window', type=int, default=5)
    parser.add_argument('--annotations-root', type=Path, default=ANNOTATIONS)
    parser.add_argument('--split-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--inspect-samples', type=int, default=0, help='Diagnostics only, unless also --smoke-test')
    parser.add_argument('--save-debug', action='store_true', help='Explicitly save a few diagnostic overlays')
    args = parser.parse_args()
    if args.torso_mode == 'pca_peak' and args.experiment not in ('morph_m1', 'morph_m2', 'morph_m4', 'morph_m8'):
        parser.error('--torso-mode pca_peak requires an explicit morph_m* experiment')
    args.torso_config = ({'method': 'pca_peak', 'peak_ratio': .68, 'min_length_ratio': .52,
                          'smooth_window': 7, 'smoothing': 'normalized_box_kernel', 'closing_kernel': 3,
                          'fallback': 'full_cleaned_mask', 'features': 'PCA-coordinate pixel spans'}
                         if args.torso_mode == 'pca_peak' else {'method': 'legacy', 'tau': args.tau})
    args.split_dir = args.split_dir or args.annotations_root/'split'
    if (min(args.batch_size, args.max_epochs, args.patience) < 1 or args.samples_per_segment < 2 or
            args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or
            not 0 < args.tau < 1 or args.median_window < 1 or args.median_window % 2 != 1 or
            args.inspect_samples < 0 or not 0 <= args.seed < 2**32-1006):
        parser.error('Invalid optimisation, profile or seed parameters')
    return args


def run_experiment(args):
    seed_everything(args.seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    print(f'Device: {args.device}; M={args.segments}, K={args.samples_per_segment}', flush=True)
    parameter_count = sum(p.numel() for p in new_model(args, args.seed).parameters() if p.requires_grad)
    print('Final config: ' + json.dumps({k: str(v) if isinstance(v, Path) else v
                                        for k, v in vars(args).items()}, sort_keys=True), flush=True)
    print(f'Total trainable parameter count: {parameter_count}', flush=True)
    folds, samples = prepare_geometry(args)
    if args.inspect_samples or args.save_debug:
        inspect_samples(args, samples, args.inspect_samples or 5)
    if args.smoke_test:
        smoke_test(args, folds, samples)
        return
    if args.inspect_samples or args.save_debug:
        print('Inspection complete; no training started.')
        return
    selected = folds if args.fold == 'all' else [folds[int(args.fold)-1]]
    for fold in selected:
        directory = args.output_dir/f'fold_{fold["fold"]}'
        check(not directory.exists() or not any(directory.iterdir()),
              f'Output already exists: {directory}; choose a new --output-dir')
    if args.torso_mode == 'pca_peak':
        diagnostic_path = args.output_dir/'geometry_diagnostics.json'
        check(not diagnostic_path.exists(), f'Refusing to overwrite {diagnostic_path}')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(diagnostic_path, {'torso_config': args.torso_config,
                   'num_instances': len(samples),
                   'fallback_count': sum(s['diagnostics']['fallback'] for s in samples.values()),
                   'instances': [{'annotation': k[0], 'shape_index': k[1], 'group_id': s['group_id'],
                                  'features': s['profile'].tolist(), **s['diagnostics']}
                                 for k, s in sorted(samples.items())]})
    results = [run_fold(fold, samples, args) for fold in selected]
    if args.fold == 'all':
        aggregates = {name: {'mean': float(np.mean([r[name] for r in results])),
                              'std': float(np.std([r[name] for r in results], ddof=1))}
                      for name in METRICS}
        write_json(args.output_dir/'summary.json', {'folds': results, 'aggregates': aggregates,
                   'std_ddof': 1, 'num_folds': 5, 'seed': args.seed, 'experiment': args.experiment,
                   'torso_mode': args.torso_mode, 'torso_config': args.torso_config,
                   'herd_aggregation': 'sum observation predictions for outer-test identities per annotation',
                   'cattle_aggregation': 'median of observation predictions',
                   'direction_aggregation': ('exp(log-weight)' if args.experiment in ('bbox', 'torso_lw', 'morph_m1') else
                                             'arithmetic mean of exp(forward) and exp(reverse)')})
        write_csv(args.output_dir/'fold_metrics.csv', results)
        print('='*60 + f'\nEXPERIMENT: {args.experiment}\n' + '='*60)
        for level in ('cattle', 'herd', 'observation'):
            print(f'{level.upper()}-LEVEL five-fold mean ± std (sample std, ddof=1):')
            for name in METRICS:
                if name.startswith(level):
                    print(f'  {name}: {aggregates[name]["mean"]:.3f} ± {aggregates[name]["std"]:.3f}')


def main():
    requested = parse_args()
    names = (EXPERIMENTS if requested.experiment == 'all' else
             [requested.experiment or f'torso_m{requested.segments}'])
    for name in names:
        args = copy.copy(requested)
        args.experiment = name
        args.segments = int(name.rsplit('_m', 1)[1]) if name not in ('bbox', 'torso_lw') else 4
        args.samples_per_segment = 32
        args.profile_region = 'fullmask' if name == 'fullmask_m4' else 'torso'
        if name == 'bbox':
            args.profile_region = None
        if args.segments != requested.segments or requested.samples_per_segment != 32:
            print(f'Experiment {name} sets M={args.segments}, K=32; experiment configuration takes precedence.')
        if requested.output_dir is None:
            suffix = '_pca_peak' if args.torso_mode == 'pca_peak' else ''
            args.output_dir = Path('outputs')/f'{name}{suffix}_seed{args.seed}'
        elif requested.experiment == 'all':
            args.output_dir = requested.output_dir/f'{name}_seed{args.seed}'
        run_experiment(args)


if __name__ == '__main__':
    main()
