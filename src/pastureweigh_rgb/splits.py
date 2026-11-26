#!/usr/bin/env python3
"""Create deterministic identity-safe, instance-level cattle CV references."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys


SEED = 42
NUM_FOLDS = 5


def sorted_ids(ids):
    # Preserve integer/string identities without silently merging their types.
    return sorted(ids, key=lambda value: (0, value) if isinstance(value, int) else (1, value))


def collect_instances(annotations_root, source):
    annotations_root, source = Path(annotations_root).resolve(), Path(source).resolve()
    if not source.is_relative_to(annotations_root):
        raise ValueError('Source must be inside the annotations root')
    if not source.is_dir():
        raise ValueError(f'Missing annotation directory: {source}')
    instances = []
    weights = {}
    origins = {}
    paths = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() == '.json')
    for path in paths:
        try:
            with path.open(encoding='utf-8-sig') as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or not isinstance(data.get('shapes'), list):
                raise ValueError('expected a Labelme object with a shapes list')
            for index, shape in enumerate(data['shapes']):
                if not isinstance(shape, dict):
                    raise ValueError(f'shapes[{index}] must be an object')
                label = shape.get('label')
                if not isinstance(label, str) or label.strip() != 'cattle':
                    continue
                group_id = shape.get('group_id')
                weight = shape.get('weight')
                location = f'{path}: shapes[{index}]'
                if group_id is None:
                    raise ValueError(f'shapes[{index}]: missing group_id')
                if type(group_id) not in (int, str):
                    raise ValueError(f'shapes[{index}]: group_id must be an integer or string')
                if weight is None:
                    raise ValueError(f'shapes[{index}]: missing weight')
                if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f'shapes[{index}]: weight must be positive and finite')
                if group_id in weights and weights[group_id] != weight:
                    raise ValueError(f'shapes[{index}]: group_id {group_id!r} has weight {weight!r}; '
                                     f'previously {weights[group_id]!r} at {origins[group_id]}')
                weights[group_id] = weight
                origins.setdefault(group_id, location)
                instances.append({
                    'annotation': path.relative_to(annotations_root).as_posix(),
                    'shape_index': index,
                    'group_id': group_id,
                    'weight': weight,
                })
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f'{path}: {exc}') from exc
    if len(weights) < NUM_FOLDS:
        raise ValueError(f'Need at least {NUM_FOLDS} cattle identities; found {len(weights)}')
    return instances, sorted_ids(weights)


def make_folds(instances, all_ids, seed=SEED):
    shuffled = list(all_ids)
    random.Random(seed).shuffle(shuffled)
    quotient, remainder = divmod(len(shuffled), NUM_FOLDS)
    folds = []
    offset = 0
    for index in range(NUM_FOLDS):
        size = quotient + int(index < remainder)
        val_ids = set(shuffled[offset:offset + size])
        offset += size
        folds.append({
            'fold': index + 1,
            'seed': seed,
            'train_group_ids': sorted_ids(set(all_ids) - val_ids),
            'val_group_ids': sorted_ids(val_ids),
            'train': [item for item in instances if item['group_id'] not in val_ids],
            'val': [item for item in instances if item['group_id'] in val_ids],
        })
    return folds


def require(condition, message):
    if not condition:
        raise ValueError(f'Validation failed: {message}')


def validate(folds, instances, all_ids, seed=SEED):
    require(len(folds) == NUM_FOLDS, 'expected exactly 5 folds')
    def key(item):
        return item['annotation'], item['shape_index']

    source = {key(item): item for item in instances}
    require(len(source) == len(instances), 'duplicate source instance references')
    source_counts = Counter(key(item) for item in instances)
    val_id_counts = Counter()
    val_instance_counts = Counter()
    sizes = []
    for index, fold in enumerate(folds, 1):
        require(fold['fold'] == index and fold['seed'] == seed, 'incorrect fold metadata')
        train_ids, val_ids = set(fold['train_group_ids']), set(fold['val_group_ids'])
        require(len(train_ids) == len(fold['train_group_ids']) and
                len(val_ids) == len(fold['val_group_ids']), f'fold {index}: repeated IDs')
        require(not train_ids & val_ids, f'fold {index}: train/val identity overlap')
        require(train_ids | val_ids == set(all_ids), f'fold {index}: incomplete identity coverage')
        for partition, ids in [('train', train_ids), ('val', val_ids)]:
            for item in fold[partition]:
                require(item == source.get(key(item)), f'fold {index}: invalid reference or metadata')
                require(item['group_id'] in ids, f'fold {index}: instance in wrong identity partition')
        require(Counter(key(item) for item in fold['train'] + fold['val']) == source_counts,
                f'fold {index}: instances missing or duplicated')
        val_id_counts.update(val_ids)
        val_instance_counts.update(key(item) for item in fold['val'])
        sizes.append(len(val_ids))
    require(val_id_counts == Counter(all_ids), 'each ID must be validation in exactly one fold')
    require(val_instance_counts == source_counts, 'each instance must be validation exactly once')
    require(max(sizes) - min(sizes) <= 1, 'identity counts are not evenly distributed')


def make_summary(folds, instances, all_ids, seed=SEED):
    return {
        'seed': seed,
        'num_folds': NUM_FOLDS,
        'total_cattle_instances': len(instances),
        'total_unique_group_ids': len(all_ids),
        'all_group_ids': all_ids,
        'folds': [{
            'fold': fold['fold'],
            'val_group_ids': fold['val_group_ids'],
            'num_val_group_ids': len(fold['val_group_ids']),
            'num_train_group_ids': len(fold['train_group_ids']),
            'num_train_instances': len(fold['train']),
            'num_val_instances': len(fold['val']),
        } for fold in folds],
    }


def heading(title):
    print('=' * 60)
    print(title)
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations-root', type=Path, default=Path('data/annotations'))
    parser.add_argument('--source', type=Path, help='Labelme directory inside the annotations root; defaults to raw-frames')
    parser.add_argument('--output-dir', type=Path, help='Defaults to <annotations-root>/split')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    source = args.source or args.annotations_root/'raw-frames'
    output = args.output_dir or args.annotations_root/'split'
    if output.resolve().is_relative_to(source.resolve()):
        parser.error('--output-dir must be outside --source')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output already exists: {output}; choose an empty directory')
    # Complete preflight and in-memory validation before creating any outputs.
    instances, all_ids = collect_instances(args.annotations_root, source)
    folds = make_folds(instances, all_ids, args.seed)
    validate(folds, instances, all_ids, args.seed)
    summary = make_summary(folds, instances, all_ids, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        (output / f'fold_{fold["fold"]}.json').write_text(
            json.dumps(fold, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # Validate the actual saved references against freshly read annotations.
    saved_paths = sorted(output.glob('fold_*.json'))
    saved_folds = [json.loads(path.read_text(encoding='utf-8')) for path in saved_paths]
    current_instances, current_ids = collect_instances(args.annotations_root, source)
    require(current_instances == instances and current_ids == all_ids, 'source annotations changed')
    validate(saved_folds, current_instances, current_ids, args.seed)
    require(saved_folds == folds, 'saved folds differ from deterministic split')
    saved_summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    require(saved_summary == make_summary(saved_folds, current_instances, current_ids, args.seed),
            'saved summary does not match folds')

    heading('DATASET')
    print(f'Cattle instances: {len(instances)}')
    print(f'Unique cattle IDs: {len(all_ids)}')
    print(f'Group IDs: {all_ids}\n')
    for fold in saved_summary['folds']:
        heading(f'FOLD {fold["fold"]}')
        print(f'Train IDs: {fold["num_train_group_ids"]}')
        print(f'Val IDs: {fold["num_val_group_ids"]}')
        print(f'Train instances: {fold["num_train_instances"]}')
        print(f'Val instances: {fold["num_val_instances"]}')
        print(f'Val group IDs: {fold["val_group_ids"]}\n')
    heading('VALIDATION')
    print('Train/val identity overlap: 0 for all folds')
    print(f'Validation ID coverage: {len(all_ids)} / {len(all_ids)}')
    print(f'Validation instance coverage: {len(instances)} / {len(instances)}')
    print('All checks passed.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
