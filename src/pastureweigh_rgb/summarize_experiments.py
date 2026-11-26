#!/usr/bin/env python3
"""Read completed experiment summaries; write a comparison without ranking."""

import argparse
import csv
import json
import math
from pathlib import Path


METHOD_NAMES = {
    'bbox': 'BBox', 'torso_lw': 'Torso L/W',
    'morph_m1': 'Morphometry (M=1)', 'morph_m2': 'Morphometry (M=2)',
    'morph_m4': 'Morphometry (M=4)', 'morph_m8': 'Morphometry (M=8)',
    'fullmask_m4': 'Full Mask (M=4)',
    'torso_m1': 'Torso Profile (M=1)', 'torso_m4': 'Torso Profile (M=4)',
    'torso_m8': 'Torso Profile (M=8)', 'torso_m16': 'Torso Profile (M=16)',
}
EXPERIMENTS = tuple(METHOD_NAMES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-root', type=Path, default=Path('outputs'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    columns = ['experiment', 'cattle_MAE', 'cattle_RMSE', 'cattle_MAPE',
               'herd_MAE', 'herd_RMSE', 'herd_MAPE']
    rows = []
    for experiment in EXPERIMENTS:
        row = dict.fromkeys(columns, 'missing')
        row['experiment'] = experiment
        path = args.outputs_root/f'{experiment}_seed{args.seed}'/'summary.json'
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if (data['num_folds'] != 5 or len(data['folds']) != 5 or
                    {fold['fold'] for fold in data['folds']} != {1, 2, 3, 4, 5} or
                    data['experiment'] != experiment or data['seed'] != args.seed):
                raise ValueError('Summary is not a completed matching five-fold experiment')
            values = {}
            for column in columns[1:]:
                key = column + ('_percent' if column.endswith('MAPE') else '_kg')
                aggregate = data['aggregates'][key]
                mean, std = float(aggregate['mean']), float(aggregate['std'])
                if not math.isfinite(mean) or not math.isfinite(std):
                    raise ValueError(f'Non-finite metric: {key}')
                values[column] = f'{mean:.3f} ± {std:.3f}'
            row.update(values)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'{experiment}: missing (incomplete/unreadable summary: {exc})')
        rows.append(row)
    args.outputs_root.mkdir(parents=True, exist_ok=True)
    output = args.outputs_root/'ablation_summary.csv'
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{"Experiment":24} {"Cattle MAPE (%)":24} {"Herd MAPE (%)":24}')
    for row in rows:
        print(f'{METHOD_NAMES[row["experiment"]]:24} {row["cattle_MAPE"]:24} {row["herd_MAPE"]:24}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
