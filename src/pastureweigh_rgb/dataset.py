"""Read-only instance references and the actual Labelme cropped-PNG mask schema."""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


def decode_mask(shape, height, width):
    """Place the PNG crop at integer, inclusive xyxy corners, without resizing."""
    if shape.get('shape_type') != 'mask':
        raise ValueError(f'Unsupported shape_type: {shape.get("shape_type")!r}; expected mask')
    points = np.asarray(shape.get('points'), dtype=np.float64)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError('Mask requires two finite xy corners')
    if not np.array_equal(points, np.rint(points)):
        raise ValueError('Mask corners must be integer pixel coordinates')
    (x0, y0), (x1, y1) = points.astype(int)
    if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
        raise ValueError('Mask corners are outside the declared image dimensions')
    encoded = base64.b64decode(shape['mask'], validate=True)
    if not encoded.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Expected base64-encoded PNG mask')
    crop = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
    if crop is None or crop.ndim != 2 or crop.shape != (y1-y0+1, x1-x0+1):
        raise ValueError('Decoded PNG dimensions do not match inclusive mask corners')
    values = set(np.unique(crop).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f'Expected a binary PNG mask, found values {values}')
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1+1, x0:x1+1] = crop > 0
    if not mask.any():
        raise ValueError('Decoded instance mask is empty')
    return mask


class PastureWeighRGBDataset(Dataset):
    """Return a full-resolution binary mask and metadata, never an RGB tensor."""

    def __init__(self, fold_json, partition='train', annotations_root=None):
        self.fold_json = Path(fold_json).resolve()
        self.root = (Path(annotations_root).resolve() if annotations_root is not None
                     else self.fold_json.parent.parent)
        if partition not in ('train', 'val'):
            raise ValueError('partition must be train or val (outer held-out test)')
        self.references = json.loads(self.fold_json.read_text(encoding='utf-8'))[partition]
        self._documents = {}

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        ref = self.references[index]
        path = (self.root / ref['annotation']).resolve()
        location = f'{ref["annotation"]}: shapes[{ref["shape_index"]}]'
        try:
            if not path.is_relative_to(self.root):
                raise ValueError('Annotation reference escapes the annotations root')
            if path not in self._documents:
                self._documents[path] = json.loads(path.read_text(encoding='utf-8'))
            doc = self._documents[path]
            shapes = doc['shapes']
            shape_index = ref['shape_index']
            if type(shape_index) is not int or not 0 <= shape_index < len(shapes):
                raise ValueError('Invalid shape_index')
            shape = shapes[shape_index]
            if shape['label'].strip() != 'cattle':
                raise ValueError('Referenced shape is not cattle')
            if shape.get('group_id') is None or shape['group_id'] != ref['group_id']:
                raise ValueError('group_id does not match the split')
            weight = shape.get('weight')
            if (type(weight) not in (int, float) or not np.isfinite(weight) or
                    weight <= 0 or weight != ref['weight']):
                raise ValueError('weight must be positive, finite and match the split')
            height, width = doc['imageHeight'], doc['imageWidth']
            if type(height) is not int or type(width) is not int or min(height, width) <= 0:
                raise ValueError('Invalid image dimensions')
            image_path = (path.parent / doc['imagePath']).resolve()
            if not image_path.is_file():
                raise ValueError(f'Image does not exist: {image_path}')
            mask = decode_mask(shape, height, width)
            return {**ref, 'annotation_path': path, 'image_path': image_path,
                    'weight': float(weight), 'mask': mask}
        except Exception as exc:
            raise ValueError(f'{location}: {exc}') from exc
