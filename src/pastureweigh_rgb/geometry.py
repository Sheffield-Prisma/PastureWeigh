"""Ellipse-guided torso widths in original pixel units; no resizing or scale features."""

import cv2
import numpy as np


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError('No foreground connected component')
    component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == component).astype(np.uint8)


def clean_mask(mask):
    """Fill holes, retain largest component, return full mask and exclusive xyxy bbox."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError('Expected a finite H x W mask')
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        raise ValueError('Empty input mask')
    padded = np.pad(binary, 1)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 1)
    filled = np.logical_or(padded, flooded == 0)[1:-1, 1:-1].astype(np.uint8)
    cleaned = largest_component(filled)
    ys, xs = np.nonzero(cleaned)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
    return cleaned, bbox


def extract_profile(mask, segments=4, samples_per_segment=32, tau=0.25,
                    median_window=5, return_debug=False, profile_region='torso',
                    torso_dimensions=False, morph_segments=None):
    if morph_segments is not None and morph_segments not in (1, 2, 4, 8):
        raise ValueError('Morphometry segments must be 1/2/4/8')
    if profile_region not in ('torso', 'fullmask'):
        raise ValueError('profile_region must be torso or fullmask')
    if segments not in (1, 4, 8, 16) or samples_per_segment < 2:
        raise ValueError('segments must be 1/4/8/16 and samples_per_segment >= 2')
    if not 0 < tau < 1 or median_window < 1 or median_window % 2 != 1:
        raise ValueError('Require 0 < tau < 1 and positive odd median_window')
    cleaned, bbox = clean_mask(mask)
    x0, y0, x1, y1 = bbox
    # One explicit background pixel on every edge preserves boundary distances,
    # including for masks touching the image edge. No pixel-scale change.
    local = np.pad(cleaned[y0:y1, x0:x1], 1)
    origin = np.array([x0-1, y0-1], dtype=np.float64)
    distance = cv2.distanceTransform(local, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dmax = float(distance.max())
    if not np.isfinite(dmax) or dmax <= 0:
        raise ValueError('Invalid distance-transform maximum')
    radius = tau * dmax
    core = largest_component((distance >= radius).astype(np.uint8))
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5 or cv2.contourArea(contour) <= 0:
        raise ValueError(f'Core cannot support ellipse fitting: area={int(core.sum())}, Dmax={dmax}')
    center, axes, angle = cv2.fitEllipse(contour)
    # OpenCV angle describes its first returned diameter. Rotate by 90 degrees
    # if the second diameter is the major axis. All calculations below use xy.
    if axes[0] >= axes[1]:
        major, minor, theta_deg = axes[0], axes[1], angle
    else:
        major, minor, theta_deg = axes[1], axes[0], angle + 90.0
    theta = np.deg2rad(theta_deg % 180.0)
    if not np.isfinite([*center, major, minor, theta]).all() or minor <= 0:
        raise ValueError('Degenerate fitted core ellipse')
    a, b = major/2 + radius, minor/2 + radius
    yy, xx = np.indices(local.shape, dtype=np.float64)
    dx, dy = xx-center[0], yy-center[1]
    u = dx*np.cos(theta) + dy*np.sin(theta)
    v = -dx*np.sin(theta) + dy*np.cos(theta)
    expanded = (u/a)**2 + (v/b)**2 <= 1.0
    torso = local.astype(bool) & expanded
    if torso.sum() < 5:
        raise ValueError('Expanded ellipse intersection has insufficient torso pixels')
    if torso_dimensions or morph_segments is not None:
        # Reuse exactly the same torso pixels and ellipse coordinates; no bbox,
        # +1 pixel convention, profile sampling or amplitude normalisation.
        length = float(np.max(u[torso]) - np.min(u[torso]))
        width = float(np.max(v[torso]) - np.min(v[torso]))
        features = np.array([length, width], dtype=np.float32)
        if not np.isfinite(features).all() or not (features > 0).all():
            raise ValueError('Torso length/width must be finite and positive')
        if morph_segments is not None and morph_segments > 1:
            longitudinal, transverse = u[torso], v[torso]
            edges = np.linspace(longitudinal.min(), longitudinal.max(), morph_segments+1)
            # [left, right), except the last interval includes its right endpoint.
            bins = np.clip(np.searchsorted(edges, longitudinal, side='right')-1, 0, morph_segments-1)
            widths = []
            for segment in range(morph_segments):
                values = transverse[bins == segment]
                if not len(values):
                    raise ValueError(f'Empty torso segment {segment+1}/{morph_segments}')
                widths.append(float(values.max()-values.min()))
            features = np.array([length, *widths], dtype=np.float32)
            if not np.isfinite(features).all() or not (features > 0).all():
                raise ValueError('All morphometry lengths/widths must be finite and positive')
            return features, {'torso_length_px': length, 'segment_widths_px': widths,
                              'input_shape': [morph_segments+1]}, None
        return features, {'torso_length_px': length, 'torso_width_px': width,
                          'input_shape': [2]}, None
    region = torso if profile_region == 'torso' else local.astype(bool)
    longitudinal, transverse = u[region], v[region]
    lower, upper = float(longitudinal.min()), float(longitudinal.max())
    if upper-lower < 2:
        raise ValueError('Torso longitudinal interval is too small for a width profile')
    # Dense bins of at most one original pixel along u.
    num_bins = max(3, int(np.ceil(upper-lower)))
    edges = np.linspace(lower, upper, num_bins+1)
    positions = (edges[:-1]+edges[1:])/2
    bins = np.clip(np.searchsorted(edges, longitudinal, side='right')-1, 0, num_bins-1)
    minima, maxima = np.full(num_bins, np.inf), np.full(num_bins, -np.inf)
    np.minimum.at(minima, bins, transverse)
    np.maximum.at(maxima, bins, transverse)
    valid = np.isfinite(minima) & np.isfinite(maxima)
    if valid.sum() < 2:
        raise ValueError('Insufficient valid longitudinal width bins')
    widths = np.interp(positions, positions[valid], (maxima-minima)[valid])
    widths = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(widths, median_window//2, mode='edge'), median_window), axis=-1)
    bounds = np.linspace(lower, upper, segments+1)
    # Only the longitudinal sampling coordinate is normalised; width stays pixels.
    t = np.linspace(0, 1, samples_per_segment)
    sampling = bounds[:-1, None] + (bounds[1:]-bounds[:-1])[:, None]*t
    profile = np.interp(sampling, positions, widths).astype(np.float32)
    if profile.shape != (segments, samples_per_segment) or not np.isfinite(profile).all() or profile.max() <= 0:
        raise ValueError('Invalid final width profile')
    diagnostics = {
        'mask_area': int(np.count_nonzero(mask)), 'cleaned_area': int(cleaned.sum()),
        'bbox_xyxy': list(bbox), 'Dmax': dmax, 'core_area': int(core.sum()),
        'ellipse_center_xy': (np.asarray(center)+origin).tolist(),
        'major_axis_diameter': float(major), 'minor_axis_diameter': float(minor),
        'theta_radians': float(theta), 'theta_degrees': float(np.rad2deg(theta)),
        'expanded_semi_axes': [float(a), float(b)], 'torso_area': int(torso.sum()),
        'profile_min': float(profile.min()), 'profile_max': float(profile.max()),
        'profile_mean': float(profile.mean()), 'profile_shape': list(profile.shape),
        'profile_region': profile_region, 'profile_region_area': int(region.sum()),
    }
    debug = None
    if return_debug:
        debug = {'local_mask': local, 'core': core, 'torso': torso.astype(np.uint8),
                 'expanded_ellipse': expanded.astype(np.uint8), 'origin_xy': origin.astype(int)}
    return profile, diagnostics, debug


def extract_bbox(mask):
    cleaned, (x0, y0, x1, y1) = clean_mask(mask)
    # clean_mask returns exclusive maxima; x1-x0 equals inclusive xmax-xmin+1.
    features = np.array([x1-x0, y1-y0], dtype=np.float32)
    return features, {'bbox_xyxy_inclusive': [x0, y0, x1-1, y1-1],
                      'cleaned_area': int(cleaned.sum()), 'input_shape': [2]}


def extract_torso_lw(mask, tau=0.25):
    features, diagnostics, _ = extract_profile(mask, tau=tau, torso_dimensions=True)
    return features, diagnostics


def extract_morphometry(mask, segments=4, tau=0.25):
    features, diagnostics, _ = extract_profile(mask, tau=tau, morph_segments=segments)
    return features, diagnostics


# Diagnostic-only alternative. Existing entry points above remain unchanged.
WIDTH_SMOOTH_WINDOW = 7
HIGH_RATIO = 0.55
LOW_RATIO = 0.35


class TorsoGeometryError(ValueError):
    """A diagnostic failure, with intermediate geometry when available."""

    def __init__(self, reason, diagnostics=None, debug=None):
        super().__init__(reason)
        self.diagnostics = diagnostics or {}
        self.debug = debug


def _true_runs(values):
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))


def _bottleneck_interval(widths, edges, high, low, gap_bins=3, low_run_bins=3):
    """Head-tail symmetric seed selection and persistent low-width crossings."""
    broad = widths >= high
    closed = broad.copy()
    for start, end in _true_runs(~broad):
        if start > 0 and end < len(broad)-1 and end-start+1 <= gap_bins:
            closed[start:end+1] = True
    components = _true_runs(closed)
    if not components:
        raise ValueError('no_central_broad_component')
    distances = [max(float(edges[start]), -float(edges[end+1]), 0.) for start, end in components]
    nearest = min(distances)
    candidates = [component for component, distance in zip(components, distances)
                  if np.isclose(distance, nearest, atol=1e-9, rtol=0)]
    if nearest > gap_bins * float(edges[1]-edges[0]) + 1e-9 or len(candidates) != 1:
        raise ValueError('no_unambiguous_central_broad_component_within_3_bins')
    start, end = candidates[0]

    def extend(endpoint, step):
        last = endpoint
        below_count = 0
        first_below = None
        index = endpoint + step
        while 0 <= index < len(widths):
            if widths[index] < low:
                if below_count == 0:
                    first_below = index
                below_count += 1
                if below_count >= low_run_bins:
                    return first_below - step, True
            else:
                below_count = 0
            last = index
            index += step
        return last, False

    left, left_crossing = extend(start, -1)
    right, right_crossing = extend(end, 1)
    return left, right, closed, (int(start), int(end)), left_crossing, right_crossing


def extract_torso_width_bottleneck(mask, tau=0.25, high_ratio=HIGH_RATIO,
                                   low_ratio=LOW_RATIO, smooth_window=WIDTH_SMOOTH_WINDOW,
                                   return_debug=False):
    """Return full-resolution torso mask, diagnostics, optional local debug arrays.

    The ellipse only supplies centre/orientation. Final support is cleaned mask
    intersected with a longitudinal interval, with no ellipse envelope. Ratios
    below 30% or above 90% are flags, never reasons to substitute a legacy mask.
    """
    if not (0 < tau < 1 and 0 < low_ratio < high_ratio < 1):
        raise ValueError('Require 0 < tau < 1 and 0 < low_ratio < high_ratio < 1')
    if not isinstance(smooth_window, (int, np.integer)) or smooth_window < 1 or smooth_window % 2 != 1:
        raise ValueError('smooth_window must be a positive odd integer')
    cleaned, bbox = clean_mask(mask)
    x0, y0, x1, y1 = bbox
    local = np.pad(cleaned[y0:y1, x0:x1], 1)
    origin = np.array([x0-1, y0-1], dtype=np.float64)
    # Match legacy DT, core, contour fitting and orientation exactly.
    distance = cv2.distanceTransform(local, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dmax = float(distance.max())
    if not np.isfinite(dmax) or dmax <= 0:
        raise TorsoGeometryError('invalid_distance_transform_maximum')
    core = largest_component((distance >= tau*dmax).astype(np.uint8))
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5 or cv2.contourArea(contour) <= 0:
        raise TorsoGeometryError('core_cannot_support_ellipse')
    center, axes, angle = cv2.fitEllipse(contour)
    if axes[0] >= axes[1]:
        major, minor, theta_deg = axes[0], axes[1], angle
    else:
        major, minor, theta_deg = axes[1], axes[0], angle + 90.
    theta = np.deg2rad(theta_deg % 180.)
    if not np.isfinite([*center, major, minor, theta]).all() or minor <= 0:
        raise TorsoGeometryError('degenerate_core_ellipse')
    yy, xx = np.indices(local.shape, dtype=np.float64)
    dx, dy = xx-center[0], yy-center[1]
    u = dx*np.cos(theta) + dy*np.sin(theta)
    v = -dx*np.sin(theta) + dy*np.cos(theta)
    foreground = local.astype(bool)
    longitudinal, transverse = u[foreground], v[foreground]
    lower, upper = float(longitudinal.min()), float(longitudinal.max())
    if upper-lower < 2:
        raise TorsoGeometryError('full_mask_longitudinal_extent_too_short')
    n_bins = max(3, int(np.ceil(upper-lower)))
    edges = np.linspace(lower, upper, n_bins+1)
    positions = (edges[:-1]+edges[1:])/2
    bins = np.clip(np.searchsorted(edges, longitudinal, side='right')-1, 0, n_bins-1)
    minima, maxima = np.full(n_bins, np.inf), np.full(n_bins, -np.inf)
    np.minimum.at(minima, bins, transverse)
    np.maximum.at(maxima, bins, transverse)
    valid = np.isfinite(minima) & np.isfinite(maxima)
    empty_runs = _true_runs(~valid)
    if valid.sum() < 2 or (~valid).sum() > max(3, int(np.ceil(.05*n_bins))) or any(e-s+1 > 3 for s, e in empty_runs):
        raise TorsoGeometryError('too_many_empty_width_bins')
    raw = np.full(n_bins, np.nan)
    raw[valid] = maxima[valid]-minima[valid]
    interpolated = np.interp(positions, positions[valid], raw[valid])
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(interpolated, smooth_window//2, mode='edge'), smooth_window), axis=-1)
    w_ref = float(np.percentile(smooth[valid], 95))
    high, low = high_ratio*w_ref, low_ratio*w_ref
    diagnostics = dict(original_area=int(np.count_nonzero(mask)), cleaned_area=int(cleaned.sum()),
                       bbox_xyxy=list(bbox), ellipse_center_xy=(np.asarray(center)+origin).tolist(),
                       theta_radians=float(theta), theta_degrees=float(np.rad2deg(theta)),
                       w_ref=w_ref, high_threshold=high, low_threshold=low,
                       high_ratio=high_ratio, low_ratio=low_ratio, smooth_window=int(smooth_window),
                       closing_gap_bins=3, boundary_low_run_bins=3, bin_size_px=float(edges[1]-edges[0]),
                       empty_bins=int((~valid).sum()), original_longitudinal_length=upper-lower)
    debug = dict(local_mask=local, core=core, origin_xy=origin.astype(int), u=u, v=v,
                 positions=positions, edges=edges, w_raw=raw, w_interpolated=interpolated, w_smooth=smooth)
    if not np.isfinite(w_ref) or w_ref <= 0:
        raise TorsoGeometryError('nonpositive_width_reference', diagnostics, debug)
    try:
        left, right, broad, seed, left_crossing, right_crossing = _bottleneck_interval(smooth, edges, high, low)
    except ValueError as exc:
        raise TorsoGeometryError(str(exc), diagnostics, debug) from exc
    u_start, u_end = float(edges[left]), float(edges[right+1])
    torso = foreground & (u >= u_start) & (u <= u_end)
    area = int(torso.sum())
    if area == 0 or not u_start < u_end:
        raise TorsoGeometryError('empty_or_invalid_torso_interval', diagnostics, debug)
    length = float(np.ptp(u[torso]))
    area_ratio = area/diagnostics['original_area']
    length_ratio = length/(upper-lower)
    flags = []
    if area < 5:
        flags.append('torso_area_below_5_pixels')
    if length < 3 or length_ratio < .30:
        flags.append('torso_interval_too_short_lt3px_or_lt30pct')
    if area_ratio < .30:
        flags.append('torso_area_retention_below_30pct')
    if area_ratio > .90:
        flags.append('torso_area_retention_above_90pct')
    diagnostics.update(u_start=u_start, u_end=u_end, new_torso_area=area,
                       new_longitudinal_length=length, area_retention=area_ratio,
                       longitudinal_retention=length_ratio, diagnostic_flags=flags,
                       left_low_crossing=left_crossing, right_low_crossing=right_crossing)
    debug.update(torso=torso.astype(np.uint8), broad_closed=broad, seed_indices=seed)
    result = np.zeros_like(cleaned)
    result[y0:y1, x0:x1] = torso[1:-1, 1:-1]
    return result, diagnostics, debug if return_debug else None
