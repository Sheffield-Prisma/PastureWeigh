"""PCA/peak-width torso core, used only by explicit pca_peak opt-in runs."""
import cv2
import numpy as np

if __package__:
    from .geometry import clean_mask, largest_component
else:
    from geometry import clean_mask, largest_component


def extract_torso_pca_peak(mask, peak_ratio=.68, min_length_ratio=.52,
                           smooth_window=7, return_debug=False):
    """Return full-resolution core, explicit fallback diagnostics and local arrays.

    PCA uses all input foreground pixels. Widths retain pixel units; smoothing
    is a seven-bin normalized box kernel, not sample amplitude normalization.
    Small-core fallback returns the complete cleaned mask. Final cleanup is
    clipped to the chosen interval and cleaned support to avoid added branches.
    """
    if not 0 < peak_ratio < 1 or not 0 < min_length_ratio <= 1:
        raise ValueError('Ratios must lie in (0, 1] with peak_ratio < 1')
    if not isinstance(smooth_window, (int, np.integer)) or smooth_window < 1 or smooth_window % 2 != 1:
        raise ValueError('smooth_window must be a positive odd integer')
    original = (np.asarray(mask) > 0).astype(np.uint8)
    cleaned, bbox = clean_mask(mask)
    ys, xs = np.nonzero(original)
    if len(xs) < 3:
        raise ValueError('Too few foreground pixels for PCA')
    points = np.column_stack([xs, ys]).astype(np.float64)
    center = points.mean(axis=0)
    covariance = (points-center).T @ (points-center) / len(points)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    p = vectors[:, -1]
    if p[np.argmax(np.abs(p))] < 0:
        p = -p
    q = np.array([-p[1], p[0]])
    x0,y0,x1,y1 = bbox
    # Crop covers cleaned support, while the profile uses original foreground.
    yy,xx = np.indices((y1-y0, x1-x0), dtype=float)
    dx,dy = xx+x0-center[0], yy+y0-center[1]
    u,v = dx*p[0]+dy*p[1], dx*q[0]+dy*q[1]
    longitudinal, transverse = (points-center) @ p, (points-center) @ q
    lo, hi = float(longitudinal.min()), float(longitudinal.max())
    local = cleaned[y0:y1, x0:x1]
    diag = dict(bbox_xyxy=list(bbox), pca_center_xy=center.tolist(), major_axis=p.tolist(),
                transverse_axis=q.tolist(), eigenvalues=eigenvalues.tolist(),
                peak_ratio=peak_ratio, min_length_ratio=min_length_ratio,
                smooth_window=int(smooth_window), smoothing='normalized_box_kernel', closing_kernel=3,
                original_area=int(original.sum()), full_cleaned_area=int(cleaned.sum()),
                original_longitudinal_length=hi-lo, fallback=False, fallback_reason='',
                candidate_u_start=None, candidate_u_end=None, candidate_length_ratio=None,
                w_peak=None, threshold=None)
    debug = dict(local_mask=local, original_mask=original[y0:y1,x0:x1], u=u,v=v,
                 positions=np.array([]),w_raw=np.array([]),w_smooth=np.array([]))
    reason = ''
    candidate = None
    if not np.isfinite(eigenvalues).all() or hi-lo < 3:
        reason = 'invalid_or_too_short_pca_geometry'
    elif eigenvalues[-1] <= 1.05*max(eigenvalues[0], 1e-12):
        reason = 'near_isotropic_mask_unstable_axis'
    else:
        # Centre the end bins on foreground extrema: integer-aligned masks then
        # have symmetric support under head-tail reversal (no half-open bias).
        count = max(3, int(np.ceil(hi-lo))+1)
        step = (hi-lo)/(count-1)
        edges = np.linspace(lo-step/2,hi+step/2,count+1)
        positions = (edges[:-1]+edges[1:])/2
        bins = np.clip(np.searchsorted(edges,longitudinal,side='right')-1,0,count-1)
        minima,maxima = np.full(count,np.inf),np.full(count,-np.inf)
        np.minimum.at(minima,bins,transverse)
        np.maximum.at(maxima,bins,transverse)
        valid = np.isfinite(minima) & np.isfinite(maxima)
        raw = np.full(count,np.nan)
        raw[valid] = maxima[valid]-minima[valid]
        if valid.sum() < 2 or (~valid).sum() > max(3,int(np.ceil(.05*count))):
            reason = 'unstable_profile_too_many_empty_bins'
        else:
            widths = np.interp(positions,positions[valid],raw[valid])
            smooth = np.convolve(np.pad(widths,smooth_window//2,mode='edge'),
                                 np.ones(smooth_window)/smooth_window,mode='valid')
            peak = float(smooth.max())
            diag.update(w_peak=peak,threshold=peak_ratio*peak,bin_size_px=float(edges[1]-edges[0]))
            debug.update(positions=positions,w_raw=raw,w_smooth=smooth,edges=edges)
            if not np.isfinite(peak) or peak <= 0:
                reason = 'invalid_peak_width'
            else:
                peaks = np.flatnonzero(np.isclose(smooth,peak,rtol=1e-12,atol=1e-12))
                # A plateau is a single interval; among separate equal peaks use
                # distance to the PCA centre rather than a preferred head side.
                nearest = np.min(np.abs(positions[peaks]))
                tied = peaks[np.isclose(np.abs(positions[peaks]),nearest,atol=1e-9,rtol=0)]
                index = int(tied[0])
                left=right=index
                while left>0 and smooth[left-1]>=peak_ratio*peak:
                    left-=1
                while right<count-1 and smooth[right+1]>=peak_ratio*peak:
                    right+=1
                if any(t<left or t>right for t in tied):
                    reason = 'ambiguous_equal_peak_components'
                start,end = float(edges[left]),float(edges[right+1])
                candidate = local.astype(bool) & (u>=start) & (u<=end)
                length = float(np.ptp(u[candidate])) if candidate.any() else 0.
                diag.update(candidate_u_start=start,candidate_u_end=end,
                            candidate_length_ratio=length/(hi-lo),candidate_area=int(candidate.sum()))
                if not reason and length/(hi-lo)<min_length_ratio:
                    reason = 'candidate_length_below_52pct'
                if not reason:
                    closed = cv2.morphologyEx(np.pad(candidate.astype(np.uint8),2),cv2.MORPH_CLOSE,
                                             np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
                    filled,_ = clean_mask(closed)
                    core = filled[2:-2,2:-2].astype(bool) & local.astype(bool) & (u>=start) & (u<=end)
                    if core.sum()<5:
                        reason = 'cleaned_core_too_small'
                    else:
                        core = largest_component(core.astype(np.uint8))
                        if np.ptp(u[core.astype(bool)])/(hi-lo)<min_length_ratio:
                            reason = 'post_cleanup_length_below_52pct'
    if reason:
        core = local.copy()
        diag.update(fallback=True,fallback_reason=reason)
    support = core.astype(bool)
    yy,xx = np.nonzero(core)
    rectangle = cv2.minAreaRect(np.column_stack([xx+x0,yy+y0]).astype(np.float32))
    length,width = sorted(rectangle[1],reverse=True)
    if not np.isfinite([length,width]).all() or width<=0:
        raise ValueError('Invalid final rotated rectangle')
    diag.update(torso_area=int(core.sum()),area_retention=float(core.sum()/original.sum()),
                longitudinal_retention=float(np.ptp(u[support])/(hi-lo)),
                u_start=float(u[support].min()),u_end=float(u[support].max()),
                length_px=float(length),width_px=float(width),
                rectangle_xy=cv2.boxPoints(rectangle).tolist())
    result = np.zeros_like(cleaned)
    result[y0:y1,x0:x1] = core
    debug.update(torso=core,candidate=candidate)
    return result,diag,debug if return_debug else None


def extract_morphometry_pca(mask, segments=4):
    """Existing [L, W1, ..., WM] definition in the selected torso's PCA axes.

    Keeps original pixel spans and shared longitudinal partitions. Minimum-area
    rectangle dimensions remain diagnostic only, so segment geometry is coherent.
    """
    if segments not in (1, 2, 4, 8):
        raise ValueError('Morphometry segments must be 1/2/4/8')
    _, diagnostics, debug = extract_torso_pca_peak(mask, return_debug=True)
    torso = debug['torso'].astype(bool)
    longitudinal, transverse = debug['u'][torso], debug['v'][torso]
    length = float(np.ptp(longitudinal))
    edges = np.linspace(longitudinal.min(), longitudinal.max(), segments+1)
    bins = np.clip(np.searchsorted(edges, longitudinal, side='right')-1, 0, segments-1)
    widths = []
    for segment in range(segments):
        values = transverse[bins == segment]
        if not len(values):
            raise ValueError(f'Empty PCA torso segment {segment+1}/{segments}')
        widths.append(float(np.ptp(values)))
    features = np.array([length, *widths], dtype=np.float32)
    if not np.isfinite(features).all() or not (features > 0).all():
        raise ValueError('PCA morphometry features must be finite and positive')
    diagnostics.update(torso_length_px=length, segment_widths_px=widths,
                       feature_coordinate_system='PCA major/transverse axes', input_shape=[segments+1])
    return features, diagnostics
