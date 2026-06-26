"""_reconstruction_impl — extracted from cell0_v32 (no plotting, no matplotlib)."""
from __future__ import annotations
import numpy as np

def normalize(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.min(x)
    m = np.max(x)
    if m < 1e-12:
        return np.zeros_like(x)
    return x / m


def _bilinear_sample(img, y, x):
    H, W = img.shape
    y = np.clip(y, 0, H - 1.001); x = np.clip(x, 0, W - 1.001)
    y0 = int(np.floor(y)); x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1); x1 = min(x0 + 1, W - 1)
    dy = y - y0; dx = x - x0
    return (
        img[y0, x0]*(1-dy)*(1-dx) + img[y1, x0]*dy*(1-dx) +
        img[y0, x1]*(1-dy)*dx     + img[y1, x1]*dy*dx
    )

def _smooth4(x, rounds=8):
    y = x.copy().astype(float)
    for _ in range(rounds):
        y = (0.50*y + 0.125*np.roll(y,1,0) + 0.125*np.roll(y,-1,0) +
                       0.125*np.roll(y,1,1) + 0.125*np.roll(y,-1,1))
    return y

def _anisotropic_gaussian_grid(H, W, center, direction, sigma_parallel=18.0, sigma_perp=5.0):
    rr_, cc_ = np.indices((H, W))
    y0, x0 = center
    dyy = rr_ - y0; dxx = cc_ - x0
    d = np.array(direction, dtype=float)
    n = np.linalg.norm(d)
    d = d / (n + 1e-12) if n > 1e-8 else np.array([1.0, 0.0])
    parallel = dyy * d[0] + dxx * d[1]
    perp     = -dyy * d[1] + dxx * d[0]
    return np.exp(-0.5 * ((parallel / sigma_parallel)**2 + (perp / sigma_perp)**2))

def _local_maxima_2d(x, threshold_ratio=0.55, min_sep=10, top_k=8):
    Xs = _smooth4(x, rounds=3)
    cand = (
        (Xs > np.roll(Xs, 1, 0)) & (Xs > np.roll(Xs, -1, 0)) &
        (Xs > np.roll(Xs, 1, 1)) & (Xs > np.roll(Xs, -1, 1))
    )
    thr = threshold_ratio * np.max(Xs) if np.max(Xs) > 1e-12 else 0.0
    cand = cand & (Xs > thr)
    ys, xs = np.where(cand)
    vals = Xs[ys, xs]
    order = np.argsort(-vals)
    picked = []
    for idx in order:
        y, x_ = int(ys[idx]), int(xs[idx])
        if all((y - py)**2 + (x_ - px)**2 >= min_sep**2 for py, px in picked):
            picked.append((y, x_))
            if len(picked) >= top_k:
                break
    return np.array(picked, dtype=float) if picked else np.zeros((0, 2), dtype=float)

def _triangle_score(p0, p1, p2, field):
    """Cell-43-style: area * angle balance * mean field value at vertices."""
    a = np.linalg.norm(p1 - p0); b = np.linalg.norm(p2 - p1); c = np.linalg.norm(p0 - p2)
    s = 0.5 * (a + b + c)
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return -np.inf
    area = np.sqrt(area_sq)
    # angle balance: how close to equilateral (penalise long thin tris)
    eps = 1e-9
    cosA = (b*b + c*c - a*a) / (2*b*c + eps)
    cosB = (a*a + c*c - b*b) / (2*a*c + eps)
    cosC = (a*a + b*b - c*c) / (2*a*b + eps)
    angA = np.arccos(np.clip(cosA, -1, 1))
    angB = np.arccos(np.clip(cosB, -1, 1))
    angC = np.arccos(np.clip(cosC, -1, 1))
    target = np.pi / 3.0
    balance = np.exp(-((angA-target)**2 + (angB-target)**2 + (angC-target)**2) * 1.0)
    # value: mean field at vertices
    val = (_bilinear_sample(field, p0[0], p0[1]) +
           _bilinear_sample(field, p1[0], p1[1]) +
           _bilinear_sample(field, p2[0], p2[1])) / 3.0
    return area * balance * (0.1 + val)

def _choose_seed_triangle(pts, field):
    n = len(pts)
    best = -np.inf; tri = None
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = _triangle_score(pts[i], pts[j], pts[k], field)
                if s > best:
                    best = s; tri = np.array([pts[i], pts[j], pts[k]])
    return tri

def _bezier_quad(p0, p1, pc, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return ((1-t)**2)*p0 + 2*(1-t)*t*pc + (t**2)*p1

# --- Main entry ---------------------------------------------


def _reconstruct_picture_impl(*, source, seed=None, n_lobes_override=None):
    """UNIFIED reconstruction pipeline. Works for any source + any seed.

    Args:
        source: 2D array — the structural field to reconstruct from.
            For engine use: composite of density/memory/heat/charge/inverse_field.
            For arbitrary input: ANY 2D pattern.
        seed: 2D array or None — the constraint/seed pattern that shapes
            the master transfer. If None, the system builds its own soft
            constraint from the source's gradient.
        n_lobes_override: int or None — force a lobe count, otherwise auto-detect
            from source peaks.

    Returns:
        dict with all chain intermediates + 'master_regrowth' (the picture).

    The pipeline runs the original notebook chain on the source:
        source → inward_defined_home (cell 34)
               → single_point_flower (cell 37)
               → master_output (cell 41 master, seed-constrained)
               → accepted_triangles, regrown (cells 43/44/45)
               → weighted_peak, gap_signature, curves_cubed (cells 48/50/51)
               → final_point (cell 53)
               → MASTER GLOBAL-CENTRED — picture flower
    """
    X = normalize(source)
    constraint = seed
    H, W = X.shape

    # Lightweight proxies for heat/blank/entropy that the chain still needs.
    # entropy proxy: smoothed source residual (was heat + tension from engine)
    entropy_proxy = normalize(np.abs(X - _smooth4(X, rounds=6)))

    # blank field: high where the source is empty
    blank_field = normalize(1.0 - _smooth4(X, rounds=8))

    # common points: peaks of X — these define the LOBES of the reconstruction.
    pts_common = _local_maxima_2d(X, threshold_ratio=0.55, min_sep=10, top_k=8)
    if len(pts_common) < 2:
        pts_common = np.array([[H//2, W//2], [H//2 - 20, W//2 + 20]], dtype=float)

    # n_lobes governs the multiplicity of EVERY downstream sampler so the
    # full reconstruction inherits the same N-fold structure as the flower.
    if n_lobes_override is not None:
        n_lobes = int(max(3, min(12, n_lobes_override)))
    else:
        n_lobes = int(max(3, min(12, len(pts_common))))

    # blank points: peaks of blank_field
    pts_blank = _local_maxima_2d(blank_field, threshold_ratio=0.55, min_sep=12, top_k=max(3, n_lobes // 2))

    # =================================================================
    # ===== Stage 1 — FULL ORIGINAL CHAIN =====
    # 1a) inward-defined structure (cell 34)
    # 1b) single-point flower (cell 37, PCA-based) using 1a as source
    # 1c) master transfer system (cell 41) using 1b as source +
    #     custom constraint = engine seed (anti_seed_pattern) by default
    # =================================================================
    rr_, cc_ = np.indices((H, W))

    # ---------- 1a. Inward-defined structure (cell 34) ----------
    # centre from mass of X (the whole-field first moment, not attractor avg)
    mass_X = X + 1e-8
    cy_mass = float(np.sum(rr_ * mass_X) / np.sum(mass_X))
    cx_mass = float(np.sum(cc_ * mass_X) / np.sum(mass_X))
    center_mass = np.array([cy_mass, cx_mass], dtype=float)

    # re-extract attractors from X ranked by value (cell 34 uses this, not
    # the _local_maxima_2d we used elsewhere — keep both for fidelity)
    flat_idx = np.argsort(X.ravel())[::-1]
    inward_pts = []
    min_dist_inward = 12
    for idx in flat_idx[:600]:    # cap scan length for speed
        y_, x_ = np.unravel_index(idx, X.shape)
        p = np.array([float(y_), float(x_)])
        if all(np.linalg.norm(p - q) > min_dist_inward for q in inward_pts):
            inward_pts.append(p)
        if len(inward_pts) >= n_lobes:
            break
    if len(inward_pts) < 3:
        inward_pts = pts_common[:n_lobes].tolist()
    inward_pts = np.array(inward_pts, dtype=float)

    # inward accumulation: per attractor, walk attractor -> centre stamping G
    inward_field = np.zeros((H, W), dtype=float)
    inward_path_sum = np.zeros((H, W), dtype=float)
    for p in inward_pts:
        inward_dir = center_mass - p
        dist = np.linalg.norm(inward_dir)
        n_steps_inward = max(12, int(dist))
        for s in range(n_steps_inward):
            a = s / max(n_steps_inward - 1, 1)
            pos = (1 - a) * p + a * center_mass
            G = _anisotropic_gaussian_grid(
                H, W, center=pos, direction=inward_dir,
                sigma_parallel=8.0 + 0.08 * dist, sigma_perp=2.8,
            )
            w = 0.35 + 0.65 * a   # stronger near centre
            inward_field += w * G
            inward_path_sum += G
    inward_field = inward_field / (inward_path_sum + 1e-6)
    inward_field = normalize(inward_field)

    gy_inw, gx_inw = np.gradient(inward_field)
    edge_constraint = normalize(np.sqrt(gx_inw**2 + gy_inw**2))
    grid_constraint = normalize(
        0.50 * inward_field + 0.30 * edge_constraint + 0.20 * X
    )
    grid_self_defined = grid_constraint.copy()
    for _ in range(18):
        diffusion = (
            np.roll(grid_self_defined, 1, 0) + np.roll(grid_self_defined, -1, 0) +
            np.roll(grid_self_defined, 1, 1) + np.roll(grid_self_defined, -1, 1)
        ) / 4.0
        grid_self_defined = normalize(
            0.55 * grid_self_defined + 0.25 * diffusion + 0.20 * grid_constraint
        )
    inward_defined_home = normalize(
        0.55 * grid_self_defined + 0.25 * inward_field + 0.20 * edge_constraint
    )

    # ---------- 1b. Single-point flower (cell 37 — PCA from inward_defined_home) ----------
    X_flower = inward_defined_home
    weights_f = np.array([_bilinear_sample(X_flower, p[0], p[1]) for p in inward_pts], dtype=float)
    if weights_f.sum() > 1e-12:
        weights_f = weights_f / weights_f.sum()
    else:
        weights_f = np.ones_like(weights_f) / max(1, len(weights_f))
    center = np.sum(inward_pts * weights_f[:, None], axis=0)

    low_f = normalize(_smooth4(X_flower, rounds=10))
    residual_f = X_flower - low_f
    residual_n = normalize(np.abs(residual_f))   # this is the "residual" used downstream
    gy, gx = np.gradient(residual_f)             # gradients for curve bending

    P_f = inward_pts - center
    Cov_f = (P_f * weights_f[:, None]).T @ P_f
    try:
        eigvals_f, eigvecs_f = np.linalg.eigh(Cov_f)
    except np.linalg.LinAlgError:
        eigvals_f = np.array([1.0, 1.0]); eigvecs_f = np.eye(2)
    order_f = np.argsort(eigvals_f)[::-1]; eigvecs_f = eigvecs_f[:, order_f]
    pca_dirs = [eigvecs_f[:, i] for i in range(min(3, eigvecs_f.shape[1]))]
    all_pca_dirs = []
    for d in pca_dirs:
        all_pca_dirs.append(d); all_pca_dirs.append(-d)

    cy_i = int(np.clip(round(center[0]), 0, H-1)); cx_i = int(np.clip(round(center[1]), 0, W-1))
    sp_seed = np.zeros((H, W), dtype=float); sp_seed[cy_i, cx_i] = 1.0
    sp_seed = normalize(_smooth4(sp_seed, rounds=5))

    reproj_f = np.zeros((H, W), dtype=float); path_f = np.zeros((H, W), dtype=float)
    for d in all_pca_dirs:
        for s in range(1, 40):    # 40 steps (was 55, slightly reduced for time budget)
            a = s / 39.0
            pos = center + a * (18 + 26 * a) * d
            y, x = pos
            if not (0 <= y < H and 0 <= x < W):
                continue
            flow = np.array([_bilinear_sample(gy, y, x), _bilinear_sample(gx, y, x)], dtype=float)
            d_eff = d + 0.8 * flow
            nrm = np.linalg.norm(d_eff)
            d_eff = d_eff / (nrm + 1e-12) if nrm > 1e-8 else d
            G = _anisotropic_gaussian_grid(
                H, W, center=(y, x), direction=d_eff,
                sigma_parallel=12.0 + 8.0 * a, sigma_perp=3.5 + 1.5 * (1 - a),
            )
            local_r = _bilinear_sample(residual_n, y, x)
            w = (1.1 - 0.5 * a) * (0.35 + 0.65 * local_r)
            reproj_f += w * G; path_f += G
    reproj_f = reproj_f / (path_f + 1e-6); reproj_f = normalize(reproj_f)
    edge_memory_f = normalize(np.sqrt(np.gradient(X_flower)[0]**2 + np.gradient(X_flower)[1]**2))
    single_point_flower = normalize(
        0.55 * reproj_f + 0.25 * edge_memory_f + 0.20 * sp_seed
    )

    # ---------- 1c. MASTER TRANSFER SYSTEM (cell 41) ----------
    source_field = single_point_flower
    # constraint: external (engine seed) or default soft gradient
    if constraint is None:
        gy0, gx0 = np.gradient(source_field)
        cl = normalize(np.sqrt(gx0**2 + gy0**2))
        constraint_layer = normalize(0.6 * _smooth4(cl, rounds=4) + 0.4 * source_field)
    else:
        constraint_layer = normalize(constraint)

    # per-attractor outward directions from inward_pts on the new source
    weights_h = np.array([_bilinear_sample(source_field, p[0], p[1]) for p in inward_pts], dtype=float)
    if weights_h.sum() > 1e-12:
        weights_h = weights_h / weights_h.sum()
    else:
        weights_h = np.ones_like(weights_h) / max(1, len(weights_h))
    center_h = np.sum(inward_pts * weights_h[:, None], axis=0)

    dirs_h = []; dir_weights_h = []
    for p, w in zip(inward_pts, weights_h):
        d = p - center_h
        n_ = np.linalg.norm(d)
        if n_ > 1e-8:
            dirs_h.append(d / n_); dir_weights_h.append(w)
    dirs_h = np.array(dirs_h, dtype=float)
    dir_weights_h = np.array(dir_weights_h, dtype=float)
    if dir_weights_h.sum() > 1e-12:
        dir_weights_h = dir_weights_h / dir_weights_h.sum()

    cy_h, cx_h = center_h
    dy_h = rr_ - cy_h; dx_h = cc_ - cx_h
    r_h = np.sqrt(dy_h**2 + dx_h**2)
    wave_signature = np.cos(0.22 * r_h) * np.exp(-r_h**2 / (2 * 12.0**2))
    heat_core = normalize(wave_signature)

    gy_s, gx_s = np.gradient(source_field)
    edge_memory_h = normalize(np.sqrt(gx_s**2 + gy_s**2))
    transfer_fabric = normalize(
        0.40 * source_field + 0.20 * edge_memory_h +
        0.20 * heat_core   + 0.20 * constraint_layer
    )

    heat_decay = normalize(heat_core * np.exp(-r_h / 22.0))
    implanted_heat = normalize(
        0.60 * transfer_fabric * heat_decay +
        0.25 * _smooth4(heat_decay, rounds=4) +
        0.15 * constraint_layer
    )

    outward_field = np.zeros((H, W), dtype=float)
    out_support   = np.zeros((H, W), dtype=float)
    for d, dw in zip(dirs_h, dir_weights_h):
        for s in range(1, 50):   # 50 steps (was 70, reduced)
            a = s / 49.0
            pos = center_h + a * (10 + 28 * a) * d
            y, x = pos
            if not (0 <= y < H and 0 <= x < W):
                continue
            G = _anisotropic_gaussian_grid(
                H, W, center=(y, x), direction=d,
                sigma_parallel=10.0 + 9.0 * a, sigma_perp=3.5 + 1.0 * (1 - a),
            )
            local_heat = _bilinear_sample(implanted_heat, y, x)
            local_constraint = _bilinear_sample(constraint_layer, y, x)
            w = dw * (1.15 - 0.45 * a) * (0.25 + 0.50 * local_heat + 0.25 * local_constraint)
            outward_field += w * G; out_support += G
    outward_field = outward_field / (out_support + 1e-6); outward_field = normalize(outward_field)

    spent_energy = normalize(np.clip(outward_field - 0.55 * implanted_heat, 0, None))

    bypass_field = np.zeros((H, W), dtype=float)
    bypass_support = np.zeros((H, W), dtype=float)
    for d, dw in zip(dirs_h, dir_weights_h):
        inward = -d
        for s in range(1, 40):    # 40 steps (was 55, reduced)
            a = s / 39.0
            pos = center_h + (1 - a) * 22.0 * d
            y, x = pos
            if not (0 <= y < H and 0 <= x < W):
                continue
            G = _anisotropic_gaussian_grid(
                H, W, center=(y, x), direction=inward,
                sigma_parallel=8.0 + 4.0 * (1 - a), sigma_perp=3.0,
            )
            local_spent = _bilinear_sample(spent_energy, y, x)
            local_constraint = _bilinear_sample(constraint_layer, y, x)
            w = dw * a * (0.20 + 0.55 * local_spent + 0.25 * local_constraint)
            bypass_field += w * G; bypass_support += G
    bypass_field = bypass_field / (bypass_support + 1e-6); bypass_field = normalize(bypass_field)

    transfer_layer = normalize(
        0.40 * implanted_heat + 0.35 * outward_field +
        0.15 * bypass_field   + 0.10 * constraint_layer
    )
    semi_closed_field = normalize(
        0.30 * transfer_fabric + 0.20 * heat_core +
        0.25 * outward_field   + 0.15 * bypass_field +
        0.10 * constraint_layer
    )
    master_output = semi_closed_field.copy()
    for _ in range(10):
        master_output = normalize(
            0.65 * master_output + 0.35 * _smooth4(master_output, rounds=2)
        )

    # =================================================================
    # The new FLOWER for all downstream stages = MASTER OUTPUT.
    # Downstream code (geometry, perigram, petals, etc.) reads this.
    # =================================================================
    flower = master_output
    edge_memory = edge_memory_h    # used by some downstream stages

    # ============================================================
    # ===== ORIGINAL TRIANGULATION + BOUNDARY + CURVES =====
    # Cells 43 (triangulation), 44 (boundary reconstruction), 45 (curves
    # + collapse criterion). All run on master_output as source.
    # ============================================================
    src43 = master_output

    def _find_center43(field):
        sm = _smooth4(field, rounds=10)
        rs = normalize(np.abs(field - sm))
        mix43 = normalize(0.65 * sm + 0.35 * rs)
        w43 = mix43 + 1e-12
        r0 = float(np.sum(rr_ * w43) / np.sum(w43))
        c0 = float(np.sum(cc_ * w43) / np.sum(w43))
        return np.array([r0, c0], dtype=float), sm, rs, mix43

    def _lm43(x_, threshold_ratio=0.68):
        thr = threshold_ratio * float(np.max(x_))
        if thr <= 0:
            return []
        cand = (
            (x_ > np.roll(x_, 1, 0)) & (x_ > np.roll(x_, -1, 0)) &
            (x_ > np.roll(x_, 1, 1)) & (x_ > np.roll(x_, -1, 1)) & (x_ > thr)
        )
        ys, xs = np.where(cand)
        return [(int(ys[i]), int(xs[i]), float(x_[ys[i], xs[i]])) for i in range(len(ys))]

    def _merge43(plist, min_dist=9):
        if not plist:
            return np.zeros((0, 2)), np.zeros((0,))
        pa = np.array([[p[0], p[1]] for p in plist], dtype=float)
        va = np.array([p[2] for p in plist], dtype=float)
        used = np.zeros(len(plist), dtype=bool)
        mp = []; mv = []
        for i in range(len(plist)):
            if used[i]:
                continue
            grp = [i]; used[i] = True; changed = True
            while changed:
                changed = False
                for j in range(len(plist)):
                    if used[j]:
                        continue
                    if np.any(np.linalg.norm(pa[j] - pa[grp], axis=1) < min_dist):
                        used[j] = True; grp.append(j); changed = True
            ws = va[grp]; ps = pa[grp]
            mp.append(np.sum(ps * ws[:, None], axis=0) / (float(np.sum(ws)) + 1e-12))
            mv.append(float(np.max(ws)))
        return np.array(mp), np.array(mv)

    def _area43(p0, p1, p2):
        v1 = p1 - p0; v2 = p2 - p0
        return 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])

    def _angs43(p0, p1, p2):
        def _a(a, b, c):
            ba = a - b; bc = c - b
            cv = np.clip(float(np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-12)), -1, 1)
            return float(np.degrees(np.arccos(cv)))
        return np.array([_a(p1, p0, p2), _a(p0, p1, p2), _a(p0, p2, p1)])

    def _score43(p0, p1, p2, field):
        ds = np.array([np.linalg.norm(p1-p0), np.linalg.norm(p2-p1), np.linalg.norm(p0-p2)])
        area = _area43(p0, p1, p2)
        if area < 10:
            return np.inf
        ang = _angs43(p0, p1, p2)
        cn = (p0 + p1 + p2) / 3.0
        r_ = int(np.clip(round(cn[0]), 0, field.shape[0]-1))
        c_ = int(np.clip(round(cn[1]), 0, field.shape[1]-1))
        return 0.40 * float(np.std(ds)) + 0.30 * float(np.std(ang)) - 35.0 * float(field[r_, c_])

    def _seed_tri(pts43, vals43, field):
        if len(pts43) < 3:
            return None, np.inf
        best = None; best_s = np.inf
        order = np.argsort(vals43)[::-1][:min(len(vals43), 8)]
        for ai in range(len(order)):
            for bi in range(ai+1, len(order)):
                for ci in range(bi+1, len(order)):
                    i, j, k = int(order[ai]), int(order[bi]), int(order[ci])
                    s = _score43(pts43[i], pts43[j], pts43[k], field)
                    if s < best_s:
                        best_s = s; best = (i, j, k)
        return best, best_s

    def _angle_about(p, c_):
        return float(np.arctan2(p[0]-c_[0], p[1]-c_[1]))

    def _find_diag(i, pts43, c_, used, rtol=0.75):
        p = pts43[i]; ai = _angle_about(p, c_)
        ri = float(np.linalg.norm(p - c_)) + 1e-12
        best_j = None; best_s = -np.inf
        for j in range(len(pts43)):
            if j == i or j in used:
                continue
            q = pts43[j]; aj = _angle_about(q, c_)
            rj = float(np.linalg.norm(q - c_)) + 1e-12
            dang = abs(float(np.arctan2(np.sin(aj-ai), np.cos(aj-ai))))
            if abs(rj - ri) / max(ri, rj) > rtol:
                continue
            sc = 2.0 * (-abs(dang - np.pi/2)) + 0.5 * (-abs(rj-ri)/max(ri, rj))
            if sc > best_s:
                best_s = sc; best_j = j
        return best_j

    def _find_across(i, pts43, c_, used):
        p = pts43[i]; v = p - c_; nv = float(np.linalg.norm(v)) + 1e-12
        best_j = None; best_s = np.inf
        for j in range(len(pts43)):
            if j == i or j in used:
                continue
            q = pts43[j]; w_ = q - c_; nw = float(np.linalg.norm(w_)) + 1e-12
            cosang = float(np.dot(v, w_) / (nv*nw))
            if cosang > -0.2:
                continue
            sc = float(np.linalg.norm(q-p)) - 25.0 * (-cosang)
            if sc < best_s:
                best_s = sc; best_j = j
        return best_j

    # run cell-43 triangulation
    center43, smooth43, residual43, mix43 = _find_center43(src43)
    raw43 = _lm43(src43, threshold_ratio=0.68)
    pts43, vals43 = _merge43(raw43, min_dist=9)
    accepted_triangles = []
    seed_score = np.inf
    if len(pts43) >= 3:
        seed_tri, seed_score = _seed_tri(pts43, vals43, src43)
        if seed_tri is not None:
            accepted_triangles.append(seed_tri)
            used_pts = set(seed_tri); frontier = list(seed_tri)
            for _ in range(8):
                new_tris = []; next_front = []
                for i in frontier:
                    j_d = _find_diag(i, pts43, center43, used_pts)
                    j_a = _find_across(i, pts43, center43, used_pts)
                    if j_d is None or j_a is None or j_d == j_a:
                        continue
                    tri = tuple(sorted((i, j_d, j_a)))
                    if tri in accepted_triangles or tri in new_tris:
                        continue
                    p0_, p1_, p2_ = pts43[tri[0]], pts43[tri[1]], pts43[tri[2]]
                    s = _score43(p0_, p1_, p2_, src43)
                    if np.isfinite(s) and s < seed_score + 25:
                        new_tris.append(tri)
                        for jj in tri:
                            if jj not in used_pts:
                                next_front.append(jj)
                if not new_tris:
                    break
                accepted_triangles.extend(new_tris)
                used_pts.update(next_front)
                frontier = next_front

    # cell 44: boundary / interior / vertex / reconstructed_structure / regrown
    boundary_field = np.zeros((H, W), dtype=float)
    interior_field = np.zeros((H, W), dtype=float)
    vertex_field   = np.zeros((H, W), dtype=float)

    def _dist_seg(rr_a, cc_a, p0_, p1_):
        v0 = np.asarray(p0_, dtype=float); v1 = np.asarray(p1_, dtype=float)
        dv = v1 - v0; denom = float(np.dot(dv, dv)) + 1e-12
        t = ((rr_a - v0[0]) * dv[0] + (cc_a - v0[1]) * dv[1]) / denom
        t = np.clip(t, 0.0, 1.0)
        nr = v0[0] + t * dv[0]; nc = v0[1] + t * dv[1]
        return np.sqrt((rr_a-nr)**2 + (cc_a-nc)**2)

    def _tri_mask(rr_a, cc_a, p0_, p1_, p2_):
        def sg(px, py, ax, ay, bx, by):
            return (px - bx) * (ay - by) - (ax - bx) * (py - by)
        x_ = cc_a; y_ = rr_a
        b1 = sg(x_, y_, p0_[1], p0_[0], p1_[1], p1_[0]) < 0.0
        b2 = sg(x_, y_, p1_[1], p1_[0], p2_[1], p2_[0]) < 0.0
        b3 = sg(x_, y_, p2_[1], p2_[0], p0_[1], p0_[0]) < 0.0
        return (b1 == b2) & (b2 == b3)

    tri_count = 0
    for tri in accepted_triangles:
        p0_, p1_, p2_ = pts43[tri[0]], pts43[tri[1]], pts43[tri[2]]
        area = _area43(p0_, p1_, p2_)
        if area < 5:
            continue
        tri_count += 1
        d01 = _dist_seg(rr_, cc_, p0_, p1_)
        d12 = _dist_seg(rr_, cc_, p1_, p2_)
        d20 = _dist_seg(rr_, cc_, p2_, p0_)
        d_edge = np.minimum(np.minimum(d01, d12), d20)
        boundary_field += np.exp(-(d_edge**2) / (2 * 2.6**2))
        inside = _tri_mask(rr_, cc_, p0_, p1_, p2_).astype(float)
        cn = (p0_ + p1_ + p2_) / 3.0
        d_cent = np.sqrt((rr_ - cn[0])**2 + (cc_ - cn[1])**2)
        interior_sigma = max(6.0, np.sqrt(area) * 0.9)
        interior_field += np.exp(-(d_cent**2) / (2 * interior_sigma**2)) * inside
        for pp in (p0_, p1_, p2_):
            dv_ = np.sqrt((rr_ - pp[0])**2 + (cc_ - pp[1])**2)
            vertex_field += np.exp(-(dv_**2) / (2 * 2.2**2))

    boundary_field = normalize(boundary_field)
    interior_field = normalize(interior_field)
    vertex_field   = normalize(vertex_field)
    anti_boundary = normalize(1.0 - boundary_field)
    structure_field = normalize(
        0.50 * boundary_field + 0.25 * interior_field +
        0.15 * vertex_field   + 0.10 * src43
    )
    translated_boundary = normalize(
        0.65 * _smooth4(boundary_field, rounds=8) +
        0.35 * _smooth4(interior_field, rounds=8)
    )
    reconstructed_structure = normalize(
        0.45 * translated_boundary + 0.30 * structure_field +
        0.15 * _smooth4(src43, rounds=6) + 0.10 * anti_boundary
    )
    regrown = reconstructed_structure.copy()
    for _ in range(20):    # 30 originally — reduced for time
        regrown = normalize(
            0.72 * regrown + 0.18 * _smooth4(regrown, rounds=1) +
            0.10 * structure_field
        )

    # cell 45: per-triangle curves + collapse criterion
    residual45_field = normalize(np.abs(src43 - _smooth4(src43, rounds=10)))
    gy45, gx45 = np.gradient(residual45_field)

    def _furthest_diag(tri):
        ti = tuple(int(x) for x in tri)
        tp = pts43[list(ti)]
        ctri = (tp[0] + tp[1] + tp[2]) / 3.0
        dcen = [float(np.linalg.norm(p - center43)) for p in tp]
        apex = tp[int(np.argmax(dcen))]
        dd = apex - ctri; nd = float(np.linalg.norm(dd)) + 1e-12
        dd = dd / nd
        best_j = None; best_s = -np.inf
        for j in range(len(pts43)):
            if j in ti:
                continue
            q = pts43[j]; v_ = q - ctri; d_ = float(np.linalg.norm(v_))
            if d_ < 1e-6:
                continue
            u_ = v_ / d_
            align = float(np.dot(u_, dd))
            sc = 1.5 * d_ + 70.0 * align
            if align > 0.25 and sc > best_s:
                best_s = sc; best_j = j
        return best_j, ctri, dd

    def _closest_across45(j_d, ref_dir):
        if j_d is None:
            return None
        q0 = pts43[j_d]; v0 = q0 - center43; nv0 = float(np.linalg.norm(v0)) + 1e-12
        best_k = None; best_s = np.inf
        for k in range(len(pts43)):
            if k == j_d:
                continue
            q = pts43[k]; v_ = q - center43; nv = float(np.linalg.norm(v_)) + 1e-12
            a0 = float(np.dot(v0, ref_dir)); a1 = float(np.dot(v_, ref_dir))
            if a0 * a1 > 0:
                continue
            sc = float(np.linalg.norm(q - q0)) + 20.0 * (float(np.dot(v0, v_)/(nv0*nv)) + 1.0)
            if sc < best_s:
                best_s = sc; best_k = k
        return best_k

    curve_list = []; curve_info = []
    for tri in accepted_triangles:
        j_d, ctri, dd = _furthest_diag(tri)
        if j_d is None:
            continue
        k_a = _closest_across45(j_d, dd)
        if k_a is None:
            continue
        p_c = ctri; p_d = pts43[j_d]; p_a = pts43[k_a]
        # curve 1
        mid1 = 0.5 * (p_c + p_d)
        flow1 = np.array([_bilinear_sample(gy45, mid1[0], mid1[1]),
                          _bilinear_sample(gx45, mid1[0], mid1[1])])
        if float(np.linalg.norm(flow1)) < 1e-6:
            v_ = p_d - p_c; flow1 = np.array([-v_[1], v_[0]])
        flow1 = flow1 / (float(np.linalg.norm(flow1)) + 1e-12)
        ctrl1 = mid1 + 0.18 * float(np.linalg.norm(p_d - p_c)) * flow1
        curve_list.append(_bezier_quad(p_c, p_d, ctrl1, n=80))
        curve_info.append({"kind": "center_to_diag", "tri": tri})
        # curve 2
        mid2 = 0.5 * (p_d + p_a)
        flow2 = np.array([_bilinear_sample(gy45, mid2[0], mid2[1]),
                          _bilinear_sample(gx45, mid2[0], mid2[1])])
        if float(np.linalg.norm(flow2)) < 1e-6:
            v_ = p_a - p_d; flow2 = np.array([-v_[1], v_[0]])
        flow2 = flow2 / (float(np.linalg.norm(flow2)) + 1e-12)
        ctrl2 = mid2 + 0.15 * float(np.linalg.norm(p_a - p_d)) * flow2
        curve_list.append(_bezier_quad(p_d, p_a, ctrl2, n=80))
        curve_info.append({"kind": "diag_to_across", "tri": tri})

    # collapse criterion
    collapse_score = 0.0; var_ratio = 0.0; mean_perp_d = 0.0
    if len(curve_list) > 0:
        cloud_rc = np.vstack(curve_list)
        cloud_xy = np.column_stack([cloud_rc[:, 1], -cloud_rc[:, 0]])
        mean_xy = cloud_xy.mean(axis=0)
        Xc_ = cloud_xy - mean_xy
        cov_ = Xc_.T @ Xc_ / max(len(Xc_) - 1, 1)
        ev, evec = np.linalg.eigh(cov_)
        ord_ = np.argsort(ev)[::-1]; ev = ev[ord_]; evec = evec[:, ord_]
        pdir = evec[:, 0]; pdir = pdir / (float(np.linalg.norm(pdir)) + 1e-12)
        var_ratio = float(ev[0] / (float(np.sum(ev)) + 1e-12))
        perp = np.array([
            float(np.linalg.norm((p - mean_xy) - float(np.dot(p - mean_xy, pdir)) * pdir))
            for p in cloud_xy
        ])
        mean_perp_d = float(np.mean(perp))
        scale = float(np.max(np.linalg.norm(cloud_xy - mean_xy, axis=1))) + 1e-12
        collapse_score = float(var_ratio / (1.0 + mean_perp_d / scale))
    integration_complete = bool(collapse_score > 0.70)

    # ============================================================
    # ===== ISOLATE EXTERNAL LIGHT SOURCE (cells 48 → 49 → 50 → 51 → 53) =====
    # Cell 48: residual collapse → weighted_peak, tangent
    # Cell 49: reverse triangulation around weighted_peak → reverse_triangles
    # Cell 50: gap_signature from outer/inner triangle gap
    # Cell 51: curves_cubed_soft, gap_plus_curves3, curves3_peak,
    #          reconstructed_gap_curves3
    # Cell 53: depth + perspective + collapse to peak+seed → final_point
    # ============================================================

    # ---------- Cell 48: residual collapse → weighted_peak + tangent ----------
    # Build curve density from forward curves (cell 45 curve_list — proxy for
    # cell 47's reverse_curves) + residual reconstruction substrate.
    cell48_curve_density = np.zeros((H, W), dtype=float)
    for B in curve_list:
        for q in B:
            r_ = int(np.clip(round(q[0]), 0, H-1))
            c_ = int(np.clip(round(q[1]), 0, W-1))
            cell48_curve_density[r_, c_] += 1.0
    cell48_curve_density = normalize(_smooth4(cell48_curve_density, rounds=6))
    resid0 = regrown
    residual_hotspot = normalize(0.55 * resid0 + 0.45 * cell48_curve_density)
    residual_hotspot = normalize(residual_hotspot ** 1.35)
    # top 1.5% of the distribution, weighted-averaged → weighted_peak
    q_top = 0.985
    thr_top = float(np.quantile(residual_hotspot, q_top))
    top_mask = residual_hotspot >= thr_top
    top_pts = np.argwhere(top_mask).astype(float)
    if len(top_pts) > 0:
        w_top = residual_hotspot[top_mask]
        w_top = w_top / (float(np.sum(w_top)) + 1e-12)
        weighted_peak = np.sum(top_pts * w_top[:, None], axis=0)
    else:
        idx_p = int(np.argmax(residual_hotspot.ravel()))
        weighted_peak = np.array(np.unravel_index(idx_p, residual_hotspot.shape), dtype=float)
    RP = weighted_peak.copy()
    # tangent from PCA of 20 nearest curve sample points
    if len(curve_list) > 0:
        all_curve_pts = np.vstack(curve_list).astype(float)
        d_cp = np.linalg.norm(all_curve_pts - weighted_peak[None, :], axis=1)
        nearest_ids = np.argsort(d_cp)[:min(20, len(all_curve_pts))]
        near = all_curve_pts[nearest_ids]
        if len(near) >= 2:
            mean_n = near.mean(axis=0)
            X_n = near - mean_n
            cov_t = X_n.T @ X_n / max(len(X_n) - 1, 1)
            ev_t, evec_t = np.linalg.eigh(cov_t)
            order_t = np.argsort(ev_t)[::-1]
            tangent = evec_t[:, order_t[0]]
        else:
            tangent = np.array([0.0, 1.0])
    else:
        tangent = np.array([0.0, 1.0])
    tangent = tangent / (float(np.linalg.norm(tangent)) + 1e-12)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    reverse_map = regrown

    # ---------- Cell 49: reverse triangulation around RP ----------
    reverse_triangles = []
    reverse_terminal_triangle = None
    all_reverse_curves = []
    reverse_curve_field = np.zeros((H, W), dtype=float)
    if len(pts43) >= 3:
        d_to_peak = np.linalg.norm(pts43 - RP[None, :], axis=1)
        support_arr = np.array([_bilinear_sample(reverse_map, p[0], p[1]) for p in pts43])
        d_thr = float(np.quantile(d_to_peak, 0.45))
        s_thr = float(np.quantile(support_arr, 0.50))
        cand_idx = np.where((d_to_peak <= d_thr) & (support_arr >= s_thr))[0]
        if len(cand_idx) < 3:
            cand_idx = np.argsort(d_to_peak)[:min(6, len(pts43))]
        cand_pts = pts43[cand_idx]
        order_c = np.argsort(np.linalg.norm(cand_pts - RP[None, :], axis=1))
        cand_pts2 = cand_pts[order_c]

        chosen = []
        for pp in cand_pts2:
            if len(chosen) == 0:
                chosen.append(pp)
            elif len(chosen) == 1:
                if float(np.linalg.norm(pp - chosen[0])) > 1e-6:
                    chosen.append(pp); break
            else:
                break

        def _tri_area_local(a, b, c):
            return 0.5 * abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))

        if len(chosen) >= 2:
            R1 = RP.copy(); R2 = chosen[0].copy(); R3 = chosen[1].copy()
            if _tri_area_local(R1, R2, R3) < 1.0:
                R2 = RP + 6.0 * tangent
                R3 = RP + 6.0 * normal
            reverse_triangle_1 = np.vstack([R1, R2, R3])
            reverse_triangles.append(reverse_triangle_1.copy())

            tri_ = reverse_triangle_1
            edges = [(0,1), (1,2), (2,0)]
            edge_lengths = [float(np.linalg.norm(tri_[i] - tri_[j])) for i, j in edges]
            imax = int(np.argmax(edge_lengths))
            edge_A = tri_[edges[imax][0]].copy()
            edge_B = tri_[edges[imax][1]].copy()

            rev_thr = float(np.quantile(reverse_map, 0.70))
            all_d2 = np.linalg.norm(pts43 - RP[None, :], axis=1)
            near_order = np.argsort(all_d2)
            extra_pts = []
            for idx in near_order:
                pp = pts43[idx]
                if (float(np.linalg.norm(pp - R1)) < 1e-6 or
                    float(np.linalg.norm(pp - R2)) < 1e-6 or
                    float(np.linalg.norm(pp - R3)) < 1e-6):
                    continue
                if _bilinear_sample(reverse_map, pp[0], pp[1]) >= rev_thr:
                    extra_pts.append(pp.copy())
                if len(extra_pts) >= 3:
                    break

            for Pnew in extra_pts:
                Tnew = np.vstack([edge_A, edge_B, Pnew])
                if _tri_area_local(Tnew[0], Tnew[1], Tnew[2]) < 1.0:
                    continue
                reverse_triangles.append(Tnew.copy())
                e2 = [(0,1), (1,2), (2,0)]
                best_score = -np.inf; best_edge = None
                for a_, b_ in e2:
                    mid_e = 0.5 * (Tnew[a_] + Tnew[b_])
                    s_ = float(np.linalg.norm(mid_e - RP))
                    if s_ > best_score:
                        best_score = s_; best_edge = (a_, b_)
                edge_A = Tnew[best_edge[0]].copy()
                edge_B = Tnew[best_edge[1]].copy()

            local_scale = float(np.median(np.linalg.norm(cand_pts - RP[None, :], axis=1)))
            if not np.isfinite(local_scale) or local_scale < 2.0:
                local_scale = 6.0
            ss = 0.45 * local_scale
            rt1 = RP + ss * tangent
            rt2 = RP - 0.8 * ss * tangent + 0.65 * ss * normal
            rt3 = RP - 0.8 * ss * tangent - 0.65 * ss * normal
            reverse_terminal_triangle = np.vstack([rt1, rt2, rt3])

            gy_rev, gx_rev = np.gradient(reverse_map)
            for T in reverse_triangles:
                centroid = np.mean(T, axis=0)
                mid_ = 0.5 * (centroid + RP)
                flow_ = np.array([
                    _bilinear_sample(gy_rev, mid_[0], mid_[1]),
                    _bilinear_sample(gx_rev, mid_[0], mid_[1])
                ])
                if float(np.linalg.norm(flow_)) < 1e-6:
                    flow_ = normal.copy()
                flow_ = -flow_ / (float(np.linalg.norm(flow_)) + 1e-12)
                ctrl_ = mid_ + 0.18 * float(np.linalg.norm(centroid - RP)) * flow_
                B = _bezier_quad(centroid, RP, ctrl_, n=80)
                all_reverse_curves.append(B)
                for q in B:
                    r_ = int(np.clip(round(q[0]), 0, H-1))
                    c_ = int(np.clip(round(q[1]), 0, W-1))
                    reverse_curve_field[r_, c_] += 1.0

            rtc = np.mean(reverse_terminal_triangle, axis=0)
            mid_t = 0.5 * (rtc + RP)
            flow_t = np.array([
                _bilinear_sample(gy_rev, mid_t[0], mid_t[1]),
                _bilinear_sample(gx_rev, mid_t[0], mid_t[1])
            ])
            if float(np.linalg.norm(flow_t)) < 1e-6:
                flow_t = tangent.copy()
            flow_t = -flow_t / (float(np.linalg.norm(flow_t)) + 1e-12)
            ctrl_t = mid_t + 0.16 * float(np.linalg.norm(rtc - RP)) * flow_t
            Bterm = _bezier_quad(rtc, RP, ctrl_t, n=80)
            all_reverse_curves.append(Bterm)
            for q in Bterm:
                r_ = int(np.clip(round(q[0]), 0, H-1))
                c_ = int(np.clip(round(q[1]), 0, W-1))
                reverse_curve_field[r_, c_] += 1.0
            reverse_curve_field = normalize(_smooth4(reverse_curve_field, rounds=4))

    # ---------- Cell 50: real gap_signature from outer/inner triangle gap ----------
    gap_signature = np.zeros((H, W), dtype=float)
    final_gap_model = np.zeros((H, W), dtype=float)
    outer_tri_50 = None; inner_tri_50 = None
    if len(reverse_triangles) > 0 and reverse_terminal_triangle is not None:
        inner_tri_50 = reverse_terminal_triangle.copy()
        all_rev_vertices = np.vstack(reverse_triangles)
        d_rv = np.linalg.norm(all_rev_vertices - RP[None, :], axis=1)
        order_rv = np.argsort(d_rv)[::-1]
        outer_cand = []
        for idx in order_rv:
            p = all_rev_vertices[idx]
            ok = True
            for q in outer_cand:
                if float(np.linalg.norm(p - q)) < 3.0:
                    ok = False; break
            if ok:
                outer_cand.append(p.copy())
            if len(outer_cand) == 3:
                break
        if len(outer_cand) < 3:
            c_inner = inner_tri_50.mean(axis=0)
            outer_cand = [c_inner + 1.8 * (p - c_inner) for p in inner_tri_50]
        outer_tri_50 = np.array(outer_cand, dtype=float)
        c_outer = outer_tri_50.mean(axis=0)
        angles_o = np.arctan2(outer_tri_50[:, 0] - c_outer[0], outer_tri_50[:, 1] - c_outer[1])
        outer_tri_50 = outer_tri_50[np.argsort(angles_o)]

        # rasterize triangle masks (vectorised barycentric)
        def _bary_mask(A, B, C):
            def sg(px, py, ax, ay, bx, by):
                return (px - bx) * (ay - by) - (ax - bx) * (py - by)
            x_ = cc_; y_ = rr_
            b1 = sg(x_, y_, A[1], A[0], B[1], B[0]) < 0.0
            b2 = sg(x_, y_, B[1], B[0], C[1], C[0]) < 0.0
            b3 = sg(x_, y_, C[1], C[0], A[1], A[0]) < 0.0
            return (b1 == b2) & (b2 == b3)

        outer_mask50 = _bary_mask(outer_tri_50[0], outer_tri_50[1], outer_tri_50[2])
        inner_mask50 = _bary_mask(inner_tri_50[0], inner_tri_50[1], inner_tri_50[2])
        gap_mask50 = outer_mask50 & (~inner_mask50)
        gap_field50 = normalize(_smooth4(gap_mask50.astype(float), rounds=6))

        AP_y = rr_ - RP[0]; AP_x = cc_ - RP[1]
        proj_n = AP_y * normal[0] + AP_x * normal[1]
        axis_weight = normalize(np.exp(-(proj_n**2) / (2 * 7.0**2)))

        gap_signature = normalize(0.65 * gap_field50 + 0.35 * (gap_field50 * axis_weight))

        inner_centroid_50 = inner_tri_50.mean(axis=0)
        bridge_dist50 = np.sqrt((rr_ - inner_centroid_50[0])**2 + (cc_ - inner_centroid_50[1])**2)
        bridge_kernel50 = normalize(np.exp(-(bridge_dist50**2) / (2 * 5.0**2)))
        center_fill = normalize(0.55 * gap_signature + 0.45 * bridge_kernel50)
        final_gap_model = normalize(0.45 * src43 + 0.35 * gap_signature + 0.20 * center_fill)
    else:
        # fallback: blank gap_signature
        center_fill = np.zeros((H, W), dtype=float)
        final_gap_model = src43.copy()

    # ---------- Cell 51: curves_cubed_soft → gap_plus_curves3 → curves3_peak ----------
    gap_signature_n = normalize(gap_signature.copy())
    curve_field_n = normalize(reverse_curve_field.copy())
    curve_gap = normalize(curve_field_n * gap_signature_n)
    curves_cubed = normalize(curve_gap ** 3)
    curves_cubed_soft = normalize(_smooth4(curves_cubed, rounds=6))
    gap_plus_curves3 = normalize(0.60 * gap_signature_n + 0.40 * curves_cubed_soft)
    gap_plus_curves3_strong = normalize(0.45 * gap_signature_n + 0.55 * curves_cubed_soft)
    peak_idx_51 = int(np.argmax(gap_plus_curves3.ravel()))
    curves3_peak = np.array(np.unravel_index(peak_idx_51, gap_plus_curves3.shape), dtype=float)
    reconstructed_gap_curves3 = normalize(
        0.35 * src43 + 0.35 * gap_signature_n + 0.30 * curves_cubed_soft
    )

    # ---------- Cell 53: depth + perspective + collapse to peak+seed ----------
    # Now uses the REAL gap_signature and curves_cubed_soft as `curve3`,
    # and reconstructed_gap_curves3 as the `recon` field.
    curve3 = curves_cubed_soft
    seed_pt = curves3_peak.copy()
    cell53_recon = reconstructed_gap_curves3

    final_point = RP.copy()
    depth_field = np.zeros((H, W), dtype=float)
    perspective_field = np.zeros((H, W), dtype=float)
    intersection = np.zeros((H, W), dtype=float)
    collapsed_peak_seed = np.zeros((H, W), dtype=float)
    if float(np.linalg.norm(seed_pt - RP)) >= 2.0:
        peak53 = RP; seed53 = seed_pt
        axis_vec = seed53 - peak53
        axis_len = float(np.linalg.norm(axis_vec))
        axis_dir = axis_vec / (axis_len + 1e-12)
        axis_perp = np.array([-axis_dir[1], axis_dir[0]])
        mid_axis = 0.5 * (peak53 + seed53)
        dpar  = (rr_ - mid_axis[0]) * axis_dir[0]  + (cc_ - mid_axis[1]) * axis_dir[1]
        dperp = (rr_ - mid_axis[0]) * axis_perp[0] + (cc_ - mid_axis[1]) * axis_perp[1]

        dist_peak = np.sqrt((rr_ - peak53[0])**2 + (cc_ - peak53[1])**2)
        dist_seed = np.sqrt((rr_ - seed53[0])**2 + (cc_ - seed53[1])**2)
        depth_raw = normalize(
            0.45 * normalize(dist_peak) + 0.45 * normalize(dist_seed) +
            0.10 * normalize(np.abs(dperp))
        )
        depth_field = normalize(_smooth4(normalize(depth_raw * cell53_recon), rounds=3))

        perspective_raw = normalize(
            np.exp(-(dperp**2) / (2 * (0.18 * max(H, W))**2)) *
            normalize(np.abs(dpar))
        )
        perspective_field = normalize(_smooth4(normalize(perspective_raw * gap_signature), rounds=3))

        forward_lift = normalize(0.50 * cell53_recon + 0.30 * depth_field + 0.20 * curve3)
        reverse_lift = normalize(0.50 * gap_signature + 0.30 * perspective_field + 0.20 * curve3)
        intersection = normalize(_smooth4(normalize(forward_lift * reverse_lift), rounds=4))

        # bridge mask along the peak-seed segment
        AB = seed53 - peak53; AB2 = float(np.dot(AB, AB)) + 1e-12
        AP_y = rr_ - peak53[0]; AP_x = cc_ - peak53[1]
        t_seg = (AP_y * AB[0] + AP_x * AB[1]) / AB2
        t_seg = np.clip(t_seg, 0.0, 1.0)
        closest_y = peak53[0] + t_seg * AB[0]
        closest_x = peak53[1] + t_seg * AB[1]
        dist_seg = np.sqrt((rr_ - closest_y)**2 + (cc_ - closest_x)**2)
        bridge_mask = normalize(np.exp(-(dist_seg**2) / (2 * 6.0**2)))
        bridge_fill = normalize(bridge_mask * intersection)
        gap_removed = normalize(0.55 * intersection + 0.45 * bridge_fill)

        peak_kernel = normalize(np.exp(-((rr_ - peak53[0])**2 + (cc_ - peak53[1])**2) / (2 * 5.0**2)))
        seed_kernel = normalize(np.exp(-((rr_ - seed53[0])**2 + (cc_ - seed53[1])**2) / (2 * 5.0**2)))
        collapsed_peak_seed = normalize(_smooth4(normalize(
            0.40 * gap_removed + 0.30 * bridge_mask +
            0.15 * peak_kernel + 0.15 * seed_kernel
        ), rounds=4))

        f_idx = int(np.argmax(collapsed_peak_seed.ravel()))
        final_point = np.array(np.unravel_index(f_idx, collapsed_peak_seed.shape), dtype=float)

    # ============================================================
    # ===== MASTER GLOBAL-CENTERED — picture-quality final output =====
    # Global field = master_output, seed field = reconstructed_gap_curves3.
    # Produces closure_centre_field, master_constraint_form,
    # master_unfolded_field, master_regrowth — the picture-quality flower.
    # ============================================================
    def _weighted_centre(field):
        f = normalize(field)
        w = f + 1e-12
        r0 = float(np.sum(rr_ * w) / np.sum(w))
        c0 = float(np.sum(cc_ * w) / np.sum(w))
        return np.array([r0, c0], dtype=float)

    def _weighted_cov(field, centre):
        f = normalize(field)
        dr = rr_ - centre[0]; dc = cc_ - centre[1]
        w_ = f / (float(np.sum(f)) + 1e-12)
        crr = float(np.sum(w_ * dr * dr))
        crc = float(np.sum(w_ * dr * dc))
        ccc = float(np.sum(w_ * dc * dc))
        cov = np.array([[crr, crc], [crc, ccc]], dtype=float)
        ev, evec = np.linalg.eigh(cov)
        order_ = np.argsort(ev)[::-1]
        ev = ev[order_]; evec = evec[:, order_]
        main = evec[:, 0]; cross = evec[:, 1]
        main = main / (float(np.linalg.norm(main)) + 1e-12)
        cross = cross / (float(np.linalg.norm(cross)) + 1e-12)
        return main, cross, float(np.sqrt(max(ev[0], 1e-8))), float(np.sqrt(max(ev[1], 1e-8)))

    def _top_mask(field, q=0.92):
        f = normalize(field)
        return (f >= float(np.quantile(f, q))).astype(float)

    master_global_field = master_output
    master_seed_field = reconstructed_gap_curves3

    global_centre = _weighted_centre(master_global_field)
    seed_centre_local = _weighted_centre(master_seed_field)
    master_agreement = normalize(master_seed_field * master_global_field)
    g_support = normalize(_smooth4(master_global_field, rounds=8))
    s_support = normalize(_smooth4(master_seed_field, rounds=8))
    support_agreement = normalize(g_support * s_support)

    closure_centre_field = normalize(_smooth4(normalize(
        0.30 * master_seed_field + 0.25 * master_global_field +
        0.25 * master_agreement   + 0.20 * support_agreement
    ), rounds=8))
    system_centre = _weighted_centre(closure_centre_field)
    centre_main, centre_cross, sig_main, sig_cross = _weighted_cov(closure_centre_field, system_centre)

    d_main  = (rr_ - system_centre[0]) * centre_main[0]  + (cc_ - system_centre[1]) * centre_main[1]
    d_cross = (rr_ - system_centre[0]) * centre_cross[0] + (cc_ - system_centre[1]) * centre_cross[1]
    radial = np.sqrt((rr_ - system_centre[0])**2 + (cc_ - system_centre[1])**2)

    seed_projected_m = normalize(_smooth4(normalize(
        0.55 * master_seed_field + 0.25 * master_agreement + 0.20 * closure_centre_field
    ), rounds=6))

    period_main  = max(6.0, 1.35 * sig_main)
    period_cross = max(6.0, 1.35 * sig_cross)
    rhythm_main  = 0.5 * (1.0 + np.cos(2 * np.pi * d_main  / period_main))
    rhythm_cross = 0.5 * (1.0 + np.cos(2 * np.pi * d_cross / period_cross))

    # NEW — angular rhythm = n_lobes-fold petal pattern around the system centre
    theta_centre = np.arctan2(rr_ - system_centre[0], cc_ - system_centre[1])
    # phase = angle of the first attractor relative to centre, so petals align
    # with the engine's lobes rather than the absolute frame
    if len(inward_pts) > 0:
        first_attr = inward_pts[0]
        phase_offset = float(np.arctan2(first_attr[0] - system_centre[0],
                                        first_attr[1] - system_centre[1]))
    else:
        phase_offset = 0.0
    angular_rhythm = 0.5 * (1.0 + np.cos(n_lobes * (theta_centre - phase_offset)))
    # crisper petals than v28 — but still not starburst (was 1.3, now 1.7)
    angular_rhythm_soft = angular_rhythm ** 1.7
    # wider inner cutoff so petals truly bloom from the outside
    inner_cutoff = 1.0 - np.exp(-(radial**2) / (2 * (0.6 * sig_main)**2))
    angular_petals = angular_rhythm_soft * inner_cutoff

    elliptic_pref = np.exp(
        -(d_main**2 / (2 * (1.8 * sig_main)**2) +
          d_cross**2 / (2 * (1.8 * sig_cross)**2))
    )
    boundary_memory = normalize(
        0.50 * _smooth4(_top_mask(master_seed_field, q=0.93), rounds=6) +
        0.50 * _smooth4(_top_mask(master_global_field, q=0.93), rounds=6)
    )

    master_constraint_form = normalize(_smooth4(normalize(
        0.28 * elliptic_pref +
        0.10 * elliptic_pref * rhythm_main +
        0.08 * elliptic_pref * rhythm_cross +
        0.30 * elliptic_pref * angular_petals +   # PETALS — now annular-biased
        0.14 * boundary_memory +
        0.10 * seed_projected_m
    ), rounds=8))

    master_root_support = normalize(_smooth4(normalize(
        0.40 * closure_centre_field + 0.35 * master_agreement +
        0.25 * master_constraint_form
    ), rounds=8))

    # === ANNULAR envelope — bloom pushed further out (was 1.6·σ, now 2.2·σ) ===
    r_peak  = 2.2 * sig_main           # ring radius where petals bloom
    r_sigma = 1.0 * sig_main           # ring width (slightly wider for softness)
    env_annular = np.exp(-((radial - r_peak)**2) / (2 * r_sigma**2))

    # Much dimmer central heart (was 0.10, now 0.03)
    heart_kernel = np.exp(-(radial**2) / (2 * (0.65 * sig_main)**2))

    axis_channel  = np.exp(-(d_cross**2) / (2 * (1.15 * sig_cross)**2))
    cross_channel = np.exp(-(d_main**2)  / (2 * (1.15 * sig_main )**2))
    petal_channel = normalize(env_annular * angular_petals)
    master_expansion = normalize(_smooth4(normalize(
        0.03 * heart_kernel +                  # very dim centre
        0.20 * env_annular +                   # annular bloom backdrop
        0.14 * axis_channel  * rhythm_main +
        0.10 * cross_channel * rhythm_cross +
        0.36 * petal_channel +                 # PETAL expansion (more weight)
        0.14 * master_constraint_form
    ), rounds=6))

    master_unfolded_field = normalize(_smooth4(normalize(
        0.45 * master_constraint_form * master_expansion +
        0.20 * seed_projected_m +
        0.20 * closure_centre_field +
        0.15 * master_global_field
    ), rounds=8))

    collapsed_root = normalize(_smooth4(normalize(
        0.55 * master_unfolded_field +
        0.25 * master_constraint_form +
        0.20 * master_root_support
    ), rounds=8))
    rc_idx = int(np.argmax(collapsed_root.ravel()))
    master_root_candidate = np.array(np.unravel_index(rc_idx, collapsed_root.shape), dtype=float)

    root_dist = np.sqrt((rr_ - master_root_candidate[0])**2 + (cc_ - master_root_candidate[1])**2)
    master_root_kernel = normalize(_smooth4(normalize(
        np.exp(-(root_dist**2) / (2 * 5.0**2)) * collapsed_root
    ), rounds=6))

    # === REGROWTH = the picture-quality flower ===
    # Multiply the centred lobe blend by the inner cutoff so it doesn't
    # contribute to centre brightness — pushes the bloom outward.
    lobes_outward = inward_defined_home * inner_cutoff
    master_regrowth = normalize(_smooth4(normalize(
        0.18 * master_root_kernel +
        0.40 * master_expansion +            # annular bloom dominates more
        0.18 * master_constraint_form +
        0.24 * lobes_outward                 # lobes with centre suppressed
    ), rounds=7))
    master_regrowth = normalize(master_regrowth ** 0.72)   # was 0.80 — sharper contrast

    return dict(
        # Core
        X=X.copy(),
        n_lobes=int(n_lobes),
        # Stage 1 — full chain intermediates (inward, flower, master transfer)
        inward_defined_home=inward_defined_home.copy(),
        single_point_flower=single_point_flower.copy(),
        constraint_layer=constraint_layer.copy(),
        heat_core=heat_core.copy(),
        transfer_fabric=transfer_fabric.copy(),
        implanted_heat=implanted_heat.copy(),
        outward_field=outward_field.copy(),
        spent_energy=spent_energy.copy(),
        bypass_field=bypass_field.copy(),
        transfer_layer=transfer_layer.copy(),
        master_output=master_output.copy(),
        inward_pts=inward_pts.copy(),
        center_mass=center_mass.copy(),
        # Cells 43/44/45 outputs
        center43=center43.copy(),
        pts43=pts43.copy(),
        accepted_triangles=[tuple(int(x) for x in t) for t in accepted_triangles],
        boundary_field=boundary_field.copy(),
        interior_field=interior_field.copy(),
        vertex_field=vertex_field.copy(),
        translated_boundary=translated_boundary.copy(),
        reconstructed_structure=reconstructed_structure.copy(),
        regrown=regrown.copy(),
        curve_list=[c.copy() for c in curve_list],
        collapse_score=float(collapse_score),
        var_ratio=float(var_ratio),
        mean_perp_d=float(mean_perp_d),
        integration_complete=bool(integration_complete),
        tri_count=int(tri_count),
        # Cells 49 + 53: external-light-source isolation
        ext_RP=RP.copy(),
        ext_tangent=tangent.copy(),
        ext_seed_pt=seed_pt.copy(),
        ext_final_point=final_point.copy(),
        ext_depth_field=depth_field.copy(),
        ext_perspective_field=perspective_field.copy(),
        ext_intersection=intersection.copy(),
        ext_collapsed_peak_seed=collapsed_peak_seed.copy(),
        ext_reverse_triangles=[t.copy() for t in reverse_triangles],
        ext_reverse_terminal_triangle=(
            reverse_terminal_triangle.copy() if reverse_terminal_triangle is not None else None
        ),
        ext_all_reverse_curves=[c.copy() for c in all_reverse_curves],
        ext_reverse_curve_field=reverse_curve_field.copy(),
        ext_curve3=curve3.copy(),
        ext_gap_signature=gap_signature.copy(),
        # Cell 48 outputs
        ext_residual_hotspot=residual_hotspot.copy(),
        ext_cell48_curve_density=cell48_curve_density.copy(),
        # Cell 50 outputs
        ext_outer_tri_50=(outer_tri_50.copy() if outer_tri_50 is not None else None),
        ext_inner_tri_50=(inner_tri_50.copy() if inner_tri_50 is not None else None),
        ext_gap_signature_real=gap_signature.copy(),
        ext_final_gap_model=final_gap_model.copy(),
        ext_center_fill=center_fill.copy(),
        # Cell 51 outputs
        ext_curves_cubed_soft=curves_cubed_soft.copy(),
        ext_gap_plus_curves3=gap_plus_curves3.copy(),
        ext_gap_plus_curves3_strong=gap_plus_curves3_strong.copy(),
        ext_curves3_peak=curves3_peak.copy(),
        ext_reconstructed_gap_curves3=reconstructed_gap_curves3.copy(),
        # MASTER GLOBAL-CENTERED — picture-quality outputs
        master_global_field=master_global_field.copy(),
        master_seed_field=master_seed_field.copy(),
        master_global_centre=global_centre.copy(),
        master_system_centre=system_centre.copy(),
        master_centre_main=centre_main.copy(),
        master_centre_cross=centre_cross.copy(),
        master_closure_centre_field=closure_centre_field.copy(),
        master_constraint_form=master_constraint_form.copy(),
        master_expansion=master_expansion.copy(),
        master_unfolded_field=master_unfolded_field.copy(),
        master_root_support=master_root_support.copy(),
        master_root_candidate=master_root_candidate.copy(),
        master_root_kernel=master_root_kernel.copy(),
        master_regrowth=master_regrowth.copy(),
        # The canonical picture output for the unified API
        picture=master_regrowth.copy(),
    )

# Backwards-compat name (older callers used reconstruct_geometry)
