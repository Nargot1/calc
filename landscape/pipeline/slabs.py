"""Slab pass (v3): split steep terrain faces into horizontal rock bands and push the bands in/out as slabs.

Walls (faces steeper than WALL_SLOPE) are cut by global horizontal planes at CITY_Z + k*STEP + {0, 18, 36} (jittered;
bands: base / mid / grass lip, offsets 0 / +3.5 / +4.5 studs so the wall bulges and the lip overhangs; the lip cut is a
single cut, so the lip face leans outward instead of needing a ledge strip)
and by vertical seam planes (x = const for walls facing +-y, y = const for walls facing +-x; brick-shifted per band).
Every cut is a double cut (z +- EPS) so the thin strip between the two planes becomes the ledge / seam side face once the
two bands move by different amounts. Vertices shared with uncut faces move too, and T-junctions are resolved before the
offsets are applied, so the mesh never cracks. Pure numpy, no bpy.
"""
import numpy as np

EPS = 0.05

def _split(poly, axis, value):
    """Sutherland-Hodgman split of a convex polygon (list of 3-tuples) by the axis-aligned plane coord[axis] = value.
    Returns (below, above) polygons (either may be empty)."""
    below, above = [], []
    n = len(poly)
    for i in range(n):
        p = poly[i]; q = poly[(i + 1) % n]
        dp = p[axis] - value; dq = q[axis] - value
        if dp <= 0: below.append(p)
        if dp >= 0: above.append(p)
        if (dp < 0 < dq) or (dq < 0 < dp):
            t = dp / (dp - dq)
            r = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t)
            if axis == 0: r = (value, r[1], r[2])
            elif axis == 1: r = (r[0], value, r[2])
            else: r = (r[0], r[1], value)
            below.append(r); above.append(r)
    if len(below) < 3: below = []
    if len(above) < 3: above = []
    return below, above

def _double_cut(pieces, axis, value, eps, strip_flag):
    """cut every piece (poly, flags) by the planes value-eps / value+eps -> below, strip(flagged), above"""
    out = []
    for poly, fl in pieces:
        lo = min(p[axis] for p in poly); hi = max(p[axis] for p in poly)
        if hi <= value - eps or lo >= value + eps:
            out.append((poly, fl)); continue
        if eps <= 0:
            below, above = _split(poly, axis, value)
            if below: out.append((below, fl))
            if above: out.append((above, fl))
            continue
        below, rest = _split(poly, axis, value - eps)
        strip, above = _split(rest, axis, value + eps) if rest else ([], [])
        if below: out.append((below, fl))
        if strip: out.append((strip, fl | strip_flag))
        if above: out.append((above, fl))
    return out

def _hash01(*args):
    v = 0.0
    for i, a in enumerate(args):
        v += np.asarray(a, float) * (12.9898, 78.233, 37.719, 4.1414)[i % 4]
    return np.modf(np.sin(v) * 43758.5453)[0] % 1.0

def slab_pass(co, tris, *, city_z, step, x0, y0, nx, ny, cell, wall_slope=55.0, z_min=-50.0,
              band_offsets=(0.0, 18.0, 36.0), band_jitter=(0.0, 2.0, 1.0), band_strip=(True, True, False),
              deltas=(0.0, 3.5, 4.5), seam_spacing=(34.0, 50.0), seam_bands=(1,), seg_jitter=0.8, seg_alt=0.6,
              fixed=None, inward=None, allow=None, seed=5, verbose=True):
    """co (N,3) float, tris (F,3) int.  fixed(x,y,z)->bool mask of vertices that must not move; inward(x,y,z)->bool mask
    of vertices whose slabs may only move inward (city cliff under the user's beach parts); allow: per-face bool mask of faces
    that may become slabs (None = every steep face).
    Returns co2, tris2, flags dict(wall, ledge, seam, band_type, orig)."""
    rng = np.random.default_rng(seed)
    co = np.asarray(co, float); tris = np.asarray(tris, np.int64)
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    fn = np.cross(b - a, c - a); fl = np.linalg.norm(fn, axis=1); fn = fn / np.maximum(fl, 1e-12)[:, None]
    slope = np.degrees(np.arccos(np.clip(fn[:, 2], -1, 1)))
    cz = (a[:, 2] + b[:, 2] + c[:, 2]) / 3
    wall = (slope > wall_slope) & (cz > z_min) & (fl > 1e-6)
    if allow is not None: wall &= allow          # e.g. only terraced walls + the city cliff, not the free mountain faces
    nw = int(wall.sum())
    # ---------------- cut planes
    zlo, zhi = co[:, 2].min(), co[:, 2].max()
    k0 = int(np.floor((zlo - city_z) / step)) - 1; k1 = int(np.ceil((zhi - city_z) / step)) + 1
    levels = []; strips = []
    for k in range(k0, k1 + 1):
        for i, off in enumerate(band_offsets):
            levels.append(city_z + k * step + off + rng.uniform(-band_jitter[i], band_jitter[i])); strips.append(band_strip[i])
    order = np.argsort(levels); levels = np.array(levels)[order]; strips = np.array(strips)[order]
    nb = len(levels) + 1; nbt = len(band_offsets)
    # vertical seams: base sequences per family + per-band shift (brick pattern)
    def seq(lo, hi):
        pos = [lo - 40.0]
        while pos[-1] < hi + 40.0: pos.append(pos[-1] + rng.uniform(*seam_spacing))
        return np.array(pos)
    base = [seq(x0, x0 + nx * cell), seq(y0, y0 + ny * cell)]       # family 0: planes x = const (walls facing +-y); 1: y = const
    shift = rng.uniform(0, seam_spacing[1], (nb, 2))
    seams = [[base[f] + shift[bi, f] for f in range(2)] for bi in range(nb)]
    family = (np.abs(fn[:, 0]) > np.abs(fn[:, 1])).astype(int)       # 1: faces +-x -> seams are y = const planes
    seam_axis = [0, 1]
    # ---------------- clip wall triangles
    polys = []   # (vertex tuples, flags(bit0 wall, bit1 ledge strip, bit2 seam strip), orig face)
    for fi in np.nonzero(wall)[0]:
        poly = [tuple(co[tris[fi, j]]) for j in range(3)]
        pieces = [(poly, 1)]
        lo = min(p[2] for p in poly); hi = max(p[2] for p in poly)
        sel = (levels > lo - EPS) & (levels < hi + EPS)
        for zc, st in zip(levels[sel], strips[sel]):
            pieces = _double_cut(pieces, 2, float(zc), EPS if st else 0.0, 2 if st else 0)   # single cut = no ledge, the upper band just leans
        fam = family[fi]; ax = seam_axis[fam]
        out = []
        for poly_, fl_ in pieces:
            zm = sum(p[2] for p in poly_) / len(poly_)
            bi = int(np.searchsorted(levels, zm, side='right'))
            if (bi - 1) % nbt not in seam_bands or (fl_ & 2):
                out.append((poly_, fl_)); continue
            ulo = min(p[ax] for p in poly_); uhi = max(p[ax] for p in poly_)
            sp = seams[bi][fam]; sp = sp[(sp > ulo - EPS) & (sp < uhi + EPS)]
            pcs = [(poly_, fl_)]
            for uc in sp: pcs = _double_cut(pcs, ax, float(uc), EPS, 4)
            out += pcs
        for poly_, fl_ in out: polys.append((poly_, fl_, fi))
    for fi in np.nonzero(~wall)[0]:
        polys.append(([tuple(co[tris[fi, j]]) for j in range(3)], 0, fi))
    if verbose: print(f"slab pass: {nw} wall faces of {len(tris)} -> {len(polys)} polygons before T-junction fix")
    # ---------------- weld vertices
    vid = {}; verts = []
    def getv(p):
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        i = vid.get(key)
        if i is None:
            i = len(verts); vid[key] = i; verts.append(key)
        return i
    loops = []; polys2 = []
    for poly, fl_, fi in polys:
        lp = [getv(p) for p in poly]
        lp = [v for i, v in enumerate(lp) if v != lp[i - 1]]          # drop consecutive duplicates (welded slivers)
        if len(set(lp)) >= 3: loops.append(lp); polys2.append((poly, fl_, fi))
    polys = polys2
    V = np.array(verts, float)
    n_orig = len(co)
    # ---------------- T-junction resolution: cut-generated vertices lying on edges of other polygons get inserted
    orig_keys = {(round(p[0], 3), round(p[1], 3), round(p[2], 3)) for p in map(tuple, co)}
    cutv = np.array([i for i, k in enumerate(verts) if k not in orig_keys], dtype=np.int64)
    GC = 16.0
    grid = {}
    for i in cutv:
        key = (int(V[i, 0] // GC), int(V[i, 1] // GC), int(V[i, 2] // GC))
        grid.setdefault(key, []).append(i)
    # edges to check: edges of wall polygons + edges of non-wall faces that touch a wall face
    wall_vset = set()
    for lp, (_, fl_, _) in zip(loops, polys):
        if fl_ & 1: wall_vset.update(lp)
    new_loops = []
    n_ins = 0
    for lp, (_, fl_, _) in zip(loops, polys):
        check = (fl_ & 1) or any(v in wall_vset for v in lp)
        if not check or len(cutv) == 0:
            new_loops.append(lp); continue
        out = []
        n = len(lp)
        for i in range(n):
            p = lp[i]; q = lp[(i + 1) % n]
            P = V[p]; Q = V[q]; d = Q - P; L2 = float(d @ d)
            out.append(p)
            if L2 < 1e-8: continue
            lo = np.minimum(P, Q); hi = np.maximum(P, Q)
            c0 = (lo // GC).astype(int); c1 = (hi // GC).astype(int)
            if (c1 - c0).max() > 40: continue      # huge edge (sea floor / far skirt) - no walls there
            found = []
            for gx in range(c0[0], c1[0] + 1):
                for gy in range(c0[1], c1[1] + 1):
                    for gz in range(c0[2], c1[2] + 1):
                        for j in grid.get((gx, gy, gz), ()):
                            if j == p or j == q: continue
                            R = V[j] - P; t = float(R @ d) / L2
                            if t <= 1e-6 or t >= 1 - 1e-6: continue
                            perp = R - t * d
                            if float(perp @ perp) < 1e-6: found.append((t, j))
            if found:
                found.sort(); last = None
                for t, j in found:
                    if j != last: out.append(j); last = j; n_ins += 1
        new_loops.append(out)
    loops = new_loops
    if verbose: print(f"slab pass: inserted {n_ins} T-junction vertices; {len(V)} vertices")
    # ---------------- per-vertex offsets
    hn = fn[:, :2] / np.maximum(np.linalg.norm(fn[:, :2], axis=1), 1e-9)[:, None]
    dirs = np.zeros((len(V), 2)); famcnt = np.zeros((len(V), 2)); isw = np.zeros(len(V), bool)
    for lp, (_, fl_, fi) in zip(loops, polys):
        if fl_ & 1:
            for v in lp:
                dirs[v] += hn[fi]; famcnt[v, family[fi]] += 1; isw[v] = True
    dn = np.linalg.norm(dirs, axis=1); dirs = dirs / np.maximum(dn, 1e-9)[:, None]
    bi = np.searchsorted(levels, V[:, 2] - 0.01, side='right')      # a vertex exactly on a single cut belongs to the lower band
    btype = (bi - 1) % nbt
    fam_v = (famcnt[:, 1] > famcnt[:, 0]).astype(int)
    u = np.where(fam_v == 1, V[:, 1], V[:, 0])
    seg = np.zeros(len(V), int)
    for i in np.nonzero(isw)[0]:
        seg[i] = int(np.searchsorted(seams[bi[i]][fam_v[i]], u[i]))
    D = np.asarray(deltas)[btype]
    jit = (_hash01(bi, seg, fam_v) - 0.5) * 2 * seg_jitter + seg_alt * np.where(seg % 2 == 0, 1.0, -1.0) * (btype != 3)
    delta = (D + jit) * isw
    if inward is not None:
        m = inward(V[:, 0], V[:, 1], V[:, 2]) & isw
        delta = np.where(m, 1.5 * (delta - (max(deltas) + seg_jitter + seg_alt)), delta)   # recess only, 1.5x deeper: the city cliff reads as chunky slabs
    if fixed is not None:
        delta = np.where(fixed(V[:, 0], V[:, 1], V[:, 2]), 0.0, delta)
    V2 = V.copy(); V2[:, :2] += dirs * delta[:, None]
    # ---------------- triangulate (fan from the apex that maximises the smallest triangle area, pre-offset)
    tris2 = []; flags = []
    def tri_area2(p, q, r):
        return np.linalg.norm(np.cross(V[q] - V[p], V[r] - V[p]))
    for lp, (_, fl_, fi) in zip(loops, polys):
        n = len(lp)
        if n == 3:
            tris2.append(lp); flags.append((fl_, fi)); continue
        best, bi_ = -1.0, 0
        for s in range(n):
            m = min(tri_area2(lp[s], lp[(s + j) % n], lp[(s + j + 1) % n]) for j in range(1, n - 1))
            if m > best: best, bi_ = m, s
        for j in range(1, n - 1):
            tris2.append((lp[bi_], lp[(bi_ + j) % n], lp[(bi_ + j + 1) % n])); flags.append((fl_, fi))
    tris2 = np.array(tris2, np.int64); fl = np.array([f for f, _ in flags]); orig = np.array([o for _, o in flags])
    # drop triangles that are still degenerate after the offset (slivers from cuts grazing a vertex) and exact duplicates
    area2 = np.linalg.norm(np.cross(V2[tris2[:, 1]] - V2[tris2[:, 0]], V2[tris2[:, 2]] - V2[tris2[:, 0]]), axis=1)
    keep = area2 > 2e-3
    srt = np.sort(tris2, 1); _, first = np.unique(srt, axis=0, return_index=True)
    dup = np.ones(len(tris2), bool); dup[first] = False
    keep &= ~dup
    tris2, fl, orig = tris2[keep], fl[keep], orig[keep]
    zc2 = V[tris2].mean(1)[:, 2]
    out = dict(wall=(fl & 1) > 0, ledge=(fl & 2) > 0, seam=(fl & 4) > 0, orig=orig, levels=levels, delta=delta, V_pre=V)
    out['band_type'] = np.where(out['wall'], (np.searchsorted(levels, zc2, side='right') - 1) % nbt, -1)
    if verbose:
        print(f"slab pass: {len(tris2)} tris ({int(out['wall'].sum())} wall, {int(out['ledge'].sum())} ledge, {int(out['seam'].sum())} seam), "
              f"offset range {delta.min():.2f}..{delta.max():.2f}")
    return V2, tris2, out

def check_mesh(V, T):
    """edge manifold check: returns (#boundary edges, #non-manifold edges)"""
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); e.sort(1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt == 1).sum()), int((cnt > 2).sum())

if __name__ == '__main__':
    import sys, json, time
    W = sys.argv[1].rstrip('/') + '/'
    meta = json.load(open(W + 'terrain_meta.json')); H = np.load(W + 'terrain_h.npy').astype(float)
    CELL, X0, Y0 = meta['cell'], meta['x0'], meta['y0']
    cx0, cy0, cx1, cy1 = [float(v) for v in sys.argv[2:6]]
    i0, i1 = int((cx0 - X0) / CELL), int((cx1 - X0) / CELL); j0, j1 = int((cy0 - Y0) / CELL), int((cy1 - Y0) / CELL)
    h = H[j0:j1, i0:i1][::2, ::2]; ny_, nx_ = h.shape
    xs = cx0 + np.arange(nx_) * 2 * CELL; ys = cy0 + np.arange(ny_) * 2 * CELL; XX, YY = np.meshgrid(xs, ys)
    co = np.stack([XX.ravel(), YY.ravel(), h.ravel()], 1)
    idx = np.arange(nx_ * ny_).reshape(ny_, nx_)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel(); c = idx[1:, 1:].ravel(); d = idx[1:, :-1].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    print('test mesh', len(co), 'verts', len(tris), 'tris; boundary/nonmanifold before:', check_mesh(co, tris))
    t = time.time()
    V2, T2, fl = slab_pass(co, tris, city_z=meta['city'], step=meta.get('terrace_step', 45.0), x0=X0, y0=Y0, nx=meta['nx'], ny=meta['ny'], cell=CELL)
    nb_, nm_ = check_mesh(V2, T2)
    Vp = fl['V_pre']; e = np.concatenate([T2[:, [0, 1]], T2[:, [1, 2]], T2[:, [2, 0]]]); e.sort(1)
    ue, cnt = np.unique(e, axis=0, return_counts=True); bd = ue[cnt == 1]
    onb = lambda v: (abs(v[0] - xs[0]) < 1e-3) | (abs(v[0] - xs[-1]) < 1e-3) | (abs(v[1] - ys[0]) < 1e-3) | (abs(v[1] - ys[-1]) < 1e-3)
    cracks = [ed for ed in bd if not (onb(Vp[ed[0]]) and onb(Vp[ed[1]]))]
    print('slab pass took', round(time.time() - t, 1), 's; boundary edges', nb_, 'of which cracks (not on the crop border):', len(cracks), '; non-manifold edges', nm_)
    for ed in cracks[:5]: print('  crack', Vp[ed[0]].round(3), Vp[ed[1]].round(3))
    np.savez(W + 'slab_test.npz', V=V2, T=T2, wall=fl['wall'], ledge=fl['ledge'], band=fl['band_type'])
