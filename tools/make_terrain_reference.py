#!/usr/bin/env python3
"""
Build a terrain-only reference (hills and mountains, no buildings/trees/stones)
from a single-mesh GLB diorama.

Pipeline
  1. Load the GLB (positions, UVs, indices, base-color texture).
  2. Rasterize the mesh top-down (X/Z plane, Y up) into a height map (DSM).
  3. Detect objects standing on the ground at several scales with a
     morphological top-hat, keeping only candidates that have a steep rim
     (walls), a convex dome shape (tree canopies) or are tiny compact bumps
     (stones). Pits left by open-roofed buildings are detected the same way on
     the inverted height map. Near-vertical pixels are discarded as well.
  4. Fill the removed pixels by harmonic (Laplace) interpolation from the
     surrounding ground -> terrain height map (DTM).
  5. Export: 16-bit height map PNG + metadata, hill-shade preview, and a
     decimated textured terrain mesh (GLB) in the source coordinate frame.
     The texture is the top-down base color with object pixels inpainted.

Usage
  python3 tools/make_terrain_reference.py result_1793266117047.glb terrain_reference/
"""
import argparse, io, json, struct, sys, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

S8 = np.ones((3, 3), bool)

# ----------------------------------------------------------------------------- GLB I/O
def load_glb(path):
    with open(path, 'rb') as f:
        magic, ver, length = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67, 'not a GLB file'
        clen, ctype = struct.unpack('<II', f.read(8))
        js = json.loads(f.read(clen))
        blen, btype = struct.unpack('<II', f.read(8))
        bin_ = f.read(blen)

    def acc(i):
        a = js['accessors'][i]; bv = js['bufferViews'][a['bufferView']]
        ct = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a['componentType']]
        n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}[a['type']]
        off = bv['byteOffset'] + a.get('byteOffset', 0)
        arr = np.frombuffer(bin_, dtype=ct, count=a['count'] * n, offset=off)
        return arr.reshape(a['count'], n) if n > 1 else arr

    Vs, Fs, UVs = [], [], []
    off = 0
    for mesh in js['meshes']:
        for prim in mesh['primitives']:
            V = acc(prim['attributes']['POSITION']).astype(np.float64)
            F = acc(prim['indices']).reshape(-1, 3).astype(np.int64) + off
            UV = acc(prim['attributes']['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in prim['attributes'] else np.zeros((len(V), 2))
            Vs.append(V); Fs.append(F); UVs.append(UV); off += len(V)
    V, F, UV = np.vstack(Vs), np.vstack(Fs), np.vstack(UVs)
    tex = None
    try:
        mat = js['materials'][js['meshes'][0]['primitives'][0]['material']]
        ti = mat['pbrMetallicRoughness']['baseColorTexture']['index']
        t = js['textures'][ti]
        src = t.get('source', t.get('extensions', {}).get('EXT_texture_webp', {}).get('source'))
        bv = js['bufferViews'][js['images'][src]['bufferView']]
        tex = np.array(Image.open(io.BytesIO(bin_[bv['byteOffset']:bv['byteOffset'] + bv['byteLength']])).convert('RGB'))
    except Exception as e:  # texture is optional
        print('no base color texture found:', e)
    return V, F, UV, tex

# ----------------------------------------------------------------------------- rasterization
def rasterize_top(V, F, res=1024, margin_px=2, chunk=200000):
    """Top-down rasterization on the X/Z plane. Keeps the highest surface per pixel."""
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    xmin, xmax, zmin, zmax = x.min(), x.max(), z.min(), z.max()
    px = max(xmax - xmin, zmax - zmin) / (res - 2 * margin_px)
    nx = int(np.ceil((xmax - xmin) / px)) + 2 * margin_px
    nz = int(np.ceil((zmax - zmin) / px)) + 2 * margin_px
    ox, oz = xmin - margin_px * px, zmin - margin_px * px
    H = np.full((nz, nx), -np.inf); TRI = np.full((nz, nx), -1, np.int64); B = np.zeros((nz, nx, 3))
    PX, PZ = (x - ox) / px, (z - oz) / px
    tx, tz, ty = PX[F], PZ[F], y[F]
    x0 = np.floor(tx.min(1)).astype(int).clip(0, nx - 1); x1 = np.ceil(tx.max(1)).astype(int).clip(0, nx - 1)
    z0 = np.floor(tz.min(1)).astype(int).clip(0, nz - 1); z1 = np.ceil(tz.max(1)).astype(int).clip(0, nz - 1)
    w, h = x1 - x0 + 1, z1 - z0 + 1
    counts = w * h
    flatH, flatT, flatB = H.reshape(-1), TRI.reshape(-1), B.reshape(-1, 3)
    for s in range(0, len(F), chunk):
        e = min(len(F), s + chunk)
        c = counts[s:e]; tot = int(c.sum())
        tri = np.repeat(np.arange(s, e), c)
        local = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
        wr = np.repeat(w[s:e], c)
        cx, cz = x0[tri] + local % wr, z0[tri] + local // wr
        pxc, pzc = cx + 0.5, cz + 0.5
        ax, az = tx[tri, 0], tz[tri, 0]; bx, bz = tx[tri, 1], tz[tri, 1]; qx, qz = tx[tri, 2], tz[tri, 2]
        det = (bx - ax) * (qz - az) - (qx - ax) * (bz - az)
        ok = np.abs(det) > 1e-14; det = np.where(ok, det, 1.0)
        l1 = ((bx - pxc) * (qz - pzc) - (qx - pxc) * (bz - pzc)) / det
        l2 = ((qx - pxc) * (az - pzc) - (ax - pxc) * (qz - pzc)) / det
        l3 = 1 - l1 - l2
        inside = ok & (l1 >= -0.02) & (l2 >= -0.02) & (l3 >= -0.02)
        tri, cx, cz, l1, l2, l3 = tri[inside], cx[inside], cz[inside], l1[inside], l2[inside], l3[inside]
        hgt = l1 * ty[tri, 0] + l2 * ty[tri, 1] + l3 * ty[tri, 2]
        key = cz * nx + cx
        order = np.lexsort((hgt, key)); ks = key[order]
        last = np.ones(len(ks), bool); last[:-1] = ks[1:] != ks[:-1]
        sel = order[last]; kk = key[sel]
        better = hgt[sel] > flatH[kk]; kk, sel = kk[better], sel[better]
        flatH[kk] = hgt[sel]; flatT[kk] = tri[sel]; flatB[kk] = np.stack([l1[sel], l2[sel], l3[sel]], 1)
    # splat vertices too so sub-pixel triangles leave no holes
    vk = np.floor(PZ).astype(int).clip(0, nz - 1) * nx + np.floor(PX).astype(int).clip(0, nx - 1)
    np.maximum.at(flatH, vk, y)
    return dict(H=H, TRI=TRI, B=B, nx=nx, nz=nz, px=px, ox=ox, oz=oz)

def sample_texture(tex, UV, F, TRI, B):
    th, tw = tex.shape[:2]
    out = np.zeros(TRI.shape + (3,), np.uint8)
    m = TRI >= 0; tri = TRI[m]; b = B[m]
    uv = UV[F[tri, 0]] * b[:, :1] + UV[F[tri, 1]] * b[:, 1:2] + UV[F[tri, 2]] * b[:, 2:3]
    u = (uv[:, 0] * (tw - 1)).round().astype(int).clip(0, tw - 1)
    v = (uv[:, 1] * (th - 1)).round().astype(int).clip(0, th - 1)
    out[m] = tex[v, u]
    return out

# ----------------------------------------------------------------------------- helpers
def nearest_fill(A, valid):
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return A[tuple(idx)]

def grad_slope(Hf, px):
    gz, gx = np.gradient(Hf, px)
    return np.hypot(gx, gz)

def hillshade(D, foot, px, az=315, alt=45):
    gz, gx = np.gradient(np.where(foot, D, D[foot].min()), px)
    az, alt = np.radians(az), np.radians(alt)
    s = np.arctan(np.hypot(gx, gz)); asp = np.arctan2(-gx, gz)
    hs = np.clip(np.sin(alt) * np.cos(s) + np.cos(alt) * np.sin(s) * np.cos(az - asp), 0, 1)
    hs[~foot] = 0
    return hs

def group_percentile(labels, values, q, n):
    out = np.zeros(n)
    if len(labels) == 0: return out
    order = np.lexsort((values, labels)); l = labels[order]; v = values[order]
    counts = np.bincount(l, minlength=n); starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    has = counts > 0
    pos = starts[has] + np.minimum(counts[has] - 1, np.floor(q / 100.0 * (counts[has] - 1) + 0.5).astype(int))
    out[has] = v[pos]
    return out

def group_mean(labels, values, n):
    cnt = np.bincount(labels, minlength=n).astype(float)
    return np.where(cnt > 0, np.bincount(labels, weights=values, minlength=n) / np.maximum(cnt, 1), 0)

def harmonic_fill(Hf, keep, foot):
    """Replace foot&~keep pixels by the solution of the Laplace equation with the kept pixels as boundary."""
    nz, nx = Hf.shape
    unknown = foot & ~keep
    uidx = np.where(unknown.ravel())[0]; nU = len(uidx)
    ids = -np.ones(Hf.size, np.int64); ids[uidx] = np.arange(nU)
    ur, uc = np.unravel_index(uidx, Hf.shape)
    rows, cols, vals = [], [], []; b = np.zeros(nU); diag = np.zeros(nU)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = ur + dr, uc + dc
        inb = (nr >= 0) & (nr < nz) & (nc >= 0) & (nc < nx)
        sel = np.where(inb)[0]; nr, nc = nr[inb], nc[inb]
        infoot = foot[nr, nc]; sel, nr, nc = sel[infoot], nr[infoot], nc[infoot]
        diag[sel] += 1
        nid = ids[nr * nx + nc]; isunk = nid >= 0
        rows.append(sel[isunk]); cols.append(nid[isunk]); vals.append(-np.ones(isunk.sum()))
        np.add.at(b, sel[~isunk], Hf[nr[~isunk], nc[~isunk]])
    rows.append(np.arange(nU)); cols.append(np.arange(nU)); vals.append(np.maximum(diag, 1e-9))
    A = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nU, nU))
    out = Hf.copy(); out.reshape(-1)[uidx] = spsolve(A, b)
    return out

# ----------------------------------------------------------------------------- object detection
OBJ_SCALES = ((3, 3.5, 0.0015), (6, 3.5, 0.0015), (12, 3.5, 0.002), (24, 4.0, 0.003), (48, 4.5, 0.004), (72, 4.5, 0.004))
PIT_SCALES = ((4, 3.5, 0.002), (8, 4.0, 0.003), (16, 4.5, 0.004))

def detect(Hf, foot, slope, scales, rim_q=60, frac_lower_min=0.6, T_L=3e-5, convex_frac_min=0.6,
           dome_max_r=12, dome_max_area=900, tiny_area=80, allow_dome=True, allow_tiny=True, tag=''):
    """Multi-scale top-hat candidates; accept those that look like objects standing on the ground.

    scales: (radius px, min rim slope [dy/dxy], min height above opened surface)
    A candidate is accepted when it is higher than most of its surrounding rim and
      - steep: the 60th percentile of the max slope around its boundary exceeds the rim threshold, or
      - dome:  mostly convex (tree canopy), small scales only, or
      - tiny:  a tiny compact bump (stone), smallest scale only.
    """
    slope_max5 = ndi.maximum_filter(slope, size=5, mode='nearest')
    L = ndi.laplace(ndi.gaussian_filter(Hf, 2.0, mode='nearest'))
    remove = np.zeros_like(foot)
    for r, T_rim, dh0 in scales:
        size = 2 * r + 1
        O = ndi.maximum_filter(ndi.minimum_filter(Hf, size=size, mode='nearest'), size=size, mode='nearest')
        cand = ((Hf - O) > dh0) & foot
        lab, n = ndi.label(cand, structure=S8)
        if n == 0: continue
        N = n + 1
        areas = np.bincount(lab.ravel(), minlength=N); areas[0] = 0
        inner_b = cand & ~ndi.binary_erosion(cand, structure=S8, border_value=0)
        steep_q = group_percentile(lab[inner_b], slope_max5[inner_b], rim_q, N)
        dil = ndi.grey_dilation(lab, size=(5, 5))
        rim = (lab == 0) & (dil > 0) & foot
        rim_lab = dil[rim]
        comp_med = group_percentile(lab[cand], Hf[cand], 50, N)
        lower = Hf[rim] < (comp_med[rim_lab] - dh0)
        rim_cnt = np.bincount(rim_lab, minlength=N).astype(float)
        frac_lower = np.where(rim_cnt > 0, np.bincount(rim_lab, weights=lower, minlength=N) / np.maximum(rim_cnt, 1), 0)
        convex = group_mean(lab[cand], (L[cand] < -T_L).astype(float), N)
        objs = ndi.find_objects(lab)
        bbarea = np.array([0] + [(s[0].stop - s[0].start) * (s[1].stop - s[1].start) for s in objs])
        fill = np.where(bbarea > 0, areas / np.maximum(bbarea, 1), 0)
        elevated = frac_lower >= frac_lower_min
        acc = (steep_q >= T_rim) & elevated
        if allow_dome and r <= dome_max_r:
            acc |= (convex >= convex_frac_min) & elevated & (areas <= dome_max_area) & (fill >= 0.4)
        if allow_tiny and r <= 4:
            acc |= (areas <= tiny_area) & (fill >= 0.4) & elevated
        acc[0] = False
        m = acc[lab]
        print(f'  {tag} r={r:2d}px candidates={n:5d} accepted={int(acc.sum()):5d} area={m.sum() / foot.sum():.3f}')
        remove |= m
    return remove

def build_dtm(H, px, T_wall=4.5, dil=2, min_island=60):
    valid = np.isfinite(H)
    Hf = nearest_fill(H, valid)
    foot = ndi.binary_fill_holes(ndi.binary_closing(valid, iterations=6))
    slope = grad_slope(Hf, px)
    obj = detect(Hf, foot, slope, OBJ_SCALES, tag='objects')
    pits = detect(-Hf, foot, slope, PIT_SCALES, allow_dome=False, allow_tiny=False, tag='pits   ')
    wall = (slope > T_wall) & foot
    remove = ndi.binary_dilation(obj | pits | wall | (foot & ~valid), iterations=dil, structure=S8) & foot
    keep = foot & ~remove
    labk, _ = ndi.label(keep); ak = np.bincount(labk.ravel()); ak[0] = 0
    small = np.isin(labk, np.where(ak < min_island)[0]) & keep
    keep &= ~small; remove |= small
    print(f'  removed {remove.sum() / foot.sum():.1%} of the footprint, interpolating ...')
    D = harmonic_fill(Hf, keep, foot)
    return D, foot, remove, Hf

# ----------------------------------------------------------------------------- mesh export
def grid_mesh(D, foot, px, ox, oz):
    nz, nx = D.shape
    vid = -np.ones(D.shape, np.int64)
    rr, cc = np.where(foot); vid[rr, cc] = np.arange(len(rr))
    V = np.stack([ox + (cc + 0.5) * px, D[rr, cc], oz + (rr + 0.5) * px], 1)
    a = vid[:-1, :-1]; b = vid[:-1, 1:]; c = vid[1:, :-1]; d = vid[1:, 1:]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    # counter-clockwise seen from +Y (up): (a, c, b) and (b, c, d) in (x, z) image layout
    F = np.vstack([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    return V, F

def export_glb(V, F, texture_img, grid, out_path, target_faces):
    import trimesh, fast_simplification
    red = max(0.0, 1.0 - target_faces / len(F))
    Vd, Fd = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=red, agg=7)
    Vd = Vd.astype(np.float64); Fd = Fd.astype(np.int64)
    uv = np.stack([(Vd[:, 0] - grid['ox']) / (grid['nx'] * grid['px']),
                   (Vd[:, 2] - grid['oz']) / (grid['nz'] * grid['px'])], 1)
    uv[:, 1] = 1.0 - uv[:, 1]   # trimesh flips V on export to match glTF's top-left origin
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=texture_img, metallicFactor=0.0, roughnessFactor=1.0, name='terrain')
    mesh = trimesh.Trimesh(Vd, Fd, visual=trimesh.visual.TextureVisuals(uv=uv, material=mat), process=False)
    mesh.fix_normals()
    mesh.metadata['name'] = 'terrain'
    mesh.export(out_path, include_normals=True)
    return mesh

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src'); ap.add_argument('out_dir')
    ap.add_argument('--res', type=int, default=1024, help='height map resolution (longest side, px)')
    ap.add_argument('--tex-res', type=int, default=2048, help='texture resolution (longest side, px)')
    ap.add_argument('--target-faces', type=int, default=120000)
    ap.add_argument('--name', default='terrain_reference')
    a = ap.parse_args()
    import os; os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    V, F, UV, tex = load_glb(a.src)
    print(f'loaded {len(V)} vertices, {len(F)} faces, bounds min {V.min(0).round(4)} max {V.max(0).round(4)}')
    R = rasterize_top(V, F, res=a.res); px = R['px']
    print(f'height map {R["nz"]}x{R["nx"]} px, pixel = {px:.6f} units  ({time.time() - t0:.0f}s)')
    D, foot, remove, Hf = build_dtm(R['H'], px)
    print(f'terrain height map done ({time.time() - t0:.0f}s)')

    # --- height map PNG (16 bit) + mask + metadata
    hmin, hmax = float(D[foot].min()), float(D[foot].max())
    q = np.zeros(D.shape, np.uint16)
    q[foot] = np.round((D[foot] - hmin) / (hmax - hmin) * 65535).astype(np.uint16)
    Image.fromarray(q).save(f'{a.out_dir}/{a.name}_heightmap16.png')
    Image.fromarray((foot * 255).astype(np.uint8)).save(f'{a.out_dir}/{a.name}_mask.png')
    Image.fromarray((remove * 255).astype(np.uint8)).save(f'{a.out_dir}/{a.name}_removed.png')
    Image.fromarray((hillshade(D, foot, px) * 255).astype(np.uint8)).save(f'{a.out_dir}/{a.name}_hillshade.png')
    meta = dict(
        source=os.path.basename(a.src), axes='x = image column (left->right), z = image row (top->bottom), y = height (up)',
        width_px=R['nx'], height_px=R['nz'], pixel_size=px,
        origin_x=R['ox'] + 0.5 * px, origin_z=R['oz'] + 0.5 * px,
        note='world_x = origin_x + col*pixel_size ; world_z = origin_z + row*pixel_size ; world_y = y_min + value/65535*(y_max-y_min)',
        y_min=hmin, y_max=hmax, source_bounds_min=V.min(0).tolist(), source_bounds_max=V.max(0).tolist(),
        mask='non-zero pixels of *_mask.png are inside the terrain footprint',
        removed='non-zero pixels of *_removed.png were classified as objects/walls and interpolated',
    )
    json.dump(meta, open(f'{a.out_dir}/{a.name}_heightmap.json', 'w'), indent=2)

    # --- texture: top-down base color with object pixels inpainted
    texture_img = None
    if tex is not None:
        Rt = rasterize_top(V, F, res=a.tex_res)
        ortho = sample_texture(tex, UV, F, Rt['TRI'], Rt['B']).astype(np.float64)
        # colour fill at height-map resolution (wider mask: object edges/shadows), then upsample into the masked areas
        ortho_lo = sample_texture(tex, UV, F, R['TRI'], R['B']).astype(np.float64)
        cmask = ndi.binary_dilation(remove, iterations=3, structure=S8) & foot
        keep_c = foot & ~cmask
        filled_lo = np.stack([harmonic_fill(ortho_lo[..., c], keep_c, foot) for c in range(3)], -1)
        zoom = (Rt['nz'] / R['nz'], Rt['nx'] / R['nx'])
        filled_hi = np.stack([ndi.zoom(filled_lo[..., c], zoom, order=1) for c in range(3)], -1)
        mask_hi = ndi.binary_dilation(ndi.zoom(cmask.astype(np.uint8), zoom, order=0) > 0, iterations=1)
        foot_hi = ndi.zoom(foot.astype(np.uint8), zoom, order=0) > 0
        out = np.where(mask_hi[..., None], filled_hi, ortho)
        # outside the footprint: nearest colour so bilinear sampling at the border stays clean
        idx = ndi.distance_transform_edt(~foot_hi, return_distances=False, return_indices=True)
        out = out[tuple(idx)]
        texture_img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
        texture_img.save(f'{a.out_dir}/{a.name}_albedo.jpg', quality=88)
        print(f'texture {Rt["nz"]}x{Rt["nx"]} done ({time.time() - t0:.0f}s)')

    # --- mesh
    Vg, Fg = grid_mesh(D, foot, px, R['ox'], R['oz'])
    print(f'grid mesh {len(Vg)} vertices, {len(Fg)} faces -> decimating to ~{a.target_faces} faces')
    tex_for_glb = Image.open(f'{a.out_dir}/{a.name}_albedo.jpg') if texture_img is not None else None
    mesh = export_glb(Vg, Fg, tex_for_glb, R, f'{a.out_dir}/{a.name}.glb', a.target_faces)
    print(f'GLB: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, {os.path.getsize(f"{a.out_dir}/{a.name}.glb") / 1e6:.1f} MB  ({time.time() - t0:.0f}s)')

if __name__ == '__main__':
    main()
