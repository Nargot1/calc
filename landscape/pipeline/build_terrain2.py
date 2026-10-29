"""Build the optimized low-poly terrain (v3: GLB-derived heightfield, terraces, rock slabs, two-tone rock) in Blender.
Run: python3 build_terrain2.py <workdir> <target_tris> <out.blend> <old.blend> [palette|texture] [slabs|noslabs]
<target_tris> is the final budget (slabs included; v2 default 50000, v3 default 65000).
"""
import bpy, bmesh, sys, json, math, time, os
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slabs import slab_pass

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
W_ = ARGS[0].rstrip('/') + '/'
TARGET_TRIS = int(ARGS[1]) if len(ARGS) > 1 else 50000
OUT_BLEND = ARGS[2] if len(ARGS) > 2 else W_ + 'landscape-v2.blend'
OLD_BLEND = ARGS[3] if len(ARGS) > 3 else 'landscape-authored.blend'
COLOR_MODE = ARGS[4] if len(ARGS) > 4 else 'palette'
SLABS = (ARGS[5] if len(ARGS) > 5 else 'slabs') == 'slabs'

meta = json.load(open(W_ + 'terrain_meta.json'))
H = np.load(W_ + 'terrain_h.npy').astype(np.float64)
MASKS = dict(np.load(W_ + 'terrain_masks.npz'))
GLBCOL = np.load(W_ + 'hf_col.npy').astype(np.float64) / 255
CELL, X0, Y0, NX, NY = meta['cell'], meta['x0'], meta['y0'], meta['nx'], meta['ny']
SEA, BEACH_Z, CITY_Z = meta['sea'], meta['beach'], meta['city']
CITY_X, CITY_Y = meta['city_x'], meta['city_y']
STEP = meta.get('terrace_step', 45.0) or 45.0
BEACH_X = (-773.0, 293.0)
SUN = np.array([0.473, 0.473, 0.743])   # direction towards the Sun lamp (rot 42 deg, 0, 135 deg)

def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
COL = {k: hexc(v) for k, v in {
    'grass_a': '#7CC247', 'grass_b': '#6DB53B', 'grass_c': '#8ACB54', 'grass_dk': '#5FA637', 'grass_far': '#6FB45A',
    'rock': '#8E7B69', 'rock_dk': '#75665A', 'rock_g': '#8D9096', 'rock_g_dk': '#6F7278', 'cliff': '#9A8470',
    'snow': '#F2F6F9', 'snow_sh': '#D9E6F0', 'ice_rock': '#A9B4BE', 'sand': '#F1DCA6', 'sand_wet': '#E4CA90',
    'rock_lit': '#AE937A', 'rock_sh': '#85798B', 'rock_lit_hi': '#A3A6AD', 'rock_sh_hi': '#6E7284', 'rock_lit_snow': '#B9C1CB', 'rock_sh_snow': '#7A8096',
    'lip': '#6FBF3F', 'lip_dk': '#5FA637',
    'floor_deep': '#3A9AC4', 'floor_mid': '#8FC4B4', 'road': '#45474D', 'water': '#25BFEE', 'haze': '#A9DCE0', 'city': '#A7B58E',
}.items()}

def sample(arr, x, y, order=1):
    cx = (np.asarray(x) - X0) / CELL; cy = (np.asarray(y) - Y0) / CELL
    return ndimage.map_coordinates(arr, [cy, cx], order=order, mode='nearest')

# ------------------------------------------------------------------ scene
bpy.ops.wm.read_homefile(use_empty=True)
sc = bpy.context.scene
sc.unit_settings.system = 'METRIC'; sc.unit_settings.scale_length = 1.0
for c in list(bpy.data.collections): bpy.data.collections.remove(c)
def new_col(name, parent=None):
    col = bpy.data.collections.new(name); (parent or sc.collection).children.link(col); return col
C_LAND = new_col('Landscape'); C_TERR = new_col('Terrain', C_LAND); C_WATER = new_col('Water', C_LAND)
C_REF = new_col('Reference_City'); C_CAM = new_col('Cameras')

mat = bpy.data.materials.new('VCol'); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes['Principled BSDF']
attr = nt.nodes.new('ShaderNodeVertexColor'); attr.layer_name = 'Col'
nt.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
bsdf.inputs['Roughness'].default_value = 1.0
if 'Specular IOR Level' in bsdf.inputs: bsdf.inputs['Specular IOR Level'].default_value = 0.0
mat_water = bpy.data.materials.new('Water'); mat_water.use_nodes = True
wb = mat_water.node_tree.nodes['Principled BSDF']
wb.inputs['Base Color'].default_value = (*srgb_to_linear(COL['water']), 1); wb.inputs['Roughness'].default_value = 0.15
wb.inputs['Alpha'].default_value = 0.72
mat_water.diffuse_color = (*srgb_to_linear(COL['water']), 0.72)

def make_mesh_object(name, verts, faces, col, colors=None, mat_=mat, flat=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces]); me.update()
    if colors is not None:
        ca = me.color_attributes.new(name='Col', type='BYTE_COLOR', domain='CORNER')
        loop_tot = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get('loop_total', loop_tot)
        lc = np.repeat(np.asarray(colors, float), loop_tot, axis=0)
        ca.data.foreach_set('color', np.concatenate([lc, np.ones((len(lc), 1))], 1).ravel())
    if flat: me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
    me.materials.append(mat_)
    ob = bpy.data.objects.new(name, me); col.objects.link(ob)
    return ob

# ------------------------------------------------------------------ 1. grid + decimate
t0 = time.time()
xs = X0 + np.arange(NX) * CELL; ys = Y0 + np.arange(NY) * CELL
XX, YY = np.meshgrid(xs, ys)
verts = np.stack([XX.ravel(), YY.ravel(), H.ravel()], 1)
idx = np.arange(NX * NY).reshape(NY, NX)
a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel(); c = idx[1:, 1:].ravel(); d = idx[1:, :-1].ravel()
iy, ix = np.meshgrid(np.arange(NY - 1), np.arange(NX - 1), indexing='ij'); alt = ((iy + ix) % 2 == 0).ravel()
t1 = np.where(alt[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
t2 = np.where(alt[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
faces = np.concatenate([t1, t2], 0)
full = make_mesh_object('TerrainFull', verts, faces, C_TERR)
print(f"grid mesh {len(verts)} verts {len(faces)} tris in {time.time()-t0:.1f}s")
# v3: weighted decimation - wall cells (heightfield slope > 50 deg) collapse first, so every terrace wall becomes a single row of
# long faces (the slab pass multiplies wall faces ~8x, so they must be cheap); the rest of the island keeps its detail.
if SLABS:
    gy_, gx_ = np.gradient(H, CELL); wall_cell = (np.degrees(np.arctan(np.hypot(gx_, gy_))) > 50).ravel()
    vg = full.vertex_groups.new(name='dec')
    vg.add(np.nonzero(wall_cell)[0].tolist(), 1.0, 'REPLACE'); vg.add(np.nonzero(~wall_cell)[0].tolist(), 0.25, 'REPLACE')
def decimate(ob, n_target):
    bpy.context.view_layer.objects.active = ob; ob.select_set(True)
    mod = ob.modifiers.new('dec', 'DECIMATE'); mod.decimate_type = 'COLLAPSE'; mod.ratio = min(1.0, n_target / len(ob.data.polygons)); mod.use_collapse_triangulate = True
    if SLABS: mod.vertex_group = 'dec'; mod.vertex_group_factor = 8.0
    bpy.ops.object.modifier_apply(modifier='dec')
def mesh_arrays(me):
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    tr = np.empty(len(me.polygons) * 3, dtype=np.int32); me.polygons.foreach_get('vertices', tr); tr = tr.reshape(-1, 3)
    # snap plateau verts exactly to CITY_Z, sea-level-ish beach verts stay
    inside = (co[:, 0] >= CITY_X[0] - 2) & (co[:, 0] <= CITY_X[1] + 2) & (co[:, 1] >= CITY_Y[0] - 2) & (co[:, 1] <= CITY_Y[1] + 2)
    co[:, 2] = np.where(inside & (np.abs(co[:, 2] - CITY_Z) < 6), CITY_Z, co[:, 2])
    return co, tr.astype(np.int64)
fixed_fn = lambda x, y, z: (x >= CITY_X[0] - 2) & (x <= CITY_X[1] + 2) & (y >= CITY_Y[0] - 2) & (y <= CITY_Y[1] + 2) & (np.abs(z - CITY_Z) < 0.5)
inward_fn = lambda x, y, z: (x >= BEACH_X[0] - 20) & (x <= BEACH_X[1] + 20) & (y >= CITY_Y[1] - 1) & (y <= CITY_Y[1] + 14) & (z > BEACH_Z - 1)   # city cliff: never into the beach parts
def run_slabs(co, tr):
    c = co[tr].mean(1)
    tw = sample(MASKS['terrace_w'], c[:, 0], c[:, 1]) if 'terrace_w' in MASKS else np.ones(len(tr))
    allow = (tw > 0.4) | inward_fn(c[:, 0], c[:, 1], c[:, 2])          # terraced walls + the city cliff; free mountain faces stay faceted
    return slab_pass(co, tr, city_z=CITY_Z, step=STEP, x0=X0, y0=Y0, nx=NX, ny=NY, cell=CELL, z_min=BEACH_Z - 4, fixed=fixed_fn, inward=inward_fn, allow=allow)
t0 = time.time()
if SLABS:
    # two-stage: decimate to a first guess, measure what the slab pass adds, decimate again from that mesh to land on the budget
    pre = int(TARGET_TRIS * 0.65)
    decimate(full, pre); me = full.data
    print(f"decimated to {len(me.polygons)} faces in {time.time()-t0:.1f}s")
    stage1 = full.data.copy()                           # keep the stage-1 mesh so later stages can start from it again
    co, tris = mesh_arrays(me)
    co2, tris2, FL = run_slabs(co, tris)
    pts = [(len(tris), len(tris2))]                     # (pre-slab faces, total after slabs)
    best = (abs(len(tris2) - TARGET_TRIS), co2, tris2, FL)
    for it in range(3):
        if abs(pts[-1][1] - TARGET_TRIS) <= 0.025 * TARGET_TRIS: break
        if len(pts) == 1: guess = int(TARGET_TRIS * pts[0][0] / pts[0][1])          # proportional
        else:                                                                       # secant on the two latest points
            (a0, b0), (a1, b1) = pts[-2], pts[-1]; k = (b1 - b0) / (a1 - a0) if a1 != a0 else 1.5; guess = int(a1 + (TARGET_TRIS - b1) / max(k, 0.5))
        guess = int(min(len(stage1.polygons) - 1, max(1000, guess)))
        t1 = time.time()
        full.data = stage1.copy(); decimate(full, guess); me = full.data
        co, tris = mesh_arrays(me)
        co2, tris2, FL = run_slabs(co, tris)
        pts.append((len(tris), len(tris2)))
        print(f"stage {it + 2}: re-decimated to {len(tris)} faces -> {len(tris2)} tris after slabs (target {TARGET_TRIS}) in {time.time()-t1:.1f}s")
        if abs(len(tris2) - TARGET_TRIS) < best[0]: best = (abs(len(tris2) - TARGET_TRIS), co2, tris2, FL)
    _, co, tris, FL = best
else:
    decimate(full, TARGET_TRIS); me = full.data
    print(f"decimated to {len(me.polygons)} faces in {time.time()-t0:.1f}s")
    co, tris = mesh_arrays(me)
    FL = dict(wall=np.zeros(len(tris), bool), ledge=np.zeros(len(tris), bool), band_type=np.full(len(tris), -1))
print(f"terrain mesh: {len(co)} verts {len(tris)} tris (slabs={SLABS}) in {time.time()-t0:.1f}s")

# ------------------------------------------------------------------ 2. per-face colours
nf = len(tris)
A_, B_, C_ = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
cen = (A_ + B_ + C_) / 3
nrm = np.cross(B_ - A_, C_ - A_); nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
slope = np.degrees(np.arccos(np.clip(nrm[:, 2], -1, 1)))
WALL, LEDGE, BTYPE = FL['wall'], FL['ledge'], FL['band_type']
z = cen[:, 2]; fx, fy = cen[:, 0], cen[:, 1]
s_snow = sample(MASKS['snow'].astype(float), fx, fy); s_snowd = sample(MASKS['snow_d'], fx, fy)
s_beach = sample(MASKS['beach'].astype(float), fx, fy); s_sea = sample(MASKS['sea'].astype(float), fx, fy)
s_road = sample(MASKS['road'].astype(float), fx, fy); s_grass = sample(MASKS['grass_d'], fx, fy); s_rock = sample(MASKS['rock_d'], fx, fy)
s_dcity = sample(MASKS['dcity'], fx, fy); s_valid = sample(MASKS['valid'].astype(float), fx, fy); s_shore = sample(MASKS['dshore'], fx, fy)
rng = np.random.default_rng(3)
col = np.zeros((nf, 3))
if COLOR_MODE == 'texture':
    for k in range(3): col[:, k] = sample(ndimage.gaussian_filter(GLBCOL[..., k], 1.0), fx, fy)
    # boost saturation a bit and lift shadows
    mean = col.mean(1, keepdims=True); col = np.clip(mean + (col - mean) * 1.25 + 0.03, 0, 1)
else:
    snow = s_snow > 0.5
    gpick = rng.random(nf)
    grass = np.where((gpick < 0.45)[:, None], COL['grass_a'], np.where((gpick < 0.8)[:, None], COL['grass_b'], COL['grass_c']))
    # darker grass where the GLB has dark/forest colours (rock_d high but grass present) -> keep lively
    far = np.clip((-fy - 700) / 500, 0, 1)[:, None]
    grass = grass * (1 - far) + COL['grass_far'] * far
    # modulate by the GLB texture luminance (forest = darker, sunny slopes = brighter)
    lum = sample(ndimage.gaussian_filter(GLBCOL @ np.array([0.3, 0.59, 0.11]), 1.5), fx, fy)
    lumn = np.clip((lum - 0.22) / 0.4, 0, 1)[:, None]
    grass = grass * (0.78 + 0.42 * lumn)
    col[:] = grass
    # rock on steep faces; threshold lower where the GLB texture is rocky, higher where grassy
    rock_thr = 40 - 14 * np.clip(s_rock * 2.5, 0, 1) + 8 * np.clip(s_grass * 2, 0, 1) + 9 * np.clip((z - 120) / 120, 0, 1)  # far/high green mountains stay greener
    rock_mix = np.clip((slope - rock_thr) / 10, 0, 1)[:, None]
    # headlands / sea cliffs that the GLB paints as bare rock
    bare = (s_rock > 0.45) & (s_grass < 0.3) & (z < 120) & (s_valid > 0.5)
    rock_mix = np.where(bare[:, None], np.maximum(rock_mix, 0.9), rock_mix)
    # v3: on terraced hillsides the shelves are grass whatever the GLB texture says there; rock only on the walls
    s_tw = sample(MASKS['terrace_w'], fx, fy) if 'terrace_w' in MASKS else np.zeros(nf)
    terraced = s_tw > 0.4
    rock_mix = np.where(terraced[:, None], np.clip((slope - 40) / 12, 0, 1)[:, None], rock_mix)
    # v3 two-tone rock: warm tan on faces turned towards the sun, cool purple-grey in the shade (soft split around grazing)
    lit = np.clip((nrm @ SUN + 0.05) / 0.30, 0, 1)[:, None]
    high = np.clip((z - 180) / 200, 0, 1)[:, None]
    low = np.clip((20 - z) / 40, 0, 1)[:, None] * 0.2        # v3: sea-level rock stays warm like the painting's headlands and city cliff (v2: 0.6)
    gray_mix = np.maximum(high, low)
    r_lit = COL['rock_lit'] * (1 - gray_mix) + COL['rock_lit_hi'] * gray_mix
    r_sh = COL['rock_sh'] * (1 - gray_mix) + COL['rock_sh_hi'] * gray_mix
    r_lit = np.where(snow[:, None], COL['rock_lit_snow'], r_lit); r_sh = np.where(snow[:, None], COL['rock_sh_snow'], r_sh)
    rock = r_sh * (1 - lit) + r_lit * lit
    rock = rock * (1 + rng.uniform(-0.03, 0.03, (nf, 1)))
    # darker base band of every wall (band 0 of the slab pass) - the painting's boulder-strewn feet
    base_band = WALL & ~LEDGE & (BTYPE == 0)
    rock = np.where(base_band[:, None], rock * 0.82, rock)
    snowc = np.where((slope < 22)[:, None], COL['snow'], COL['snow_sh'])
    col = np.where(snow[:, None], snowc, col)
    # in the snow zone rock shows only on steep faces
    rock_mix = np.where(snow[:, None], np.clip((slope - 42) / 10, 0, 1)[:, None], rock_mix)
    # slab walls and their ledges are always rock, whatever their slope after the offsets
    rock_mix = np.where((WALL | LEDGE)[:, None], 1.0, rock_mix)
    col = col * (1 - rock_mix) + rock * rock_mix
    # grass lip: the top ~6 studs of every wall (band 2) just below the shelf edge; snow lip on the snow side
    lip = WALL & ~LEDGE & (BTYPE == 2) & (z > BEACH_Z + 2)
    lipc = np.where((rng.random(nf) < 0.6)[:, None], COL['lip'], COL['lip_dk'])
    col = np.where((lip & ~snow)[:, None], lipc, col)
    col = np.where((lip & snow)[:, None], COL['snow_sh'], col)
    # sea cliffs under the plateau front: rocky (two-tone as well)
    cliff = (fy > CITY_Y[1]) & (fy < CITY_Y[1] + 40) & (z > BEACH_Z + 2) & (z < CITY_Z - 2)
    col = np.where((cliff & (slope > 30) & ~lip)[:, None], rock, col)
    # sand
    is_sand = ((s_beach > 0.3) | ((z > SEA - 3) & (z < BEACH_Z + 4) & (slope < 25) & (s_shore < 120) & (fy > 400))) & (slope < 35) & ~WALL & ~LEDGE
    sand = np.where((z < BEACH_Z - 4)[:, None], COL['sand_wet'], COL['sand'])
    col = np.where(is_sand[:, None], sand, col)
    # underwater
    uw = (z < SEA - 0.5)
    depth = np.clip((SEA - 3 - z) / 14, 0, 1)[:, None]
    floor = COL['sand_wet'] * (1 - depth) + COL['floor_deep'] * depth
    col = np.where(uw[:, None], floor, col)
    # road on the +X hill
    on_road = (s_road > 0.4) & (slope < 45) & ~WALL & ~LEDGE
    col = np.where(on_road[:, None], COL['road'], col)
    # plateau: hidden under the city; colour by side (icy district = snow, rest = grass)
    plat = (s_dcity < 20) & (slope < 15)   # flat plateau faces only (not the cliff faces at its edge)
    col = np.where((plat & (s_snowd > 0.12))[:, None], COL['snow'], col)
    col = np.where((plat & (s_snowd <= 0.12))[:, None], COL['grass_b'], col)
    # haze with distance from the city centre
    dist = np.hypot(fx + 196, fy - 76)
    haze = np.clip((dist - 1500) / 1800, 0, 0.3)[:, None]
    col = col * (1 - haze) + COL['haze'] * haze
    col = col * (1 + rng.uniform(-0.045, 0.045, (nf, 1)) * (~plat)[:, None])   # no jitter on the flat plateau
col = np.clip(col, 0, 1)
col_lin = srgb_to_linear(col)

# ------------------------------------------------------------------ 3. chunks
MAXT = 9800
def chunk_faces(face_ids, x0, y0, x1, y1, name_prefix, out):
    if len(face_ids) <= MAXT:
        if len(face_ids): out.append((name_prefix, face_ids))
        return
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    cx = cen[face_ids, 0]; cy = cen[face_ids, 1]
    for qi, (mx, my) in enumerate([(cx < xm, cy < ym), (cx >= xm, cy < ym), (cx < xm, cy >= ym), (cx >= xm, cy >= ym)]):
        sel = face_ids[mx & my]
        bx0, bx1 = (x0, xm) if qi % 2 == 0 else (xm, x1); by0, by1 = (y0, ym) if qi < 2 else (ym, y1)
        chunk_faces(sel, bx0, by0, bx1, by1, f"{name_prefix}{qi}", out)
chunks = []
TX = np.linspace(X0, X0 + CELL * (NX - 1), 5); TY = np.linspace(Y0, Y0 + CELL * (NY - 1), 5)
tile_i = np.clip(np.searchsorted(TX, cen[:, 0], side='right') - 1, 0, 3); tile_j = np.clip(np.searchsorted(TY, cen[:, 1], side='right') - 1, 0, 3)
for j in range(4):
    for i in range(4):
        fid = np.nonzero((tile_i == i) & (tile_j == j))[0]
        chunk_faces(fid, TX[i], TY[j], TX[i + 1], TY[j + 1], f"Terrain_{j}{i}_", chunks)
print("chunks:", len(chunks), "sizes:", sorted([len(f) for _, f in chunks], reverse=True)[:12])
total = 0
for name, fid in chunks:
    f = tris[fid]
    uniq, inv = np.unique(f.ravel(), return_inverse=True)
    v = co[uniq]; fl = inv.reshape(-1, 3)
    ctr = (v.min(0) + v.max(0)) / 2; ctr[2] = 0
    ob = make_mesh_object(name.rstrip('_'), v - ctr, fl, C_TERR, colors=col_lin[fid]); ob.location = ctr
    total += len(fl)
print("total terrain tris", total)
bpy.data.objects.remove(full, do_unlink=True)

# ------------------------------------------------------------------ 4. water plane (big)
E = 8000
wv = [(X0 - E, Y0 - E, SEA), (X0 + CELL * NX + E, Y0 - E, SEA), (X0 + CELL * NX + E, Y0 + CELL * NY + E, SEA), (X0 - E, Y0 + CELL * NY + E, SEA)]
make_mesh_object('Water', wv, [(0, 1, 2, 3)], C_WATER, colors=np.array([srgb_to_linear(COL['water'])]), mat_=mat_water)

# ------------------------------------------------------------------ 5. reference city + cameras from the old file
# source of the city + cameras: either the original landscape-authored.blend ('Context', normals flipped)
# or a previous landscape-v2*.blend ('Context_City', normals already fixed, + MatchCam)
with bpy.data.libraries.load(OLD_BLEND, link=False) as (src, dst):
    dst.objects = [n for n in src.objects if n in ('Context', 'Context_City', 'RefCam', 'TopCam', 'ViewCam', 'MatchCam')]
for ob in dst.objects:
    if ob is None: continue
    if ob.type == 'CAMERA': C_CAM.objects.link(ob)
    else:
        C_REF.objects.link(ob)
        if ob.name == 'Context':  # the raw export has all normals flipped
            bm = bmesh.new(); bm.from_mesh(ob.data); bmesh.ops.reverse_faces(bm, faces=bm.faces); bm.to_mesh(ob.data); bm.free()
        ob.name = 'Context_City'
        if not ob.data.materials: ob.data.materials.append(mat)
        else:
            for i in range(len(ob.data.materials)): ob.data.materials[i] = mat
sc.camera = bpy.data.objects['RefCam']
sc.render.resolution_x, sc.render.resolution_y = 1672, 941
SCR = os.path.dirname(os.path.abspath(__file__)) + '/'
if os.path.exists(SCR + 'fitcam.json') and 'MatchCam' not in bpy.data.objects:
    fc = json.load(open(SCR + 'fitcam.json'))
    cd = bpy.data.cameras.new('MatchCam'); cd.lens = fc['lens']; cd.sensor_fit = 'VERTICAL'; cd.sensor_height = 24; cd.clip_end = 100000; cd.clip_start = 1
    cob = bpy.data.objects.new('MatchCam', cd); C_CAM.objects.link(cob); cob.location = fc['pos']; cob.rotation_euler = fc['rot']
ld = bpy.data.lights.new('Sun', 'SUN'); ld.energy = 3.0; ld.angle = math.radians(4)
lo = bpy.data.objects.new('Sun', ld); C_CAM.objects.link(lo); lo.rotation_euler = (math.radians(42), 0, math.radians(135))
w = bpy.data.worlds.new('World'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.36, 0.62, 0.95, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 1.0
for lc in bpy.context.view_layer.layer_collection.children:
    if lc.collection.name == 'Reference_City': lc.exclude = True
bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
print("saved", OUT_BLEND)
