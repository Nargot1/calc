"""v2 vegetation: GLB-derived tree candidates + hill scatter + rocks + backdrop islands. Instances are linked duplicates
of 14 flat-shaded assets; ground height from a BVH of the real terrain chunks.
Run: python3 build_vegetation2.py <workdir> <blend> [extra_density 0..2]
"""
import bpy, bmesh, sys, json, math, os
import numpy as np
from scipy import ndimage
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
W_ = ARGS[0].rstrip('/') + '/'; BLEND = ARGS[1]; EXTRA = float(ARGS[2]) if len(ARGS) > 2 else 1.0
meta = json.load(open(W_ + 'terrain_meta.json'))
H = np.load(W_ + 'terrain_h.npy').astype(np.float64)
MASKS = dict(np.load(W_ + 'terrain_masks.npz'))
CELL, X0, Y0, NX, NY = meta['cell'], meta['x0'], meta['y0'], meta['nx'], meta['ny']
SEA, BEACH_Z, CITY_Z = meta['sea'], meta['beach'], meta['city']
CITY_X, CITY_Y = meta['city_x'], meta['city_y']
cands = json.load(open(W_ + 'tree_candidates.json'))
rng = np.random.default_rng(21)

def srgb_to_linear(c):
    c = np.asarray(c, float); return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
def sample(arr, x, y, order=1):
    scalar = np.ndim(x) == 0
    cx = (np.atleast_1d(np.asarray(x, float)) - X0) / CELL; cy = (np.atleast_1d(np.asarray(y, float)) - Y0) / CELL
    r = ndimage.map_coordinates(arr, [cy, cx], order=order, mode='nearest')
    return float(r[0]) if scalar else r
GY, GX = np.gradient(H, CELL); SLOPE = np.degrees(np.arctan(np.hypot(GX, GY)))
# v3: wall feet = flat cells with a wall rising > 15 studs within ~10 studs (terraced zones only) -> boulders, like the painting
TW = MASKS['terrace_w'] if 'terrace_w' in MASKS else np.zeros_like(H)
FOOT = ((ndimage.maximum_filter(H, size=5) - H) > 15) & (SLOPE < 18) & (TW > 0.3)

bpy.ops.wm.open_mainfile(filepath=BLEND)
sc = bpy.context.scene
mat = bpy.data.materials['VCol']
def get_col(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name); (parent or sc.collection).children.link(col)
    return col
C_ASSETS = get_col('Assets'); C_VEG = get_col('Vegetation'); C_ROCKS = get_col('Rocks'); C_ISL = get_col('DistantIslands')
for c in (C_VEG, C_ROCKS, C_ISL, C_ASSETS):
    for ob in list(c.objects): bpy.data.objects.remove(ob, do_unlink=True)

# BVH of the terrain chunks for exact ground height
verts = []; polys = []; off = 0
for ob in bpy.data.collections['Terrain'].objects:
    me = ob.data; M = ob.matrix_world
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3) @ np.array(M.to_3x3()).T + np.array(M.translation)
    verts.extend(map(tuple, co))
    for p in me.polygons: polys.append(tuple(off + i for i in p.vertices))
    off += len(me.vertices)
bvh = BVHTree.FromPolygons(verts, polys)
def ground_z(x, y):
    hit = bvh.ray_cast(Vector((x, y, 2000)), Vector((0, 0, -1)), 5000)
    return hit[0].z if hit[0] is not None else sample(H, x, y)

# ------------------------------------------------------------------ mesh helpers (same as v1)
class MB:
    def __init__(self): self.v = []; self.f = []; self.c = []
    def add(self, verts, faces, color):
        base = len(self.v); self.v += [tuple(p) for p in verts]
        for fc in faces:
            self.f.append(tuple(base + i for i in fc)); self.c.append(np.asarray(color, float))
    def cone(self, cx, cy, z0, r0, r1, h, n, color, bottom=False, jitter=0.0):
        ring0 = [(cx + r0 * math.cos(2 * math.pi * i / n + jitter), cy + r0 * math.sin(2 * math.pi * i / n + jitter), z0) for i in range(n)]
        if r1 > 0.05:
            ring1 = [(cx + r1 * math.cos(2 * math.pi * i / n + jitter), cy + r1 * math.sin(2 * math.pi * i / n + jitter), z0 + h) for i in range(n)]
            self.add(ring0 + ring1, [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)], color)
        else:
            self.add(ring0 + [(cx, cy, z0 + h)], [(i, (i + 1) % n, n) for i in range(n)], color)
        if bottom:
            base = len(self.v); self.v += ring0
            self.f.append(tuple(base + i for i in range(n))[::-1]); self.c.append(np.asarray(color, float) * 0.8)
    def icosphere(self, cx, cy, cz, r, sub, color, scale=(1, 1, 1), jitter=0.0, seed=0):
        bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=sub + 1, radius=1.0)
        rr = np.random.default_rng(seed); verts = []
        for v in bm.verts:
            j = 1 + rr.uniform(-jitter, jitter)
            verts.append((cx + v.co.x * r * scale[0] * j, cy + v.co.y * r * scale[1] * j, cz + v.co.z * r * scale[2] * j))
        faces = [tuple(v.index for v in f.verts) for f in bm.faces]
        base = len(self.v); self.v += verts
        for fc in faces:
            self.f.append(tuple(base + i for i in fc)); self.c.append(np.asarray(color, float))
        bm.free()
    def finish(self, name, col, jitter=0.05, seed=0):
        me = bpy.data.meshes.new(name); me.from_pydata(self.v, [], self.f); me.update()
        rr = np.random.default_rng(seed)
        cols = np.array(self.c) * (1 + rr.uniform(-jitter, jitter, (len(self.c), 1)))
        nrm = np.empty(len(me.polygons) * 3); me.polygons.foreach_get('normal', nrm); nrm = nrm.reshape(-1, 3)
        cols = np.clip(cols * (0.86 + 0.14 * np.clip(nrm[:, 2:3] * 0.5 + 0.5, 0, 1)), 0, 1)
        ca = me.color_attributes.new(name='Col', type='BYTE_COLOR', domain='CORNER')
        lt = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get('loop_total', lt)
        lc = np.repeat(srgb_to_linear(cols), lt, axis=0)
        ca.data.foreach_set('color', np.concatenate([lc, np.ones((len(lc), 1))], 1).ravel())
        me.polygons.foreach_set('use_smooth', [False] * len(me.polygons)); me.materials.append(mat)
        ob = bpy.data.objects.new(name, me); col.objects.link(ob); return ob

assets = {}
def tree_round(name, seed, canopy=('#5DB53B', '#79C94B'), trunk='#6E4B2F'):
    m = MB(); r = np.random.default_rng(seed)
    m.cone(0, 0, 0, 2.8, 2.0, 17, 5, hexc(trunk))
    m.icosphere(0, 0, 26, 15.0, 1, hexc(canopy[0]), scale=(1, 1, 1.08), jitter=0.07, seed=seed)
    for k in range(2):
        a = r.uniform(0, 2 * math.pi); d = r.uniform(7, 10)
        m.icosphere(d * math.cos(a), d * math.sin(a), 20 + r.uniform(0, 7), 9.0, 0, hexc(canopy[1]), scale=(1, 1, 0.95), jitter=0.1, seed=seed + k + 1)
    return m.finish(name, C_ASSETS, seed=seed)
def tree_conifer(name, seed, cols=('#2F7E3B', '#3B9447', '#2A7236'), trunk='#5E4028', scale=1.0):
    m = MB(); r = np.random.default_rng(seed)
    m.cone(0, 0, 0, 1.9, 1.5, 8, 5, hexc(trunk))
    for k, (rad, hh) in enumerate([(11.5, 16), (9.3, 15), (6.8, 14)]):
        m.cone(0, 0, 7 + k * 10.5, rad * scale, 0, hh * scale, 7, hexc(cols[k % len(cols)]), jitter=r.uniform(0, 1))
    return m.finish(name, C_ASSETS, seed=seed)
def tree_snow_conifer(name, seed):
    m = MB(); r = np.random.default_rng(seed)
    m.cone(0, 0, 0, 1.9, 1.5, 8, 5, hexc('#5E4028'))
    for k, (rad, hh) in enumerate([(11.5, 16), (9.3, 15), (6.8, 14)]):
        m.cone(0, 0, 7 + k * 10.5, rad, 0, hh, 7, hexc(['#B6E0F1', '#C9EAF6', '#EEF7FB'][k]), jitter=r.uniform(0, 1))
    return m.finish(name, C_ASSETS, seed=seed)
def tree_palm(name, seed):
    m = MB(); r = np.random.default_rng(seed)
    n = 6; pts = []; lean = r.uniform(0.25, 0.45); a = r.uniform(0, 2 * math.pi)
    for i in range(n + 1):
        t = i / n; pts.append((math.cos(a) * lean * 34 * t * t, math.sin(a) * lean * 34 * t * t, 32 * t))
    sides = 5; rings = []
    for i, p in enumerate(pts):
        rad = 2.8 - 1.2 * (i / n)
        rings.append([(p[0] + rad * math.cos(2 * math.pi * k / sides), p[1] + rad * math.sin(2 * math.pi * k / sides), p[2]) for k in range(sides)])
    verts = [v for ring in rings for v in ring]; faces = []
    for i in range(n):
        for k in range(sides):
            faces.append((i * sides + k, i * sides + (k + 1) % sides, (i + 1) * sides + (k + 1) % sides, (i + 1) * sides + k))
    m.add(verts, faces, hexc('#8E6E4C'))
    top = pts[-1]
    for k in range(7):
        ang = 2 * math.pi * k / 7 + r.uniform(-0.2, 0.2); L = r.uniform(17, 22)
        dx, dy = math.cos(ang), math.sin(ang); px, py = -dy, dx; strip = []
        for j, (t, w, dz) in enumerate([(0, 1.4, 3.0), (0.35, 3.8, 4.2), (0.7, 3.0, 0.6), (1.0, 0.7, -6.0)]):
            cx, cy, cz = top[0] + dx * L * t, top[1] + dy * L * t, top[2] + dz
            strip += [(cx + px * w, cy + py * w, cz), (cx - px * w, cy - py * w, cz)]
        m.add(strip, [(2 * j, 2 * j + 1, 2 * j + 3, 2 * j + 2) for j in range(3)], hexc('#4DAF49' if k % 2 else '#3F9E42'))
    return m.finish(name, C_ASSETS, seed=seed)
def bush(name, seed):
    m = MB(); m.icosphere(0, 0, 3.2, 5.5, 0, hexc('#5EB243'), scale=(1.15, 1, 0.75), jitter=0.12, seed=seed)
    return m.finish(name, C_ASSETS, seed=seed)
def rock(name, seed, color='#8B8176', r=9):
    m = MB(); rr = np.random.default_rng(seed)
    m.icosphere(0, 0, r * 0.45, r, 1, hexc(color), scale=(rr.uniform(0.8, 1.3), rr.uniform(0.8, 1.3), rr.uniform(0.55, 0.8)), jitter=0.22, seed=seed)
    return m.finish(name, C_ASSETS, jitter=0.08, seed=seed)
assets['TreeRound_A'] = tree_round('TreeRound_A', 1)
assets['TreeRound_B'] = tree_round('TreeRound_B', 2, canopy=('#55AE38', '#6FC247'))
assets['TreeConifer_A'] = tree_conifer('TreeConifer_A', 3)
assets['TreeConifer_B'] = tree_conifer('TreeConifer_B', 4, cols=('#347F3D', '#3E9A4A', '#2D7A39'), scale=1.12)
assets['TreeSnowConifer_A'] = tree_snow_conifer('TreeSnowConifer_A', 5)
assets['TreeSnowConifer_B'] = tree_snow_conifer('TreeSnowConifer_B', 6)
assets['TreePalm_A'] = tree_palm('TreePalm_A', 7)
assets['TreePalm_B'] = tree_palm('TreePalm_B', 8)
assets['Bush_A'] = bush('Bush_A', 9)
assets['Rock_A'] = rock('Rock_A', 10); assets['Rock_B'] = rock('Rock_B', 11, r=7); assets['Rock_C'] = rock('Rock_C', 12, r=12)
assets['RockSnow_A'] = rock('RockSnow_A', 13, color='#8E949B'); assets['RockSnow_B'] = rock('RockSnow_B', 14, color='#9AA0A6', r=7)
for i, (n, ob) in enumerate(assets.items()):
    ob.location = (-3800 + (i % 8) * 60, 1500 + (i // 8) * 80, SEA + 2); ob['asset'] = n
print("assets tris:", {n: len(ob.data.polygons) for n, ob in assets.items()})

# ------------------------------------------------------------------ placement
placements = []
def place(asset, x, y, z, yaw, s, col):
    src = assets[asset]
    ob = bpy.data.objects.new(f"{asset}.{len(placements):04d}", src.data); col.objects.link(ob)
    ob.location = (x, y, z); ob.rotation_euler = (0, 0, math.radians(yaw)); ob.scale = (s, s, s)
    placements.append({'asset': asset, 'x': round(float(x), 2), 'y': round(float(y), 2), 'z': round(float(z), 2), 'yaw': round(float(yaw), 1), 'scale': round(float(s), 3)})
occupied = []
def free(x, y, mind):
    for q in occupied:
        if (x - q[0]) ** 2 + (y - q[1]) ** 2 < mind * mind: return False
    return True
def in_city(x, y, margin=16):
    return CITY_X[0] - margin <= x <= CITY_X[1] + margin and CITY_Y[0] - margin <= y <= CITY_Y[1] + margin
def zone(x, y):
    return dict(snow=sample(MASKS['snow'].astype(float), x, y) > 0.5, beach=sample(MASKS['beach'].astype(float), x, y) > 0.3,
                sea=sample(MASKS['sea'].astype(float), x, y) > 0.3, road=sample(MASKS['road'].astype(float), x, y) > 0.3,
                slope=sample(SLOPE, x, y), z=sample(H, x, y), valid=sample(MASKS['valid'].astype(float), x, y) > 0.5)

# a) GLB-derived candidates
n_cand = 0
for t in cands:
    x, y = t['x'], t['y']
    if in_city(x, y): continue
    zi = zone(x, y)
    if zi['sea'] or zi['road']: continue
    gz = ground_z(x, y)
    if gz < SEA + 1: continue
    s = float(np.clip(t['h'] / 32.0, 0.8, 1.35)); yaw = rng.uniform(0, 360)
    c = t['cls']
    if zi['snow'] or (c in (2, 6) and gz > 40 and x < -150):
        a = 'TreeSnowConifer_A' if rng.random() < 0.5 else 'TreeSnowConifer_B'
    elif zi['beach'] or (gz < BEACH_Z + 12 and y > 420):
        a = 'TreePalm_A' if rng.random() < 0.5 else 'TreePalm_B'
    elif c == 3 and zi['slope'] > 20:
        a = rng.choice(['Rock_A', 'Rock_B', 'Rock_C']); s = rng.uniform(0.9, 1.6)
    elif c == 5:
        a = 'TreeConifer_A' if rng.random() < 0.6 else 'TreeConifer_B'
    else:
        u = rng.random(); a = 'TreeRound_A' if u < 0.45 else 'TreeRound_B' if u < 0.7 else 'TreeConifer_A' if u < 0.88 else 'TreeConifer_B'
    if not free(x, y, 18): continue
    place(a, x, y, gz - 0.5, yaw, s, C_ROCKS if a.startswith('Rock') else C_VEG); occupied.append((x, y)); n_cand += 1
print('placed from GLB candidates', n_cand)

# b) extra scatter on grassy hills (density scaled by EXTRA)
N = int(40000 * EXTRA)
cx = rng.uniform(X0, X0 + NX * CELL, N); cy = rng.uniform(Y0, Y0 + NY * CELL, N)
valid = sample(MASKS['valid'].astype(float), cx, cy) > 0.5; sea = sample(MASKS['sea'].astype(float), cx, cy) > 0.2
beach = sample(MASKS['beach'].astype(float), cx, cy) > 0.2; road = sample(MASKS['road'].astype(float), cx, cy) > 0.2
snow = sample(MASKS['snow'].astype(float), cx, cy) > 0.5; slope = sample(SLOPE, cx, cy); z = sample(H, cx, cy)
grass = sample(MASKS['grass_d'], cx, cy); rockd = sample(MASKS['rock_d'], cx, cy); dcity = sample(MASKS['dcity'], cx, cy)
ok = valid & ~sea & ~beach & ~road & (slope < 26) & (z > BEACH_Z + 10) & (dcity > 30) & (z < 300)
dens = np.where(snow, 0.55, np.clip(grass * 1.6, 0.25, 1.0)) * np.clip(1.4 - np.hypot(cx + 196, cy - 76) / 3000, 0.3, 1.0)
ok &= rng.random(N) < dens
pts = list(zip(cx[ok], cy[ok], np.nonzero(ok)[0])); rng.shuffle(pts)
n_extra = 0
for (x, y, i) in pts:
    if not free(x, y, 46 if snow[i] else 40): continue
    gz = ground_z(x, y); s = rng.uniform(0.85, 1.25); yaw = rng.uniform(0, 360)
    if snow[i]: a = 'TreeSnowConifer_A' if rng.random() < 0.5 else 'TreeSnowConifer_B'
    elif z[i] > 170: a = 'TreeConifer_B' if rng.random() < 0.7 else 'TreeRound_B'
    else:
        u = rng.random(); a = 'TreeRound_A' if u < 0.42 else 'TreeRound_B' if u < 0.62 else 'TreeConifer_A' if u < 0.8 else 'TreeConifer_B' if u < 0.9 else 'Bush_A'
    place(a, x, y, gz - 0.5, yaw, s, C_VEG); occupied.append((x, y)); n_extra += 1
print('extra scatter', n_extra)

# c) palms along the beach back edge and headlands
n_palm = 0
for _ in range(3000):
    x = rng.uniform(-900, 450); y = rng.uniform(470, 700)
    zi = zone(x, y)
    if zi['sea'] or in_city(x, y, 10) or not zi['valid']: continue
    gz = ground_z(x, y)
    if gz < SEA + 2 or gz > CITY_Z - 20 or zi['slope'] > 30: continue
    dc = sample(MASKS['dcity'], x, y)
    on_sand_back = zi['beach'] and gz > BEACH_Z - 1.5 and 12 < dc < 70
    headland = (not zi['beach']) and gz > SEA + 4 and sample(MASKS['dshore'], x, y) < 120 and dc > 12
    if not (on_sand_back or headland): continue
    if not free(x, y, 60): continue
    place('TreePalm_A' if rng.random() < 0.5 else 'TreePalm_B', x, y, gz - 0.5, rng.uniform(0, 360), rng.uniform(0.9, 1.2), C_VEG); occupied.append((x, y)); n_palm += 1
    if n_palm >= 40: break
print('palms', n_palm)

# d) rocks: headlands / beach ends / steep-foot
n_rock = 0
for _ in range(40000):
    x = rng.uniform(X0, X0 + NX * CELL); y = rng.uniform(Y0, Y0 + NY * CELL)
    zi = zone(x, y)
    if not zi['valid'] or zi['sea'] or in_city(x, y, 10) or zi['road']: continue
    gz = ground_z(x, y)
    if gz < SEA - 2: continue
    near_sea = sample(MASKS['dshore'], x, y) < 110
    in_beach_x = (-773 - 20 < x < 293 + 20) and 484 < y < 720
    on_sand = zi['beach'] and in_beach_x and gz < BEACH_Z + 4
    beach_end = zi['beach'] and not in_beach_x and gz > SEA + 2          # sand beyond the city's beach: boulders ok
    headland = near_sea and not zi['beach'] and SEA - 1 < gz < 60
    cliff_base = (-773 < x < 293) and 486 < y < 520 and gz < -20 and rng.random() < 0.35   # foot of the city cliff
    foot = 10 < zi['slope'] < 32 and gz > 10 and not near_sea and rng.random() < 0.12
    wall_foot = sample(FOOT.astype(float), x, y) > 0.5 and gz > BEACH_Z + 4 and rng.random() < 0.6
    if on_sand and not cliff_base: continue
    if not ((headland and rng.random() < 0.8) or beach_end or cliff_base or foot or wall_foot): continue
    if not free(x, y, 34 if not wall_foot else 26): continue
    a = rng.choice(['RockSnow_A', 'RockSnow_B']) if zi['snow'] else rng.choice(['Rock_A', 'Rock_B', 'Rock_C'])
    s_ = rng.uniform(0.6, 1.1) if wall_foot else rng.uniform(0.7, 1.5)
    place(a, x, y, gz - 1.5, rng.uniform(0, 360), s_, C_ROCKS); occupied.append((x, y)); n_rock += 1
    if n_rock >= 360: break
print('rocks', n_rock, 'total instances', len(placements))

# ------------------------------------------------------------------ backdrop islands (hazy, far)
def island(name, cx, cy, peaks, seed, color='#86B8AE', color2='#9CC9C4'):
    m = MB(); rr = np.random.default_rng(seed)
    for k, (px, py, ph, rb) in enumerate(peaks):
        m.cone(cx + px, cy + py, SEA - 6, rb, 0, ph + 6, 9, hexc(color if k % 2 == 0 else color2), jitter=rr.uniform(0, 1))
    return m.finish(name, C_ISL, jitter=0.03, seed=seed)
island('Island_A', -4300, -4200, [(-250, 120, 300, 750), (380, -80, 220, 600)], 31)
island('Island_B', -6400, -3200, [(-500, 150, 380, 950), (350, -100, 260, 700)], 32)
island('Island_C', 3600, -3800, [(0, 0, 480, 1300), (900, 300, 330, 850)], 33, color='#8BBF9A', color2='#9FCBA8')

for lc in bpy.context.view_layer.layer_collection.children:
    if lc.collection.name == 'Assets': lc.exclude = True
bboxes = {}
for n, ob in assets.items():
    co = np.array([v.co[:] for v in ob.data.vertices]); bboxes[n] = {'min': co.min(0).round(2).tolist(), 'max': co.max(0).round(2).tolist(), 'tris': sum(len(p.vertices) - 2 for p in ob.data.polygons)}
json.dump({'axes': 'blender Z-up; roblox = (x, z, -y)', 'assets': bboxes, 'instances': placements}, open(W_ + 'placements.json', 'w'), indent=0)
bpy.ops.wm.save_mainfile(filepath=BLEND)
print("saved", BLEND)
