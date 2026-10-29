"""Turn the aligned GLB heightfield into a clean terrain heightfield + masks + tree candidates.
usage: python3 -I process_terrain.py <workdir> [exag] [terrace_step]
v3: terrace sharpening (steep hillsides -> flat grass shelves + near-vertical walls, 45-stud steps; 0 disables it)
"""
import sys, json, numpy as np
from scipy import ndimage
from PIL import Image
W_ = sys.argv[1].rstrip('/') + '/'; EXAG = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0   # terrace step (studs): 45 = the painting's one-tree walls, 60 = fewer, deeper shelves; 0 = v2 behaviour
tr = json.load(open(W_ + 'transform.json')); g = json.load(open(W_ + 'hf_grid.json'))
CELL = g['cell']; X0, Y0, NX, NY = g['x0'], g['y0'], g['nx'], g['ny']
H = np.load(W_ + 'hf_raw.npy'); valid = np.load(W_ + 'hf_valid.npy'); cls = np.load(W_ + 'hf_cls.npy'); col = np.load(W_ + 'hf_col.npy')
S = tr['S']; z_sea, z_city = tr['z_sea'], tr['z_city']
SEA, BEACH_Z, CITY_Z = -54.0, -46.0, -1.0
CITY_X = (-764.0, 372.0); CITY_Y = (-332.0, 484.0)
BEACH_X = (-773.0, 293.0); BEACH_Y0 = 492.0
xs = X0 + np.arange(NX) * CELL; ys = Y0 + np.arange(NY) * CELL; XX, YY = np.meshgrid(xs, ys)
def disk(r):
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]; return (xx * xx + yy * yy) <= r * r

# ---------------------------------------------------------------- 1. vertical mapping (piecewise)
def to_world(z):
    low = SEA + (z - z_sea) * ((CITY_Z - SEA) / (z_city - z_sea))
    d = (z - z_city) * S
    f = 1.0 + (EXAG - 1.0) * np.clip(d / 400.0, 0, 1)   # progressive: unchanged near the city, x EXAG for d >= 400
    high = CITY_Z + d * f
    return np.where(z < z_city, low, high)
Hw = to_world(np.nan_to_num(H, nan=z_sea))
print('EXAG', EXAG, 'world z percentiles (valid):', np.percentile(Hw[valid], [1, 25, 50, 75, 95, 99, 99.9]).round(1), 'max', Hw[valid].max().round(1))

# ---------------------------------------------------------------- 2. sea region: water-coloured low cells, closed over the pier/boats
water_c = (cls == 6) & (Hw < BEACH_Z + 6) & valid
water_c = ndimage.binary_opening(water_c, iterations=1)
water_c = ndimage.binary_closing(water_c, structure=disk(7))  # 28-stud radius closes the pier (40 wide) and boats
lab, n = ndimage.label(water_c); sizes = ndimage.sum(water_c, lab, range(1, n + 1))
water_c = np.isin(lab, 1 + np.nonzero(sizes * CELL * CELL > 8000)[0])
water_c = ndimage.binary_fill_holes(water_c)
# the front sea also includes everything (valid or not) beyond the GLB waterline at the top: cells with Y > 660 outside land
sea_mask = water_c | (~valid & (YY > 560))

# ---------------------------------------------------------------- 3. props
r_small = int(round(12 / CELL)); r_big = int(round(28 / CELL))
Hn = Hw.copy(); Hn[~valid] = SEA - 20
G12 = ndimage.grey_opening(Hn, footprint=disk(r_small)); G28 = ndimage.grey_opening(Hn, footprint=disk(r_big))
ra = int(round(1.3 * r_big)); yy, xx = np.ogrid[-ra - 1:ra + 2, -ra - 1:ra + 2]; d2 = xx * xx + yy * yy
ring = (d2 >= (ra - 0.5) ** 2) & (d2 <= (ra + 1.5) ** 2)
ringmax = ndimage.maximum_filter(Hn, footprint=ring)
D28 = Hn - G28
bump = (D28 > 5) & (Hn - ringmax > 4)
lab, n = ndimage.label(D28 > 3)
ids = np.arange(1, n + 1)
areas = np.array(ndimage.sum(D28 > 3, lab, ids)) * CELL * CELL
cores = np.array(ndimage.sum(bump, lab, ids))
maxd = np.array(ndimage.maximum(D28, lab, ids))
greenfrac = np.array(ndimage.mean(((cls == 1) | (cls == 5)).astype(float), lab, ids))  # canopy: green or dark-green/shadow
ncell = areas / (CELL * CELL)
is_prop = ((cores > 0) & ~((areas > 9000) & (cores / np.maximum(ncell, 1) < 0.25))) | ((greenfrac > 0.45) & (maxd < 70) & (areas < 60000))
prop = np.isin(lab, ids[is_prop]) & (Hn < 230) & ~sea_mask
thin = (Hn - G12 > 4) & (Hn < 230) & ~sea_mask
print('prop cells', prop.sum(), 'thin cells', thin.sum(), 'prop blobs', is_prop.sum(), 'of', n)
ground = Hw.copy()
ground[thin] = G12[thin]
ground[prop] = np.minimum(ground[prop], G28[prop])

# tree candidates from prop blobs (recomputed on the final prop mask)
labp, npb = ndimage.label(prop); idp = np.arange(1, npb + 1)
areas = np.array(ndimage.sum(prop, labp, idp)) * CELL * CELL
cyx = np.array(ndimage.center_of_mass(prop, labp, idp)).reshape(-1, 2)
hmax = np.array(ndimage.maximum(D28, labp, idp))
trees = []
rng = np.random.default_rng(7)
for k in range(npb):
    a = areas[k]
    if a < 60 or hmax[k] < 7: continue
    m = labp == k + 1
    cc = np.bincount(cls[m], minlength=8); c_top = int(np.argmax(cc[1:]) + 1)
    if a > 2000:
        yy_, xx_ = np.nonzero(m); pts = np.stack([X0 + xx_ * CELL, Y0 + yy_ * CELL], 1)
        nk = max(1, int(round(a / 1300)))
        # greedy spacing ~32 studs
        sel = pts[rng.permutation(len(pts))]; keep = []
        for p in sel:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 32 ** 2 for q in keep): keep.append(p)
            if len(keep) >= nk: break
        for p in keep:
            j, i = int((p[1] - Y0) / CELL), int((p[0] - X0) / CELL)
            trees.append({'x': float(p[0]), 'y': float(p[1]), 'z': float(ground[j, i]), 'h': float(min(D28[j, i] + 8, hmax[k])), 'r': 18.0, 'cls': int(cls[j, i]) or c_top, 'area': float(a)})
    else:
        trees.append({'x': float(X0 + cyx[k][1] * CELL), 'y': float(Y0 + cyx[k][0] * CELL), 'z': float(np.median(ground[m])), 'h': float(hmax[k]), 'r': float(np.sqrt(a / np.pi)), 'cls': c_top, 'area': float(a)})
print('tree candidates', len(trees), 'by class', np.bincount([t['cls'] for t in trees], minlength=8).tolist())

# ---------------------------------------------------------------- 4. city plateau
MARGIN, BLEND = 16.0, (36.0 if STEP > 0 else 60.0)   # v3: shorter blend so the first wall can rise right behind the city
dx = np.maximum(np.maximum(CITY_X[0] - XX, XX - CITY_X[1]), 0); dy = np.maximum(np.maximum(CITY_Y[0] - YY, YY - CITY_Y[1]), 0)
dcity = np.hypot(dx, dy)
w = np.clip((dcity - MARGIN) / BLEND, 0, 1)
above = ground > CITY_Z
ground = np.where((w < 1) & above, ground * w + CITY_Z * (1 - w), ground)   # blend only where terrain is higher (back/sides)
ground = np.where(dcity <= MARGIN, CITY_Z, ground)
# front cliff: under the user's beach parts (Y >= 492, X in beach range) the terrain must be <= -46
front = (XX >= BEACH_X[0]) & (XX <= BEACH_X[1]) & (YY > CITY_Y[1])
ramp = np.clip((YY - CITY_Y[1]) / (BEACH_Y0 - CITY_Y[1]), 0, 1)  # 0 at 484 -> 1 at 492
cliff_top = CITY_Z * (1 - ramp) + BEACH_Z * ramp
ground = np.where(front, np.minimum(ground, cliff_top), ground)
# ---------------------------------------------------------------- 4b. terrace sharpening (v3) - after the plateau blend so the ramp behind the city is terraced too
# Steep hillsides (outside plateau / beach / sea, below the peaks) are snapped to the nearest of the global shelf levels
# CITY_Z + k*STEP with a short ramp, so walls become near-vertical and shelves flat with crisp tops; the shelves keep a
# little of the original relief so they tilt gently instead of reading as platforms.
RAMP, SHELF_KEEP, TERR_LO, TERR_HI = 0.12, 0.12, 26.0, 35.0
tw = np.zeros_like(ground)
if STEP > 0:
    gs = ndimage.gaussian_filter(ground, 1.5)                                   # quantise a lightly smoothed surface
    gy, gx = np.gradient(ndimage.gaussian_filter(ground, 3.0), CELL)
    slope_l = np.degrees(np.arctan(np.hypot(gx, gy)))                           # slope at the ~12-stud scale
    tw = np.clip((slope_l - TERR_LO) / (TERR_HI - TERR_LO), 0, 1)              # 26 deg -> 35 deg: fully terraced
    relief = ndimage.maximum_filter(gs, footprint=disk(10)) - ndimage.minimum_filter(gs, footprint=disk(10))
    tw *= np.clip((relief - 24.0) / 16.0, 0, 1)                                 # isolated knolls (< ~40 studs of relief within 40 studs) stay round, no stacked discs
    tw *= np.clip((250.0 - gs) / 40.0, 0, 1)                                    # mountains / snow peaks above ~250 keep their facets
    tw *= np.clip((gs - (BEACH_Z + 3.0)) / 6.0, 0, 1)                           # not on the beach / sea floor
    tw[sea_mask] = 0
    tw[(dcity <= MARGIN) | front] = 0                                           # plateau stays flat, the front cliff keeps its -1 -> -46 profile
    t = (gs - CITY_Z) / STEP; k = np.floor(t); f = t - k
    g = np.clip((f - 0.5) / RAMP + 0.5, 0, 1)
    terr = CITY_Z + (k + g) * STEP
    terr = terr * (1 - SHELF_KEEP) + gs * SHELF_KEEP
    ground = ground * (1 - tw) + terr * tw
    ground = np.where(dcity <= MARGIN, CITY_Z, ground)
    ground = np.where(front, np.minimum(ground, cliff_top), ground)
    print('terrace cells (w>0.5)', int((tw > 0.5).sum()), 'levels', int(k[tw > 0.5].min()), '..', int(k[tw > 0.5].max()))

# ---------------------------------------------------------------- 5. beach + sea floor
land = valid & ~sea_mask
dsea = ndimage.distance_transform_edt(~land) * CELL      # on sea cells: distance to land
dshore = ndimage.distance_transform_edt(land) * CELL     # on land cells: distance to the sea
seafloor = SEA - (6 + np.minimum(dsea * 0.08, 24))
ground = np.where(sea_mask, seafloor, ground)
# beach zone: every low cell on the sea side (sand, wet sand, hut footprints, shore rocks' feet) between the land and the water
sand_c = (cls == 4) & valid & ~sea_mask & (ground < CITY_Z - 5)
lowfront = valid & ~sea_mask & (YY > 440) & (ground < CITY_Z - 15) & (dshore < 260) & (dcity > 0)
beach = ndimage.binary_closing(sand_c | lowfront, iterations=2) & ~sea_mask & valid & (ground < CITY_Z - 5)
# beach profile: -54 at the water's edge rising to -46 within ~70 studs inland, GLB detail of +-1.5 on top
prof = SEA + (BEACH_Z - SEA) * np.clip(dshore / 70.0, 0, 1)
ground = np.where(beach, np.clip(ground, prof - 1.5, prof + 1.5), ground)
# ---------------------------------------------------------------- 6. extension beyond the footprint: smooth skirt decaying to the sea floor
inv = ~valid & ~sea_mask
dist = ndimage.distance_transform_edt(inv) * CELL
src = np.where(~inv, ground, 0.0); wgt = (~inv).astype(float)
near_s = np.zeros_like(ground)
# multi-scale normalised convolution: further out, broader blur of the edge values
for sig_c in (2, 5, 10, 20, 40, 80):
    num = ndimage.gaussian_filter(src, sig_c); den = ndimage.gaussian_filter(wgt, sig_c)
    cand = num / np.maximum(den, 1e-6)
    band = inv & (dist >= sig_c * CELL * 0.6) & (den > 1e-3)
    near_s = np.where(band, cand, near_s)
# cells closer than the first band: plain nearest value
_, idx = ndimage.distance_transform_edt(inv, return_indices=True)
near_s = np.where(inv & (dist < 2 * CELL * 0.6), ground[tuple(idx)], near_s)
decay = np.exp(-dist / 220.0)
ext = near_s * decay + (SEA - 30) * (1 - decay)
ground = np.where(inv, ext, ground)
sm = ndimage.gaussian_filter(ground, 2.0)
seam = ndimage.binary_dilation(inv, iterations=4)
ground = np.where(seam, sm, ground)
ground = np.where((dcity <= MARGIN) & ~front, CITY_Z, ground)
ground = np.where(front, np.minimum(ground, cliff_top), ground)   # v3 fix: v2 re-flattened the 16-stud margin here, pushing the front cliff out to Y 500-504
ground = np.where(sea_mask & (dsea > 2 * CELL), np.minimum(ground, SEA - 4), ground)
print('final ground percentiles', np.percentile(ground, [1, 5, 25, 50, 75, 95, 99, 99.9]).round(1), 'max', ground.max().round(1))

# ---------------------------------------------------------------- masks for colouring
# nearest-valid index for extending masks into the extension region
_, nidx = ndimage.distance_transform_edt(~valid, return_indices=True)
cls = np.where(valid, cls, cls[tuple(nidx)])
high_blue = (cls == 6) & (ground > 60)          # bluish shadows on the icy mountains / hazy far peaks
snow_c = ((cls == 2) | (high_blue & (XX < -150))) & ~sea_mask & (ground > 40)
snow_d = ndimage.gaussian_filter(snow_c.astype(float), 4)
snow_zone = (snow_d > 0.18) & (XX < 0)
snow_zone = ndimage.binary_closing(snow_zone, iterations=4)
lab, n = ndimage.label(snow_zone); sizes = ndimage.sum(snow_zone, lab, range(1, n + 1))
snow_zone = np.isin(lab, 1 + np.nonzero(sizes * CELL * CELL > 20000)[0])
# icy district ground (snow at plateau level on the -X side) -> snow colour too, but within the footprint it's the user's city
road_c = (cls == 5) & valid & ~sea_mask & (dcity > MARGIN) & (ground > CITY_Z + 2) & (XX > 250) & (YY > -400) & (ground < 220)
road_d = ndimage.gaussian_filter(road_c.astype(float), 1.5)
road_zone = road_d > 0.35
lab, n = ndimage.label(road_zone); sizes = ndimage.sum(road_zone, lab, range(1, n + 1))
road_zone = np.isin(lab, 1 + np.nonzero(sizes * CELL * CELL > 4000)[0])
grass_d = ndimage.gaussian_filter((cls == 1).astype(float), 2)
rock_d = ndimage.gaussian_filter(((cls == 3) | (cls == 5)).astype(float), 2)
np.save(W_ + 'terrain_h.npy', ground.astype(np.float32))
np.savez(W_ + 'terrain_masks.npz', plateau=(dcity <= MARGIN), dcity=dcity, beach=beach, sea=sea_mask, snow=snow_zone, snow_d=snow_d, road=road_zone,
         grass_d=grass_d, rock_d=rock_d, valid=valid, prop=prop, dshore=dshore, terrace_w=tw.astype(np.float32))
json.dump({'cell': CELL, 'x0': X0, 'y0': Y0, 'nx': NX, 'ny': NY, 'sea': SEA, 'beach': BEACH_Z, 'city': CITY_Z, 'exag': EXAG,
           'city_x': CITY_X, 'city_y': CITY_Y, 'margin': MARGIN, 'terrace_step': STEP, 'shelf_keep': SHELF_KEEP if STEP > 0 else 0.0}, open(W_ + 'terrain_meta.json', 'w'), indent=1)
json.dump(trees, open(W_ + 'tree_candidates.json', 'w'))
print('snow zone cells', snow_zone.sum(), 'road zone cells', road_zone.sum(), 'beach cells', beach.sum(), 'sea cells', sea_mask.sum())

lo, hi = SEA - 40, ground.max()
hn = ((ground - lo) / (hi - lo)).clip(0, 1)
img = np.stack([hn * 255] * 3, -1).astype(np.uint8)
img[sea_mask] = (img[sea_mask] * 0.5 + np.array([0, 60, 160]) * 0.5).astype(np.uint8)
img[beach] = (img[beach] * 0.5 + np.array([200, 180, 60]) * 0.5).astype(np.uint8)
img[snow_zone] = (img[snow_zone] * 0.6 + np.array([120, 160, 255]) * 0.4).astype(np.uint8)
img[road_zone] = (255, 0, 255)
img[prop] = (img[prop] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
for t in trees:
    j, i = int((t['y'] - Y0) / CELL), int((t['x'] - X0) / CELL); img[j, i] = (0, 255, 0)
Image.fromarray(img[::-1]).save(W_ + 'terrain_preview.png')
Image.fromarray((hn[::-1] * 255).astype(np.uint8)).save(W_ + 'terrain_height.png')
