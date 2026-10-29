"""Level the GLB (undo the camera-space tilt), then fit rotation/scale/translation onto the Context city.
usage: python3 -I align_glb.py <workdir>
writes transform.json, lvl_H.npy / lvl_C.npy / lvl_grid.npy (levelled-frame rasters), preview PNGs.
"""
import sys, json, numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
W_ = sys.argv[1].rstrip('/') + '/'
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glbnp import sample_tex

# Context city facts (from HANDOFF.md)
CITY_X = (-764.0, 372.0); CITY_Y = (-332.0, 484.0)
CITY_CX, CITY_CY = (CITY_X[0] + CITY_X[1]) / 2, (CITY_Y[0] + CITY_Y[1]) / 2
CITY_W, CITY_D = CITY_X[1] - CITY_X[0], CITY_Y[1] - CITY_Y[0]
WATERLINE_Y = 600.0; BEACH_FRONT_Y = 492.0

P = np.load(W_ + 'glb_pos.npy'); UV = np.load(W_ + 'glb_uv.npy'); T = np.load(W_ + 'glb_tri.npy')
tex = np.asarray(Image.open(W_ + 'glb_tex.png').convert('RGB')); vcol = sample_tex(tex, UV)

def classify(C):
    """C float 0..1 (N,3) -> dict of masks"""
    R, G, B = C[..., 0], C[..., 1], C[..., 2]
    V = C.max(-1); S = (V - C.min(-1)) / (V + 1e-6)
    hue = np.degrees(np.arctan2(np.sqrt(3) * (G - B), 2 * R - G - B)) % 360
    return {'green': (hue > 60) & (hue < 160) & (S > 0.25) & (V > 0.2),
            'water': (hue > 170) & (hue < 230) & (S > 0.35) & (V > 0.4),
            'snow': (V > 0.72) & (S < 0.2),
            'sand': (hue > 30) & (hue < 60) & (S > 0.15) & (S < 0.6) & (V > 0.6),
            'dark': V < 0.30}

def rasterise(Pz, res, up_v, vcol):
    x0, y0 = Pz[:, 0].min(), Pz[:, 1].min()
    NX = int((Pz[:, 0].max() - x0) / res) + 2; NY = int((Pz[:, 1].max() - y0) / res) + 2
    ix = ((Pz[:, 0] - x0) / res).astype(int); iy = ((Pz[:, 1] - y0) / res).astype(int)
    H = np.full((NY, NX), np.nan); np.fmax.at(H, (iy[up_v], ix[up_v]), Pz[up_v, 2])
    order = np.argsort(Pz[up_v, 2]); Cm = np.zeros((NY, NX, 3), np.uint8)
    Cm[iy[up_v][order], ix[up_v][order]] = vcol[up_v][order]
    foot = np.zeros((NY, NX), bool); foot[iy, ix] = True; foot = ndimage.binary_closing(foot, iterations=2)
    miss = np.isnan(H)
    idx = ndimage.distance_transform_edt(miss, return_distances=False, return_indices=True)
    H = H[tuple(idx)]; Cm = Cm[tuple(idx)]
    return H, Cm, foot, (x0, y0, res, NX, NY)

# ---------------------------------------------------------------- 1. level: rotate about X so the sea is horizontal
A, B, C3 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
N = np.cross(B - A, C3 - A); nz = N[:, 2] / (np.linalg.norm(N, axis=1) + 1e-12)
up_v = np.zeros(len(P), bool); up_v[T[nz > 0.05].ravel()] = True
# sea plane in the raw frame (water-coloured top cells at the front)
H0, C0, F0, g0 = rasterise(P, 0.0025, up_v, vcol)
cl = classify(C0.astype(float) / 255)
x0, y0, res, NX, NY = g0
ys, xs = np.nonzero(cl['water'] & F0); X = x0 + xs * res; Y = y0 + ys * res; Z = H0[ys, xs]
m = (Y > 0.2) & (Z < 0.0)
Am = np.c_[X[m], Y[m], np.ones(m.sum())]; coef, *_ = np.linalg.lstsq(Am, Z[m], rcond=None)
r = Z[m] - Am @ coef; keep = np.abs(r) < 2 * r.std()
coef, *_ = np.linalg.lstsq(Am[keep], Z[m][keep], rcond=None)
tilt_x = float(np.arctan(coef[1])); tilt_y = float(np.arctan(coef[0]))
print('raw sea plane dz/dx=%.4f dz/dy=%.4f -> tilt about X %.2f deg, about Y %.2f deg' % (coef[0], coef[1], np.degrees(tilt_x), np.degrees(tilt_y)))
# rotation about X by +theta (theta = -tilt_x, since dz/dy<0 -> sea lower at +Y): makes the plane normal vertical
theta = -tilt_x
def Rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
def Rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
P1 = P @ Rx(theta).T
H1, C1, F1, g1 = rasterise(P1, 0.0025, up_v, vcol)
x0, y0, res, NX, NY = g1
cl = classify(C1.astype(float) / 255)
ys, xs = np.nonzero(cl['water'] & F1); X = x0 + xs * res; Y = y0 + ys * res; Z = H1[ys, xs]
m = (Y > 0.2); Am = np.c_[X[m], Y[m], np.ones(m.sum())]; coef, *_ = np.linalg.lstsq(Am, Z[m], rcond=None)
r = Z[m] - Am @ coef; keep = np.abs(r) < 2 * r.std(); coef, *_ = np.linalg.lstsq(Am[keep], Z[m][keep], rcond=None)
z_sea = float(coef[2] + coef[0] * X[m][keep].mean() + coef[1] * Y[m][keep].mean())
print('levelled sea plane dz/dx=%.4f dz/dy=%.4f z_sea=%.4f (resid std %.4f)' % (coef[0], coef[1], z_sea, r[keep].std()))

# ---------------------------------------------------------------- 2. ground (opened) + urban blob in the levelled frame
rr = 8; yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]; disk = (xx * xx + yy * yy) <= rr * rr
G1 = ndimage.grey_opening(H1, footprint=disk)
urban = F1 & ~cl['green'] & ~cl['water'] & ~cl['snow'] & ~cl['sand'] & (G1 < z_sea + 0.06)
u = ndimage.uniform_filter(urban.astype(float), 7) > 0.5
u = ndimage.binary_closing(u, iterations=3)
lab, n = ndimage.label(u); sizes = ndimage.sum(u, lab, range(1, n + 1)); u = lab == (1 + int(np.argmax(sizes)))
u = ndimage.binary_fill_holes(u)
ys, xs = np.nonzero(u); X = x0 + xs * res; Y = y0 + ys * res
z_city = float(np.median(G1[u])); print('urban cells', u.sum(), 'city floor z (levelled, median of opened ground)', round(z_city, 4), 'p10/p90', np.percentile(G1[u], [10, 90]).round(4))

# ---------------------------------------------------------------- 3. coastline angle: front waterline (water/sand boundary)
water = cl['water'] & F1; sand = cl['sand'] & F1
# for each column, the lowest-Y water cell in the front sea region (Y > city centre)
cyc = Y.mean()
wl_x = []; wl_y = []
for i in range(NX):
    col = np.nonzero(water[:, i] & (y0 + np.arange(NY) * res > cyc))[0]
    scol = np.nonzero(sand[:, i])[0]
    if len(col) > 20 and len(scol) > 5:
        # waterline = first water row above the last sand row... use the 10th percentile of water rows
        wl_x.append(x0 + i * res); wl_y.append(y0 + np.percentile(col, 5) * res)
wl_x = np.array(wl_x); wl_y = np.array(wl_y)
# robust line fit (only the central part in X, where the beach is)
sel = (wl_x > X.min() + 0.05) & (wl_x < X.max() - 0.05)
cf = np.polyfit(wl_x[sel], wl_y[sel], 1); resid = wl_y[sel] - np.polyval(cf, wl_x[sel]); keep = np.abs(resid) < 2 * resid.std()
cf = np.polyfit(wl_x[sel][keep], wl_y[sel][keep], 1)
coast_ang = float(np.arctan(cf[0])); print('front waterline: slope %.4f -> angle %.2f deg; mean Y %.4f' % (cf[0], np.degrees(coast_ang), np.polyval(cf, X.mean())))
psi = -coast_ang  # rotate so the coast runs along +X
P2 = P1 @ Rz(psi).T
Xr = X * np.cos(psi) - Y * np.sin(psi); Yr = X * np.sin(psi) + Y * np.cos(psi)
ub = np.percentile(Xr, [0.5, 99.5]); vb = np.percentile(Yr, [0.5, 99.5])
print('urban blob after coast-alignment: X[%.4f, %.4f] (w %.4f)  Y[%.4f, %.4f] (d %.4f)' % (ub[0], ub[1], ub[1] - ub[0], vb[0], vb[1], vb[1] - vb[0]))
wl_mean = float(np.polyval(cf, X.mean())) * np.cos(psi) + X.mean() * np.sin(psi)
# front edge of the urban blob (99th pct of Y of blob = toward the sea) and its relation to the waterline
front = vb[1]; back = vb[0]
print('urban front edge Y %.4f, waterline Y %.4f -> beach depth %.4f units' % (front, wl_mean, wl_mean - front))

# ---------------------------------------------------------------- 4. scale + translation
S_w = CITY_W / (ub[1] - ub[0]); S_d = CITY_D / (vb[1] - vb[0]); S_beach = (WATERLINE_Y - CITY_Y[1]) / (wl_mean - front)
print('scale candidates: from width %.1f, from depth %.1f, from beach depth %.1f' % (S_w, S_d, S_beach))
S = float(np.sqrt(S_w * S_d))  # geometric mean of width/depth
tx = CITY_CX - (ub[0] + ub[1]) / 2 * S
# place the urban front edge on the Context front edge (beach starts right there)
ty = CITY_Y[1] - front * S
print('chosen S=%.1f tx=%.1f ty=%.1f -> urban blob maps to X[%.0f, %.0f] Y[%.0f, %.0f]; waterline -> Y=%.0f' % (S, tx, ty, ub[0] * S + tx, ub[1] * S + tx, vb[0] * S + ty, vb[1] * S + ty, wl_mean * S + ty))
# beach level in the GLB: sand cells' median ground
sand_r = sand & (G1 < z_city) & (G1 > z_sea)
z_beach = float(np.median(G1[sand_r])) if sand_r.sum() else (z_sea + z_city) / 2
peak = float(np.nanpercentile(H1[F1], 99.9))
print('levels (levelled frame): sea %.4f beach %.4f city %.4f peak99.9 %.4f ; city-sea %.4f, peak-city %.4f' % (z_sea, z_beach, z_city, peak, z_city - z_sea, peak - z_city))
print('with uniform S: city above sea = %.0f studs (Context: 53), peak above city = %.0f studs' % ((z_city - z_sea) * S, (peak - z_city) * S))

tr = {'theta_x_deg': float(np.degrees(theta)), 'psi_z_deg': float(np.degrees(psi)), 'S': S, 'tx': float(tx), 'ty': float(ty),
      'z_sea': z_sea, 'z_beach': z_beach, 'z_city': z_city, 'z_peak': peak,
      'urban_box_levelled': {'X': ub.tolist(), 'Y': vb.tolist()}, 'waterline_y_levelled': wl_mean,
      'note': 'world = Rz(psi) Rx(theta) p ; xy *= S, += (tx,ty); z mapped separately (see heightfield stage)'}
json.dump(tr, open(W_ + 'transform.json', 'w'), indent=1); print(json.dumps(tr, indent=1))
np.save(W_ + 'lvl_H.npy', H1); np.save(W_ + 'lvl_C.npy', C1); np.save(W_ + 'lvl_G.npy', G1); np.save(W_ + 'lvl_foot.npy', F1); np.save(W_ + 'lvl_urban.npy', u)
np.save(W_ + 'lvl_grid.npy', np.array([x0, y0, res, NX, NY])); np.save(W_ + 'glb_pos_aligned_xy.npy', P2)

# preview: levelled colour map with urban blob outline, waterline, and the Context rectangle mapped back
img = C1.copy(); img[~F1] = 0
im = Image.fromarray(img[::-1]).resize((NX * 2, NY * 2), Image.NEAREST); d = ImageDraw.Draw(im)
def to_px(xw, yw): return ((xw - x0) / res * 2, (NY - (yw - y0) / res) * 2)
# Context rectangle corners in world -> levelled frame: inverse of xy transform
corners = [(CITY_X[0], CITY_Y[0]), (CITY_X[1], CITY_Y[0]), (CITY_X[1], CITY_Y[1]), (CITY_X[0], CITY_Y[1])]
pts = []
for cx, cy in corners:
    xr = (cx - tx) / S; yr = (cy - ty) / S
    xl = xr * np.cos(-psi) - yr * np.sin(-psi); yl = xr * np.sin(-psi) + yr * np.cos(-psi)
    pts.append(to_px(xl, yl))
d.polygon(pts, outline=(255, 255, 255), width=2)
for xw in np.linspace(-1300, 800, 8):  # waterline in world -> levelled
    xr = (xw - tx) / S; yr = (WATERLINE_Y - ty) / S
    xl = xr * np.cos(-psi) - yr * np.sin(-psi); yl = xr * np.sin(-psi) + yr * np.cos(-psi)
    d.ellipse([to_px(xl, yl)[0] - 2, to_px(xl, yl)[1] - 2, to_px(xl, yl)[0] + 2, to_px(xl, yl)[1] + 2], fill=(255, 255, 0))
im.save(W_ + 'align_preview.png')
lo, hi = z_sea - 0.01, peak
gn = ((H1 - lo) / (hi - lo)).clip(0, 1); gn[~F1] = 0
Image.fromarray((gn[::-1] * 255).astype(np.uint8)).resize((NX * 2, NY * 2), Image.NEAREST).save(W_ + 'align_height_levelled.png')
