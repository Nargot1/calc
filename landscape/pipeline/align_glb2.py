"""Refine the alignment: cliff line (beach -> plateau edge) per column, scale from blob width.
usage: python3 -I align_glb2.py <workdir> [S_override]
"""
import sys, json, numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
W_ = sys.argv[1].rstrip('/') + '/'
tr = json.load(open(W_ + 'transform.json'))
H1 = np.load(W_ + 'lvl_H.npy'); C1 = np.load(W_ + 'lvl_C.npy'); G1 = np.load(W_ + 'lvl_G.npy'); F1 = np.load(W_ + 'lvl_foot.npy'); u = np.load(W_ + 'lvl_urban.npy')
x0, y0, res, NX, NY = np.load(W_ + 'lvl_grid.npy'); NX = int(NX); NY = int(NY)
CITY_X = (-764.0, 372.0); CITY_Y = (-332.0, 484.0); WATERLINE_Y = 600.0
z_sea, z_city = tr['z_sea'], tr['z_city']; psi = np.radians(tr['psi_z_deg'])
cp, sp = np.cos(psi), np.sin(psi)
xs_w = x0 + np.arange(NX) * res; ys_w = y0 + np.arange(NY) * res

# cliff line: per column, scanning from the sea (high Y) inland, first cell where opened ground >= z_city - 0.006,
# restricted to the beach X-range (central part of the island)
cliff = []
thr = z_city - 0.006
for i in range(NX):
    col = G1[:, i]; f = F1[:, i]
    rows = np.nonzero(f)[0]
    if len(rows) < 20: continue
    top = rows.max()
    # must start in water/beach: ground near the top below the threshold
    if col[top - 3: top + 1].max() >= thr: continue
    j = top
    while j > 0 and col[j] < thr: j -= 1
    if j <= 0: continue
    cliff.append((xs_w[i], ys_w[j]))
cliff = np.array(cliff)
# rotate the cliff points by psi (coast alignment), keep the central beach span, fit the line
cx = cliff[:, 0] * cp - cliff[:, 1] * sp; cy = cliff[:, 0] * sp + cliff[:, 1] * cp
ub = tr['urban_box_levelled']['X']
sel = (cx > ub[0] + 0.03) & (cx < ub[1] - 0.03)
med = np.median(cy[sel]); mad = np.median(np.abs(cy[sel] - med))
keep = sel & (np.abs(cy - med) < 4 * mad + 1e-6)
cliff_y = float(np.median(cy[keep]))
print('cliff line: %d columns, median Y %.4f (MAD %.4f); X span of kept columns [%.3f, %.3f]' % (keep.sum(), cliff_y, mad, cx[keep].min(), cx[keep].max()))
wl = tr['waterline_y_levelled']
print('waterline Y %.4f -> beach depth %.4f units' % (wl, wl - cliff_y))
# beach X extent: columns with a cliff and sand in front of it
sand_cols = cx[keep]; print('beach X extent (levelled, coast-aligned): [%.3f, %.3f] = %.3f units' % (sand_cols.min(), sand_cols.max(), sand_cols.max() - sand_cols.min()))

S = float(sys.argv[2]) if len(sys.argv) > 2 else 1136.0 / (ub[1] - ub[0])
tx = (CITY_X[0] + CITY_X[1]) / 2 - (ub[0] + ub[1]) / 2 * S
ty = CITY_Y[1] - cliff_y * S
print('S=%.1f tx=%.1f ty=%.1f ; waterline -> Y=%.0f (Context 600) ; beach X -> [%.0f, %.0f] (Context -773..293)' % (S, tx, ty, wl * S + ty, sand_cols.min() * S + tx, sand_cols.max() * S + tx))
print('urban blob depth -> Y[%.0f, %.0f] (Context -332..484)' % (tr['urban_box_levelled']['Y'][0] * S + ty, tr['urban_box_levelled']['Y'][1] * S + ty))
# model extent in world after transform
xl = np.array([x0, x0 + NX * res]); yl = np.array([y0, y0 + NY * res])
corners = np.array([[xl[0], yl[0]], [xl[1], yl[0]], [xl[1], yl[1]], [xl[0], yl[1]]])
cw = np.c_[corners[:, 0] * cp - corners[:, 1] * sp, corners[:, 0] * sp + corners[:, 1] * cp] * S + [tx, ty]
print('model footprint in world: X[%.0f, %.0f] Y[%.0f, %.0f]' % (cw[:, 0].min(), cw[:, 0].max(), cw[:, 1].min(), cw[:, 1].max()))
print('vertical with uniform S: city above sea %.0f studs; peak above city %.0f studs' % ((z_city - z_sea) * S, (tr['z_peak'] - z_city) * S))
tr.update({'S': S, 'tx': float(tx), 'ty': float(ty), 'cliff_y_levelled': cliff_y})
json.dump(tr, open(W_ + 'transform.json', 'w'), indent=1)

# preview with cliff line + context rect
img = C1.copy(); img[~F1] = 0
im = Image.fromarray(img[::-1]).resize((NX * 2, NY * 2), Image.NEAREST); d = ImageDraw.Draw(im)
def to_px(xw, yw): return ((xw - x0) / res * 2, (NY - (yw - y0) / res) * 2)
for (xx, yy), k in zip(cliff, np.isin(np.arange(len(cliff)), np.nonzero(keep)[0])):
    px, py = to_px(xx, yy); d.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(255, 0, 0) if k else (120, 0, 0))
def world_to_lvl(xw, yw):
    xr = (xw - tx) / S; yr = (yw - ty) / S
    return xr * np.cos(-psi) - yr * np.sin(-psi), xr * np.sin(-psi) + yr * np.cos(-psi)
rect = [(CITY_X[0], CITY_Y[0]), (CITY_X[1], CITY_Y[0]), (CITY_X[1], CITY_Y[1]), (CITY_X[0], CITY_Y[1])]
d.polygon([to_px(*world_to_lvl(a, b)) for a, b in rect], outline=(255, 255, 255), width=2)
d.line([to_px(*world_to_lvl(-1300, 600)), to_px(*world_to_lvl(800, 600))], fill=(255, 255, 0), width=2)
d.line([to_px(*world_to_lvl(-773, 492)), to_px(*world_to_lvl(293, 492))], fill=(255, 200, 0), width=2)
im.save(W_ + 'align_preview2.png')
