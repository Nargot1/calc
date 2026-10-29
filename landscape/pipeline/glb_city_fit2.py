"""City rectangle from colour classification (urban vs nature) + road orientation. usage: python3 -I glb_city_fit2.py <workdir>"""
import sys, json, numpy as np
from scipy import ndimage
from PIL import Image
W_ = sys.argv[1].rstrip('/') + '/'
x0, y0, RES, NX, NY = np.load(W_ + 'glb_grid.npy'); NX = int(NX); NY = int(NY)
H = np.load(W_ + 'glb_H.npy'); C = np.load(W_ + 'glb_C.npy').astype(float) / 255
G = np.load(W_ + 'glb_G.npy'); foot = np.load(W_ + 'glb_foot.npy')
R, Gc, B = C[..., 0], C[..., 1], C[..., 2]
V = C.max(2); mn = C.min(2); S = (V - mn) / (V + 1e-6)
hue = np.degrees(np.arctan2(np.sqrt(3) * (Gc - B), 2 * R - Gc - B)) % 360
green = (hue > 60) & (hue < 160) & (S > 0.25) & (V > 0.2)
water = (hue > 170) & (hue < 230) & (S > 0.35) & (V > 0.4)
snow = (V > 0.72) & (S < 0.2)
sand = (hue > 30) & (hue < 60) & (S > 0.15) & (S < 0.6) & (V > 0.6)
dark = V < 0.30
urban = foot & ~green & ~water & ~snow & ~sand & (G < 0.1)
for n, m in [('green', green), ('water', water), ('snow', snow), ('sand', sand), ('dark', dark), ('urban', urban)]:
    print(n, (m & foot).sum())
# clean urban mask: majority filter + largest component
u = ndimage.uniform_filter(urban.astype(float), 7) > 0.5
u = ndimage.binary_closing(u, iterations=3)
lab, n = ndimage.label(u); sizes = ndimage.sum(u, lab, range(1, n + 1)); u = lab == (1 + int(np.argmax(sizes)))
u = ndimage.binary_fill_holes(u)
print('urban main component cells', u.sum(), 'area', u.sum() * RES * RES)

def proj_score(mask, theta):
    rot = ndimage.rotate(mask.astype(float), theta, reshape=True, order=1)
    return rot.sum(0).var() + rot.sum(1).var()
road = dark & foot & (G < 0.1) & ndimage.binary_dilation(u, iterations=10)
cands = [(proj_score(road, t), t) for t in np.arange(0, 180, 1.0)]
cands.sort(reverse=True); print('top road angles', [(round(s, 1), t) for s, t in cands[:6]])
t0 = cands[0][1]
fine = max(((proj_score(road, t), t) for t in np.arange(t0 - 1.5, t0 + 1.5, 0.1)), key=lambda a: a[0])
theta_road = float(fine[1]) % 90
# urban mask: minimum-area rectangle by brute force over angles
ys, xs = np.nonzero(u); X = xs * RES + x0; Y = ys * RES + y0
best = None
for t in np.arange(0, 90, 0.5):
    th = np.radians(t); Uc = X * np.cos(th) + Y * np.sin(th); Vc = -X * np.sin(th) + Y * np.cos(th)
    a = (np.percentile(Uc, 99.5) - np.percentile(Uc, 0.5)) * (np.percentile(Vc, 99.5) - np.percentile(Vc, 0.5))
    if best is None or a < best[0]: best = (a, t)
t_rect = best[1]
th = np.radians(t_rect); Uc = X * np.cos(th) + Y * np.sin(th); Vc = -X * np.sin(th) + Y * np.cos(th)
ub = np.percentile(Uc, [0.5, 99.5]); vb = np.percentile(Vc, [0.5, 99.5])
print('urban min-area rect angle', t_rect, 'U', ub.round(4), 'V', vb.round(4), 'size', round(ub[1] - ub[0], 4), 'x', round(vb[1] - vb[0], 4))
print('road grid angle', theta_road)
# ground height stats inside the urban rect
gi = G[u]; print('ground z inside urban: percentiles', np.percentile(gi, [5, 25, 50, 75, 95]).round(4))
# profile of ground along V (inland axis) inside urban mask
bins = np.linspace(vb[0], vb[1], 12)
for i in range(len(bins) - 1):
    m = (Vc >= bins[i]) & (Vc < bins[i + 1])
    print('  V %.3f..%.3f  ground z median %.4f  (n=%d)' % (bins[i], bins[i + 1], np.median(G[u][m]) if m.sum() else float('nan'), m.sum()))
bins = np.linspace(ub[0], ub[1], 12)
for i in range(len(bins) - 1):
    m = (Uc >= bins[i]) & (Uc < bins[i + 1])
    print('  U %.3f..%.3f  ground z median %.4f  (n=%d)' % (bins[i], bins[i + 1], np.median(G[u][m]) if m.sum() else float('nan'), m.sum()))
json.dump({'rect_angle': float(t_rect), 'U': ub.tolist(), 'V': vb.tolist(), 'road_angle': theta_road}, open(W_ + 'city_fit2.json', 'w'))
np.save(W_ + 'glb_urban.npy', u)
img = (C * 255).astype(np.uint8).copy(); img[~foot] = 0
img[u] = (img[u] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
img[road] = (255, 255, 0)
# draw rectangle corners
from PIL import ImageDraw
im = Image.fromarray(img[::-1]).resize((NX * 2, NY * 2), Image.NEAREST); d = ImageDraw.Draw(im)
corners = [(ub[0], vb[0]), (ub[1], vb[0]), (ub[1], vb[1]), (ub[0], vb[1])]
pts = []
for uu, vv in corners:
    xw = uu * np.cos(th) - vv * np.sin(th); yw = uu * np.sin(th) + vv * np.cos(th)
    px = (xw - x0) / RES * 2; py = (NY - (yw - y0) / RES) * 2; pts.append((px, py))
d.polygon(pts, outline=(255, 255, 255))
im.save(W_ + 'fit2_urban.png')
