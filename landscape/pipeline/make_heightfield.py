"""Aligned world-space heightfield + colour classes from the GLB.
usage: python3 -I make_heightfield.py <workdir> [cell]
Outputs (in workdir): hf_raw.npy (top surface, GLB-levelled z units), hf_valid.npy, hf_col.npy (uint8 RGB), hf_cls.npy (class id),
hf_grid.json, previews.
"""
import sys, json, numpy as np
from scipy import ndimage
from PIL import Image
W_ = sys.argv[1].rstrip('/') + '/'; CELL = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glbnp import sample_tex
tr = json.load(open(W_ + 'transform.json'))
P = np.load(W_ + 'glb_pos.npy'); UV = np.load(W_ + 'glb_uv.npy'); T = np.load(W_ + 'glb_tri.npy')
tex = np.asarray(Image.open(W_ + 'glb_tex.png').convert('RGB')); vcol = sample_tex(tex, UV)
th = np.radians(tr['theta_x_deg']); ps = np.radians(tr['psi_z_deg']); S = tr['S']; tx, ty = tr['tx'], tr['ty']
Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
Rz = np.array([[np.cos(ps), -np.sin(ps), 0], [np.sin(ps), np.cos(ps), 0], [0, 0, 1]])
Pw = P @ (Rz @ Rx).T
Pw[:, 0] = Pw[:, 0] * S + tx; Pw[:, 1] = Pw[:, 1] * S + ty  # z stays in levelled GLB units
A, B, C3 = Pw[T[:, 0]], Pw[T[:, 1]], Pw[T[:, 2]]
N = np.cross(B - A, C3 - A); nz = N[:, 2] / (np.linalg.norm(N, axis=1) + 1e-12)
# grid
PAD = 800.0
X0 = np.floor((Pw[:, 0].min() - PAD) / CELL) * CELL; Y0 = np.floor((Pw[:, 1].min() - PAD) / CELL) * CELL
NX = int((Pw[:, 0].max() + PAD - X0) / CELL) + 2; NY = int((Pw[:, 1].max() + PAD - Y0) / CELL) + 2
print('grid', NX, 'x', NY, 'cell', CELL, 'origin', X0, Y0)
ix = ((Pw[:, 0] - X0) / CELL).astype(int); iy = ((Pw[:, 1] - Y0) / CELL).astype(int)
# top surface from up-facing triangles (nz > 0.25 in levelled frame): rasterise triangle vertices AND triangle centroids
up = nz > 0.25
cen = (A + B + C3) / 3; cuv_col = (vcol[T[:, 0]].astype(int) + vcol[T[:, 1]] + vcol[T[:, 2]]) // 3
pts = np.concatenate([Pw[np.unique(T[up].ravel())], cen[up]]); cols = np.concatenate([vcol[np.unique(T[up].ravel())], cuv_col[up].astype(np.uint8)])
pix = ((pts[:, 0] - X0) / CELL).astype(int); piy = ((pts[:, 1] - Y0) / CELL).astype(int)
H = np.full((NY, NX), np.nan); np.fmax.at(H, (piy, pix), pts[:, 2])
order = np.argsort(pts[:, 2]); Cm = np.zeros((NY, NX, 3), np.uint8); Cm[piy[order], pix[order]] = cols[order]
valid = ~np.isnan(H)
# steep-side-wall cells: cells where only side faces exist (the relief's extruded walls) -> invalid
anyv = np.zeros((NY, NX), bool); anyv[iy, ix] = True
print('cells with any vertex', anyv.sum(), 'with up-facing surface', valid.sum())
# fill small holes (<= 2 cells from a valid cell) by nearest neighbour
dist, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
Hf = H[tuple(idx)]; Cf = Cm[tuple(idx)]
valid2 = dist <= 2.0
# the relief's side walls: top-down they appear as long streaks; detect via the mesh-footprint cells that have vertices but
# no up-facing surface, dilate and remove
wall = anyv & ~valid
wall = ndimage.binary_opening(wall, iterations=2)
valid2 &= ~ndimage.binary_dilation(wall, iterations=3)
# keep the largest valid component
lab, n = ndimage.label(valid2); sizes = ndimage.sum(valid2, lab, range(1, n + 1)); valid2 = lab == (1 + int(np.argmax(sizes)))
valid2 = ndimage.binary_fill_holes(valid2)
Hf[~valid2] = np.nan
print('valid cells', valid2.sum(), '=', round(valid2.sum() * CELL * CELL / 1e6, 2), 'M studs^2')

# colour classes (per cell, from top colour) : 0 unknown, 1 grass, 2 snow, 3 rock, 4 sand, 5 road/dark, 6 water, 7 roof/urban
Cfl = Cf.astype(float) / 255; R, G, B = Cfl[..., 0], Cfl[..., 1], Cfl[..., 2]
V = Cfl.max(2); Sat = (V - Cfl.min(2)) / (V + 1e-6)
hue = np.degrees(np.arctan2(np.sqrt(3) * (G - B), 2 * R - G - B)) % 360
cls = np.zeros((NY, NX), np.uint8)
cls[(hue > 55) & (hue < 165) & (Sat > 0.22) & (V > 0.18)] = 1          # grass/canopy
cls[(V > 0.70) & (Sat < 0.22)] = 2                                        # snow
cls[(hue > 25) & (hue < 60) & (Sat > 0.15) & (Sat < 0.65) & (V > 0.55) & (cls == 0)] = 4  # sand
cls[(hue > 165) & (hue < 235) & (Sat > 0.3) & (V > 0.35)] = 6             # water (also blue roofs)
cls[(V < 0.28)] = 5                                                        # dark: roads / shadows
cls[(cls == 0) & (Sat < 0.35) & (V >= 0.28) & (V <= 0.70)] = 3            # grey/brown mid: rock
cls[(cls == 0)] = 7                                                        # whatever remains: urban/roof/misc
cls[~valid2] = 0
names = {0: 'none', 1: 'grass', 2: 'snow', 3: 'rock', 4: 'sand', 5: 'dark', 6: 'water', 7: 'other'}
print('class counts', {names[k]: int((cls == k).sum()) for k in names})
np.save(W_ + 'hf_raw.npy', Hf); np.save(W_ + 'hf_valid.npy', valid2); np.save(W_ + 'hf_col.npy', Cf); np.save(W_ + 'hf_cls.npy', cls)
json.dump({'cell': CELL, 'x0': float(X0), 'y0': float(Y0), 'nx': NX, 'ny': NY}, open(W_ + 'hf_grid.json', 'w'))

# previews (row 0 = max Y at top)
pal = np.array([[0, 0, 0], [110, 190, 70], [240, 245, 250], [140, 125, 110], [240, 220, 160], [50, 50, 55], [40, 150, 230], [200, 120, 200]], np.uint8)
Image.fromarray(pal[cls][::-1]).save(W_ + 'hf_classes.png')
cc = Cf.copy(); cc[~valid2] = 0; Image.fromarray(cc[::-1]).save(W_ + 'hf_colour.png')
lo, hi = tr['z_sea'] - 0.01, np.nanmax(Hf)
hn = np.nan_to_num((Hf - lo) / (hi - lo), nan=0).clip(0, 1)
Image.fromarray((hn[::-1] * 255).astype(np.uint8)).save(W_ + 'hf_height.png')
