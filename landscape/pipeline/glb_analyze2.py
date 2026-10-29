"""Top-down analysis of the raw GLB (Z-up coords): max-z heightmap, top colour map, plateau detection helpers.
usage: python3 -I glb_analyze2.py <workdir> <res>
"""
import sys, numpy as np
from PIL import Image, ImageDraw
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glbnp import sample_tex
W_ = sys.argv[1].rstrip('/') + '/'; RES = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0025
P = np.load(W_ + 'glb_pos.npy'); UV = np.load(W_ + 'glb_uv.npy'); T = np.load(W_ + 'glb_tri.npy')
tex = np.asarray(Image.open(W_ + 'glb_tex.png').convert('RGB'))
vcol = sample_tex(tex, UV)  # per-vertex colour (uint8)

# upward-facing triangles only (top surface)
A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
N = np.cross(B - A, C - A); nz = N[:, 2] / (np.linalg.norm(N, axis=1) + 1e-12)
up_tri = nz > 0.05
up_v = np.zeros(len(P), bool); up_v[T[up_tri].ravel()] = True
print('tris', len(T), 'up tris', up_tri.sum(), 'up verts', up_v.sum())

x0, y0 = P[:, 0].min(), P[:, 1].min()
NX = int((P[:, 0].max() - x0) / RES) + 2; NY = int((P[:, 1].max() - y0) / RES) + 2
ix = ((P[:, 0] - x0) / RES).astype(int); iy = ((P[:, 1] - y0) / RES).astype(int)
H = np.full((NY, NX), np.nan)
sel = up_v
np.fmax.at(H, (iy[sel], ix[sel]), P[sel, 2])
# colour of the topmost vertex per cell
order = np.argsort(P[sel, 2])
Cm = np.zeros((NY, NX, 3), np.uint8)
Cm[iy[sel][order], ix[sel][order]] = vcol[sel][order]
# fill small holes in H by nearest neighbour (cells with no vertex)
from scipy import ndimage
mask = np.isnan(H)
print('grid', NX, 'x', NY, 'empty cells', mask.sum(), 'of', H.size)
idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
Hf = H[tuple(idx)]; Cf = Cm[tuple(idx)]
np.save(W_ + 'glb_H.npy', Hf); np.save(W_ + 'glb_C.npy', Cf); np.save(W_ + 'glb_grid.npy', np.array([x0, y0, RES, NX, NY]))

# previews (image row 0 = max Y = sea side at top, like the earlier colormap)
def save(img, name, scale=2):
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(W_ + name)
save(Cf[::-1], 'glb2_colormap.png')
lo, hi = np.nanpercentile(Hf, 1), np.nanpercentile(Hf, 99.5)
hn = ((Hf - lo) / (hi - lo)).clip(0, 1)
save((hn[::-1] * 255).astype(np.uint8), 'glb2_heightmap.png')
# plateau band preview: z in [-0.01, 0.06] green, beach band [-0.055,-0.015] yellow, sea <-0.06 blue, higher red-ish
band = np.zeros((NY, NX, 3), np.uint8)
band[Hf < -0.06] = (40, 90, 200)
band[(Hf >= -0.06) & (Hf < -0.01)] = (230, 210, 120)
band[(Hf >= -0.01) & (Hf < 0.06)] = (60, 200, 60)
band[(Hf >= 0.06) & (Hf < 0.13)] = (200, 120, 60)
band[Hf >= 0.13] = (240, 240, 240)
save(band[::-1], 'glb2_bands.png')
print('H percentiles', np.nanpercentile(Hf, [1, 5, 25, 50, 75, 95, 99]).round(3))
