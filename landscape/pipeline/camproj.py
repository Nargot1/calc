import numpy as np, math
# RefCam from the blend file
C = np.array([194.0, 984.0, 294.0])
rx, ry, rz = math.radians(69.7), 0.0, math.radians(159.4)
def Rx(a): return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
def Ry(a): return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
def Rz(a): return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
R = Rz(rz) @ Ry(ry) @ Rx(rx)
LENS = 22.474; SH = 24.0; W, H = 1672, 941
SW = SH * W / H  # VERTICAL sensor fit

def pix_to_ground(px, py, z):
    """ground point at height z seen at pixel (px,py)."""
    dc = np.array([(px/W - 0.5) * SW/LENS, -(py/H - 0.5) * SH/LENS, -1.0])
    d = R @ dc
    if d[2] >= -1e-6: return None
    t = (z - C[2]) / d[2]
    p = C + t*d
    return p

def world_to_pix(p):
    v = R.T @ (np.asarray(p, float) - C)
    if v[2] >= 0: return None
    x = v[0] / -v[2]; y = v[1] / -v[2]
    px = (x * LENS/SW + 0.5) * W
    py = (0.5 - y * LENS/SH) * H
    return px, py

if __name__ == '__main__':
    print("view dir", R @ np.array([0,0,-1.0]))
    for name, p in [("pier end", (-110, 933, -45)), ("city front-left corner", (372, 484, 0)), ("city front-right corner", (-764, 484, 0)), ("city back-left", (372, -332, 0)), ("city back-right", (-764, -332, 0)), ("beach left (boat)", (160, 640, -45))]:
        print(f"{name:28s} -> px {world_to_pix(p)}")
    for (px, py, z) in [(836, 470, 0), (0, 470, 0), (1672, 470, 0), (836, 940, -54), (836, 300, 100), (836, 150, 400)]:
        print("pix", (px, py), "z", z, "->", pix_to_ground(px, py, z))
