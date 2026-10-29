"""Minimal GLB reader (numpy only): positions, uvs, triangle indices, base-color texture.
Returns Z-up coordinates (glTF Y-up -> Blender Z-up: x, -z, y), i.e. exactly what bpy's importer produces.
"""
import json, struct, io
import numpy as np
from PIL import Image

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load_glb(path):
    data = open(path, 'rb').read()
    magic, version, length = struct.unpack_from('<III', data, 0)
    assert magic == 0x46546C67, 'not a GLB'
    off = 12; js = None; bins = []
    while off < length:
        clen, ctype = struct.unpack_from('<II', data, off); off += 8
        chunk = data[off:off + clen]; off += clen
        if ctype == 0x4E4F534A: js = json.loads(chunk.decode('utf-8'))
        elif ctype == 0x004E4942: bins.append(chunk)
    buf = bins[0]

    def view_bytes(bv):
        v = js['bufferViews'][bv]
        return buf[v.get('byteOffset', 0): v.get('byteOffset', 0) + v['byteLength']], v.get('byteStride')

    def accessor(ai):
        a = js['accessors'][ai]
        dt = np.dtype(CT[a['componentType']]); n = NC[a['type']]
        raw, stride = view_bytes(a['bufferView'])
        bo = a.get('byteOffset', 0)
        if stride and stride != dt.itemsize * n:
            out = np.empty((a['count'], n), dt)
            for i in range(a['count']):
                out[i] = np.frombuffer(raw, dt, n, bo + i * stride)
            return out
        return np.frombuffer(raw, dt, a['count'] * n, bo).reshape(a['count'], n)

    meshes = []
    for mi, m in enumerate(js['meshes']):
        for p in m['primitives']:
            pos = accessor(p['attributes']['POSITION']).astype(np.float64)
            uv = accessor(p['attributes']['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in p['attributes'] else None
            idx = accessor(p['indices']).astype(np.int64).reshape(-1, 3) if 'indices' in p else np.arange(len(pos)).reshape(-1, 3)
            tex = None
            if 'material' in p:
                mat = js['materials'][p['material']]
                bct = mat.get('pbrMetallicRoughness', {}).get('baseColorTexture')
                if bct is not None:
                    t = js['textures'][bct['index']]
                    src = t.get('source')
                    if src is None:  # EXT_texture_webp etc.
                        src = next(iter(t['extensions'].values()))['source']
                    im = js['images'][src]
                    raw, _ = view_bytes(im['bufferView'])
                    tex = Image.open(io.BytesIO(raw)).convert('RGB')
            # Y-up -> Z-up
            pos_z = np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], 1)
            meshes.append({'name': m.get('name', f'mesh{mi}'), 'pos': pos_z, 'uv': uv, 'tri': idx, 'tex': tex})
    return js, meshes


def sample_tex(tex_arr, uv):
    """nearest-neighbour texture sample; glTF uv origin top-left, v down."""
    h, w = tex_arr.shape[:2]
    u = np.mod(uv[:, 0], 1.0); v = np.mod(uv[:, 1], 1.0)
    xi = np.clip((u * w).astype(int), 0, w - 1); yi = np.clip((v * h).astype(int), 0, h - 1)
    return tex_arr[yi, xi]
