"""Import the GLB, apply the alignment transform (level + rotate + scale + piecewise z), add Context + cameras, save.
Run: python3 glb_aligned_blend.py <workdir> <glb> <old.blend> <out.blend> [exag]
"""
import bpy, bmesh, sys, json, math
import numpy as np
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
W_ = ARGS[0].rstrip('/') + '/'; GLB = ARGS[1]; OLD = ARGS[2]; OUT = ARGS[3]; EXAG = float(ARGS[4]) if len(ARGS) > 4 else 1.0
tr = json.load(open(W_ + 'transform.json'))
SEA, CITY_Z = -54.0, -1.0
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
sc = bpy.context.scene
obs = [o for o in bpy.data.objects if o.type == 'MESH']
th = math.radians(tr['theta_x_deg']); ps = math.radians(tr['psi_z_deg']); S = tr['S']; tx, ty = tr['tx'], tr['ty']
z_sea, z_city = tr['z_sea'], tr['z_city']
Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
Rz = np.array([[np.cos(ps), -np.sin(ps), 0], [np.sin(ps), np.cos(ps), 0], [0, 0, 1]])
for o in obs:
    me = o.data
    M = np.array(o.matrix_world)
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    co = co @ M[:3, :3].T + M[:3, 3]
    co = co @ (Rz @ Rx).T
    x = co[:, 0] * S + tx; y = co[:, 1] * S + ty; z = co[:, 2]
    low = SEA + (z - z_sea) * ((CITY_Z - SEA) / (z_city - z_sea)); high = CITY_Z + (z - z_city) * S * EXAG
    zw = np.where(z < z_city, low, high)
    me.vertices.foreach_set('co', np.stack([x, y, zw], 1).ravel()); me.update()
    o.matrix_world = np.eye(4).tolist()
    # remove parent transforms: clear parent keeping world
    o.parent = None
    print(o.name, 'bbox', np.stack([x, y, zw], 1).min(0).round(1), np.stack([x, y, zw], 1).max(0).round(1))
for o in list(bpy.data.objects):
    if o.type == 'EMPTY': bpy.data.objects.remove(o, do_unlink=True)
C_GLB = bpy.data.collections.new('GLB'); sc.collection.children.link(C_GLB)
for o in obs:
    for c in o.users_collection: c.objects.unlink(o)
    C_GLB.objects.link(o)
# make the material roughness 1 / no specular for a flat look
for m in bpy.data.materials:
    if m.use_nodes:
        b = m.node_tree.nodes.get('Principled BSDF')
        if b:
            b.inputs['Roughness'].default_value = 1.0
            if 'Specular IOR Level' in b.inputs: b.inputs['Specular IOR Level'].default_value = 0.0
            if 'Metallic' in b.inputs: b.inputs['Metallic'].default_value = 0.0
C_REF = bpy.data.collections.new('Reference_City'); sc.collection.children.link(C_REF)
C_CAM = bpy.data.collections.new('Cameras'); sc.collection.children.link(C_CAM)
with bpy.data.libraries.load(OLD, link=False) as (src, dst):
    dst.objects = [n for n in src.objects if n in ('Context', 'RefCam', 'TopCam', 'ViewCam')]
mat = bpy.data.materials.new('VCol'); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes['Principled BSDF']; attr = nt.nodes.new('ShaderNodeVertexColor'); attr.layer_name = 'Col'
nt.links.new(attr.outputs['Color'], bsdf.inputs['Base Color']); bsdf.inputs['Roughness'].default_value = 1.0
for ob in dst.objects:
    if ob is None: continue
    if ob.type == 'CAMERA': C_CAM.objects.link(ob)
    else:
        C_REF.objects.link(ob); ob.name = 'Context_City'
        bm = bmesh.new(); bm.from_mesh(ob.data); bmesh.ops.reverse_faces(bm, faces=bm.faces); bm.to_mesh(ob.data); bm.free()
        if not ob.data.materials: ob.data.materials.append(mat)
        else:
            for i in range(len(ob.data.materials)): ob.data.materials[i] = mat
sc.camera = bpy.data.objects['RefCam']; sc.render.resolution_x, sc.render.resolution_y = 1672, 941
ld = bpy.data.lights.new('Sun', 'SUN'); ld.energy = 3.0; ld.angle = math.radians(4)
lo = bpy.data.objects.new('Sun', ld); C_CAM.objects.link(lo); lo.rotation_euler = (math.radians(42), 0, math.radians(135))
w = bpy.data.worlds.new('World'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.36, 0.62, 0.95, 1)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print('saved', OUT)
