"""Render a .blend from a custom camera.
usage: python render_view.py <blend> <out.png> <camx,camy,camz> <tx,ty,tz> <lens|ortho:N> <w> <h> <samples> <collections: ALL|VISIBLE|a,b,c> [obj1,obj2 extra visible objects]
"""
import bpy, sys, time, math
from mathutils import Vector
args = sys.argv[1:]
blend, out, cam, tgt, lens, w, h, samples, cols = args[:9]
extra = args[9].split(',') if len(args) > 9 else []
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
cam = Vector([float(v) for v in cam.split(',')]); tgt = Vector([float(v) for v in tgt.split(',')])
cd = bpy.data.cameras.new('AnalysisCam'); co = bpy.data.objects.new('AnalysisCam', cd)
sc.collection.objects.link(co)
co.location = cam
d = (tgt - cam).normalized()
co.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
if lens.startswith('ortho:'):
    cd.type = 'ORTHO'; cd.ortho_scale = float(lens.split(':')[1])
else:
    cd.lens = float(lens)
cd.clip_end = 100000; cd.clip_start = 1
sc.camera = co
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = int(samples)
sc.cycles.use_denoising = False; sc.cycles.max_bounces = 2; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 0
sc.render.resolution_x = int(w); sc.render.resolution_y = int(h); sc.render.resolution_percentage = 100
sc.render.filepath = out; sc.render.image_settings.file_format = 'PNG'
sc.view_settings.view_transform = 'Standard'
if cols != 'VISIBLE':
    want = None if cols == 'ALL' else set(cols.split(','))
    def names_below(c):
        out = {c.name}
        for ch in c.children: out |= names_below(ch)
        return out
    def walk(lc, parent_wanted=False):
        for ch in lc.children:
            keep = True if want is None else (parent_wanted or bool(names_below(ch.collection) & want))
            ch.exclude = not keep; ch.collection.hide_render = not keep
            walk(ch, parent_wanted or (want is not None and ch.collection.name in want))
    walk(bpy.context.view_layer.layer_collection)
    for ob in sc.collection.objects:
        if ob.type == 'MESH':
            ob.hide_render = not (want is None or ob.name in want or ob.name in extra)
for ob in bpy.data.objects:
    if ob.name in extra:
        ob.hide_render = False
        for c in ob.users_collection:
            c.hide_render = False
            # ensure layer collection not excluded
            def find(lc):
                if lc.collection == c: lc.exclude = False; return True
                return any(find(x) for x in lc.children)
            find(bpy.context.view_layer.layer_collection)
if not any(o.type == 'LIGHT' for o in sc.objects):
    ld = bpy.data.lights.new('Sun', 'SUN'); lo = bpy.data.objects.new('Sun', ld); sc.collection.objects.link(lo)
    lo.rotation_euler = (math.radians(50), 0, math.radians(30))
for o in sc.objects:
    if o.type == 'LIGHT' and o.data.type == 'SUN': o.data.energy = max(o.data.energy, 3.0)
wld = sc.world or bpy.data.worlds.new('W'); sc.world = wld; wld.use_nodes = True
bg = wld.node_tree.nodes.get('Background')
if bg: bg.inputs[0].default_value = (0.55, 0.7, 0.9, 1); bg.inputs[1].default_value = 1.0
t = time.time(); bpy.ops.render.render(write_still=True); print("rendered", out, round(time.time()-t,1), "s")
