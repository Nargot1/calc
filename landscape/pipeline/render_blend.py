"""Render a .blend with Cycles (CPU, headless).
usage: python render_blend.py <blend> <camera> <out.png> <scale%> <samples> <col1,col2,...|ALL|VISIBLE>
"""
import bpy, sys, time
args = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
blend, cam, out, scale, samples, cols = args
scale = int(scale); samples = int(samples)
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
sc.camera = bpy.data.objects[cam]
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = samples
sc.cycles.use_denoising = False
sc.cycles.max_bounces = 2
sc.cycles.diffuse_bounces = 1
sc.cycles.glossy_bounces = 0
sc.cycles.transmission_bounces = 0
sc.render.resolution_percentage = scale
sc.render.filepath = out
sc.render.image_settings.file_format = 'PNG'
sc.view_settings.view_transform = 'Standard'
# collections visibility
if cols != 'VISIBLE':
    want = None if cols == 'ALL' else set(cols.split(','))
    def names_below(c):
        out = {c.name}
        for ch in c.children: out |= names_below(ch)
        return out
    def walk(lc, parent_wanted=False):
        for ch in lc.children:
            if want is None: keep = True
            else: keep = parent_wanted or bool(names_below(ch.collection) & want)
            ch.exclude = not keep; ch.collection.hide_render = not keep; ch.collection.hide_viewport = False
            walk(ch, parent_wanted or (want is not None and ch.collection.name in want))
    walk(bpy.context.view_layer.layer_collection)
    # objects directly in scene collection (e.g., Context)
    for ob in sc.collection.objects:
        if ob.type == 'MESH':
            ob.hide_render = not (want is None or ob.name in want)
# make sure there's a light + world
if not any(o.type == 'LIGHT' for o in sc.objects):
    bpy.ops.object.light_add(type='SUN');
w = sc.world or bpy.data.worlds.new('W'); sc.world = w
w.use_nodes = True
bg = w.node_tree.nodes.get('Background')
if bg: bg.inputs[0].default_value = (0.55, 0.7, 0.9, 1); bg.inputs[1].default_value = 1.0
for o in sc.objects:
    if o.type == 'LIGHT' and o.data.type == 'SUN':
        o.data.energy = max(o.data.energy, 3.0)
t = time.time()
bpy.ops.render.render(write_still=True)
print("rendered", out, "in", round(time.time()-t, 1), "s")
