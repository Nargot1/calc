"""Export the v2 landscape for Roblox.
Run: python3 export_fbx2.py <workdir> <landscape.blend> <export_dir>
"""
import bpy, sys, os, json
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
W_ = ARGS[0].rstrip('/') + '/'; BLEND = ARGS[1]; OUT = ARGS[2].rstrip('/') + '/'
os.makedirs(OUT + 'assets', exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=BLEND)
sc = bpy.context.scene
def all_objs(col):
    out = list(col.objects)
    for ch in col.children: out += all_objs(ch)
    return out
def export_objs(objs, path):
    bpy.ops.object.select_all(action='DESELECT')
    for lc in bpy.context.view_layer.layer_collection.children: lc.exclude = False
    for ob in objs:
        ob.hide_set(False); ob.hide_viewport = False; ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE',
                             object_types={'MESH'}, use_mesh_modifiers=True, mesh_smooth_type='OFF', colors_type='SRGB',
                             add_leaf_bones=False, bake_anim=False, axis_forward='-Z', axis_up='Y', path_mode='AUTO', embed_textures=False,
                             use_custom_props=True)
    bpy.ops.object.select_all(action='DESELECT')
    tris = sum(sum(len(p.vertices) - 2 for p in ob.data.polygons) for ob in objs if ob.type == 'MESH')
    print(f"exported {path} objects={len(objs)} tris={tris}")
    return tris
report = {}
terrain = [o for o in bpy.data.collections['Terrain'].objects if o.type == 'MESH']
report['Terrain.fbx'] = export_objs(terrain, OUT + 'Terrain.fbx')
water = [o for o in bpy.data.collections['Water'].objects if o.type == 'MESH']
report['Water.fbx'] = export_objs(water, OUT + 'Water.fbx')
isl = [o for o in bpy.data.collections['DistantIslands'].objects if o.type == 'MESH']
report['DistantIslands.fbx'] = export_objs(isl, OUT + 'DistantIslands.fbx')
inst = [o for o in bpy.data.collections['Vegetation'].objects] + [o for o in bpy.data.collections['Rocks'].objects]
report['Vegetation_Instances.fbx'] = export_objs(inst, OUT + 'Vegetation_Instances.fbx')
for ob in bpy.data.collections['Assets'].objects:
    loc = ob.location.copy(); ob.location = (0, 0, 0)
    report['assets/' + ob.name + '.fbx'] = export_objs([ob], OUT + f'assets/{ob.name}.fbx')
    ob.location = loc
report['terrain_chunks'] = {o.name: sum(len(p.vertices) - 2 for p in o.data.polygons) for o in terrain}
json.dump(report, open(OUT + 'export_report.json', 'w'), indent=1)

pl = json.load(open(W_ + 'placements.json'))
lines = ['{"%s",%.1f,%.1f,%.1f,%.0f,%.2f}' % (it['asset'], it['x'], it['y'], it['z'], it['yaw'], it['scale']) for it in pl['instances']]
bb_lines = ['\t["%s"] = {%.2f, %.2f},' % (n, b['min'][2], b['max'][2]) for n, b in pl['assets'].items()]
lua = f'''--[[
PlaceInstances.lua  (run once in Roblox Studio via the command bar, or as a Script in ServerScriptService)

Workflow:
 1. Import each file from assets/*.fbx with the 3D Importer (one MeshPart each). Keep the default scale
    (the meshes are authored at 1 Blender unit = 1 stud, same scale as the existing city).
 2. Put the imported MeshParts in a Folder named "LandscapeAssets" in ReplicatedStorage, named exactly like the files
    (TreeRound_A, TreeConifer_A, TreePalm_A, Rock_A, ...). Anchor them, keep their Size untouched.
 3. Run this script. It clones every instance into workspace.Landscape.Instances.

Coordinates: Blender (x, y, z) -> Roblox (x, z, -y).  Yaw is a rotation about the vertical axis (same sign).
Each entry: {{asset, bx, by, bz, yawDeg, scale}} in Blender coords; bz is the ground height under the instance.
]]
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local assetsFolder = ReplicatedStorage:WaitForChild("LandscapeAssets")

-- vertical bounds of each asset mesh in its own space (minZ, maxZ): base of trunks is 0, rocks sit a bit below ground
local BOUNDS = {{
{chr(10).join(bb_lines)}
}}

local INSTANCES = {{
{",".join(lines)}
}}

local root = workspace:FindFirstChild("Landscape") or Instance.new("Folder")
root.Name = "Landscape"; root.Parent = workspace
local holder = root:FindFirstChild("Instances") or Instance.new("Folder")
holder.Name = "Instances"; holder.Parent = root
holder:ClearAllChildren()

local byAsset = {{}}
for _, e in ipairs(INSTANCES) do
\tlocal name, bx, by, bz, yaw, s = e[1], e[2], e[3], e[4], e[5], e[6]
\tlocal template = assetsFolder:FindFirstChild(name)
\tif template then
\t\tlocal folder = byAsset[name]
\t\tif not folder then
\t\t\tfolder = Instance.new("Folder"); folder.Name = name; folder.Parent = holder; byAsset[name] = folder
\t\tend
\t\tlocal part = template:Clone()
\t\tpart.Anchored = true; part.CanCollide = true
\t\tpart.Size = template.Size * s
\t\tlocal b = BOUNDS[name] or {{0, 0}}
\t\tlocal centreUp = (b[1] + b[2]) / 2 * s          -- mesh pivot is its bbox centre
\t\tlocal pos = Vector3.new(bx, bz + centreUp, -by)
\t\tpart.CFrame = CFrame.new(pos) * CFrame.Angles(0, math.rad(yaw), 0)
\t\tpart.Parent = folder
\telse
\t\twarn("missing asset template: " .. name)
\tend
end
print("placed", #INSTANCES, "landscape instances")
'''
open(OUT + 'PlaceInstances.lua', 'w').write(lua)
print("wrote PlaceInstances.lua with", len(lines), "instances")

# verify: re-import Terrain.fbx into a clean scene
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.fbx(filepath=OUT + 'Terrain.fbx')
import numpy as np
n_obj = 0; tris = 0; has_col = True; maxt = 0; allco = []
for o in bpy.data.objects:
    if o.type != 'MESH': continue
    n_obj += 1
    t = sum(len(p.vertices) - 2 for p in o.data.polygons); tris += t; maxt = max(maxt, t)
    if not o.data.color_attributes: has_col = False
    co = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    M = np.array(o.matrix_world); allco.append(co @ M[:3, :3].T + M[:3, 3])
allco = np.concatenate(allco)
print(f"verify re-import Terrain.fbx: objects={n_obj} tris={tris} max_tris_per_object={maxt} vertex_colors_present={has_col}")
print("re-imported world bbox (Blender Z-up):", allco.min(0).round(1), allco.max(0).round(1))
# plateau check: vertices inside the city rectangle should be at -1
m = (allco[:, 0] > -760) & (allco[:, 0] < 368) & (allco[:, 1] > -328) & (allco[:, 1] < 472)   # 12 studs short of the front: the v3 cliff slabs recess under the edge
print("plateau verts:", m.sum(), "z range", allco[m, 2].min().round(2), allco[m, 2].max().round(2))
mf = (allco[:, 0] > -760) & (allco[:, 0] < 368) & (allco[:, 1] >= 472) & (allco[:, 1] < 486) & (allco[:, 2] > -0.5)
print("front edge verts above -0.5:", mf.sum(), "(must be 0: nothing pokes above the plateau)")
