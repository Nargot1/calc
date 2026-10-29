# Smoothie island landscape - v3 (terraces, slab walls, two-tone rock)

Low-poly terrain for the existing city (`Context`), derived from the GLB's own surface (`result_1793266117047.glb`,
repo root) and made to look like the reference painting (`inputs/reference.png`). v3 implements the three wall
passes agreed after v2:

1. **Terrace sharpening** (`pipeline/process_terrain.py`, section 4b): steep hillsides (local slope > 26-35 deg,
   below ~250 studs, outside plateau / front cliff / beach / sea) are snapped to global shelf levels
   `CITY_Z + k * STEP` (STEP = 50 studs, 3rd argument; 45 = the painting's one-tree walls, 0 = v2 behaviour) with a
   short ramp, so walls are near-vertical and shelves flat with crisp tops. Shelves keep 12 % of the original relief
   (gentle tilt). Isolated knolls (< ~40 studs of relief) stay round. Runs after the plateau blend, so the ramp right
   behind the city is terraced as well.
2. **Slab pass** (`pipeline/slabs.py`, called from `build_terrain2.py` after decimation): every terraced wall face
   (slope > 55 deg) and the city cliff is cut by global horizontal planes at `CITY_Z + k*STEP + {0, 18, 36}` into
   base / mid / lip bands and by jittered vertical seam planes on the mid band (34-50 studs, brick-shifted per band).
   Bands are pushed out as rigid slabs (base 0, mid +3.5, lip +4.5 studs, +-0.8 jitter per segment, alternating
   +-0.6): the wall bulges, the grass lip leans out over the top, the ledges between bands are real faces. Under the
   city (Y 484-498) slabs only move inward, so nothing pokes into the beach parts at z -45. T-junctions are resolved
   before the offsets, so the mesh does not crack. Decimation is vertex-group weighted (wall cells collapse first)
   so every wall is one row of long faces before the pass multiplies it.
3. **Two-tone rock** (`build_terrain2.py`, colour section): faces turned to the sun `#A58B72` warm tan, faces in the
   shade `#7B6F7E` purple-grey (soft split around grazing), grey pair above 180 studs, blue-grey pair
   (`#B9C1CB` / `#7A8096`) on the snow side; darker base band (x0.82); green lip (`#6FBF3F`/`#5FA637`) on the top
   ~6 studs of every wall, snow lip on the snow side; shelves of terraced hillsides are always grass. Slab walls and
   ledges are rock whatever their slope. `build_vegetation2.py` additionally drops boulders at the feet of the walls.

Budget: 64,064 terrain tris in 22 chunks (max 5,283 tris), 447 instances of 14 assets. The slab pass costs
~8 tris per wall face (about 24k tris here), far more than the +15k guessed in the v2 handoff; at the 65k target the
pre-slab mesh is ~40k faces, so everything that is not a wall has a little less detail than v2's 50k mesh. Set the
target higher (`build_terrain2.py ... 80000`) or pass `noslabs` as the 6th argument to trade it the other way.

## Rebuild (about 3 min on 2 CPUs, no GLB needed)
```
pip install bpy==5.2.3 scipy pillow        # bpy = Blender 5.2 as a Python 3.13 module (400 MB wheel)
W=work; mkdir -p $W; cp pipeline/data/* $W/
python3 pipeline/process_terrain.py $W 1.0 50                                    # heightfield -> terrain_h.npy, masks, trees (6 s)
python3 pipeline/build_terrain2.py $W 65000 $W/landscape-v3.blend landscape-v2.blend palette   # ~100 s
python3 pipeline/build_vegetation2.py $W $W/landscape-v3.blend 1.3                               # ~30 s
python3 pipeline/export_fbx2.py $W $W/landscape-v3.blend export                                  # FBX + Lua + re-import check
python3 pipeline/render_blend.py $W/landscape-v3.blend RefCam out.png 100 32 Landscape,Vegetation,Rocks,DistantIslands,Reference_City
```
`build_terrain2.py` takes either the original `landscape-authored.blend` (object `Context`) or any `landscape-v*.blend`
(object `Context_City`) as the source of the city and cameras. Full chain from the GLB, if a re-alignment is ever
wanted: `glbnp.py` -> `glb_analyze2.py` -> `align_glb.py` -> `align_glb2.py` -> `make_heightfield.py <W> 4`.

## Files
```
pipeline/                    scripts (python3 + bpy module, or blender -b --python ... --)
pipeline/data/               aligned heightfield inputs (transform.json, hf_grid.json, hf_raw/valid/cls/col.npy, fitcam.json)
landscape-v2.blend           v2 scene (city, cameras, v2 terrain) - the 4th argument of build_terrain2.py
landscape-v3.blend           v3 scene: Terrain chunks, Water, Assets, Vegetation, Rocks, DistantIslands, Reference_City, cameras, Sun
export/                      Terrain.fbx, Water.fbx, DistantIslands.fbx, Vegetation_Instances.fbx, assets/*.fbx, PlaceInstances.lua
renders/                     walls_closeup_v3.jpg (painting / v2 / v3 close-ups), comparison_reference_v2_v3.jpg (full RefCam),
                             v3_refcam.png, v3_wall_close.png / v3_citycliff_close.png (close cameras)
inputs/reference.png         the 1672x941 reference painting
```
Roblox: 1 Blender unit = 1 stud, Blender (x, y, z) -> Roblox (x, z, -y); see the header of `export/PlaceInstances.lua`.
Key alignment facts (unchanged from v2): city plateau at z = -1 over X[-764, 372] Y[-332, 484]; front cliff -1 -> -46
between Y 484 and 492; beach -46 -> -54; water at -54; RefCam loc (194, 984, 294) rot (69.7, 0, 159.4) lens 22.474
(vertical sensor 24 mm, 1672x941).

## Open choices (not done, ask)
- City cliff bulging +-10-15 studs into the beach parts (v2 handoff item 4) - still a dead-straight line in plan.
- Taller mountains (progressive EXAG 1.6-1.8, 2nd argument of process_terrain.py), terrace step 45 vs 50 vs 60,
  2-3x tree density (3rd argument of build_vegetation2.py), `texture` colour mode.
