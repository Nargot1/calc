# Terrain reference (surfaces only)

Terrain-only reference derived from `result_1793266117047.glb` for future map
development. Buildings, trees, stones, the pier and other props were removed;
only the ground surface remains: beach and sea in the north, the flat town
plateau in the middle, the snowy mountains in the west, the forested hills in
the east and the steep low-poly ridges along the southern coast.

| File | What it is |
|------|------------|
| `terrain_reference.glb` | Terrain mesh, 120 k triangles, 60 k vertices, with normals, UVs and an embedded 2048 px color texture. Same coordinate frame, units and position as the source GLB (Y up), so it overlays the original exactly. |
| `terrain_reference_heightmap16.png` | 16-bit height map, 1024 x 961 px, top-down view of the X/Z plane. 0 outside the footprint. |
| `terrain_reference_heightmap.json` | Height-map metadata: pixel size, world origin, height range, axis conventions and the formula to go from pixel to world coordinates. |
| `terrain_reference_mask.png` | Footprint mask (255 = inside the terrain). |
| `terrain_reference_removed.png` | Pixels that were classified as objects or vertical walls and re-interpolated (255 = removed). Useful to see where the ground is reconstructed rather than measured. |
| `terrain_reference_albedo.jpg` | The top-down color texture used by the GLB (object pixels inpainted from the surrounding ground). |
| `terrain_reference_hillshade.png` | Shaded relief of the height map for a quick look. |
| `previews/` | Renders of the result (textured, flat shaded, top view) and of the source model for comparison. |

## Coordinate frame

* Axes are the source model's: `+Y` is up, `X`/`Z` span the ground plane.
* Height map pixel `(row, col)` maps to world `x = origin_x + col * pixel_size`,
  `z = origin_z + row * pixel_size`, `y = y_min + value / 65535 * (y_max - y_min)`
  with the constants from `terrain_reference_heightmap.json`
  (pixel size 0.000899 units, heights from -0.0947 to 0.3341).
* The footprint is not axis aligned (the island is rotated about 30 degrees), so
  the corners of the height map are empty; use the mask.

## How it was made

`tools/make_terrain_reference.py` (NumPy / SciPy / Pillow / trimesh / fast-simplification):

1. Rasterize the mesh top-down into a height map (highest surface per pixel).
2. Detect objects standing on the ground with a multi-scale morphological
   top-hat. A candidate is removed when it is higher than its surroundings and
   has a steep rim (building walls, fences, the pier), a convex dome shape
   (tree canopies) or is a tiny compact bump (stones). Pits left by open-roofed
   buildings are detected the same way on the inverted height map, and
   near-vertical pixels are discarded.
3. Fill the removed pixels by harmonic interpolation from the surrounding ground.
4. Build a grid mesh over the footprint, decimate it with quadric error metrics,
   project the inpainted top-down color as texture and export GLB.

Regenerate with:

```bash
pip install numpy scipy pillow trimesh fast-simplification
python3 tools/make_terrain_reference.py result_1793266117047.glb terrain_reference/
```

Preview renders come from `tools/render_preview/` (Node + Playwright + three.js):

```bash
node tools/render_preview/render.js terrain_reference terrain_reference.glb out/terrain 0
```

## Known limitations

* The ground under dense tree groves and under buildings is interpolated, so
  those areas are smooth rather than faceted like the untouched low-poly terrain.
* The town plateau keeps its low ground terraces and sunken roads (steps of
  about 0.005 units). They are part of the ground surface, not objects, so they
  were left in place.
* Overhangs and vertical cliff faces cannot exist in a height field; they are
  represented as very steep slopes.
