"""Model training and evaluation for the coral oocyte pipeline.

Tile-level oocyte/no-oocyte classification lives here (M7a).

Layering rule: ``tile_labels`` and the tile-index and sampler helpers must not import
``torch``. Tile labels are derived from annotation geometry rather than from the tile
manifest's ``has_oocyte`` flag, which marks only the tile containing an annotation's
centroid and so misses the surrounding tiles of any oocyte wider than one tile. The
detection pipelines need that corrected label too, and keeping these modules
torch-free lets them import it without pulling in the ML stack.
"""
