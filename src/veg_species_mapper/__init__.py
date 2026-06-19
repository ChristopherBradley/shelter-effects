"""veg_species_mapper -- proof-of-concept pipeline:

  Mapillary 360 imagery (with pose)
    -> perspective crops
    -> Pl@ntNet auto species labels
    -> triangulate ground location
    -> Sentinel-2 signature for training

See scripts/run_poc.py for the end-to-end demo.
"""
__all__ = ["mapillary", "panorama", "species_id", "triangulate", "sentinel"]
