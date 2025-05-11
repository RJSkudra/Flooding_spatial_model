"""
LAS utility functions for merging LAS files and extracting metadata.
"""

import os
import logging
import json
import laspy
import numpy as np
import pdal
import warnings
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def merge_las_files(input_files: List[str], output_file: str) -> str:
    """Merge multiple LAS files into a single file using PDAL."""
    if len(input_files) <= 1:
        return input_files[0]
    merge_pipeline = {
        "pipeline": [
            *[{"type": "readers.las", "filename": file} for file in input_files],
            {"type": "filters.merge"},
            {"type": "writers.las", "filename": output_file}
        ]
    }
    try:
        pipeline = pdal.Pipeline(json.dumps(merge_pipeline))
        pipeline.execute()
        return output_file
    except Exception as e:
        logger.error(f"Error merging LAS files: {e}")
        return input_files[0]

def extract_las_metadata(las_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata (bounds, CRS) from a LAS file for use in DEM generation and visualization.

    Args:
        las_file_path (str): Path to the LAS file.

    Returns:
        Optional[Dict[str, Any]]: Metadata including bounds and CRS, or None if extraction fails.
    """
    try:
        las = laspy.read(las_file_path)
        x = las.x
        y = las.y
        z = las.z
        bounds = {
            'minx': float(np.min(x)),
            'maxx': float(np.max(x)),
            'miny': float(np.min(y)),
            'maxy': float(np.max(y)),
            'minz': float(np.min(z)),
            'maxz': float(np.max(z)),
        }
        crs = las.header.parse_crs() if hasattr(las.header, 'parse_crs') else None
        return {
            'bounds': bounds,
            'crs': crs,
            'source_crs': crs.to_string() if crs else None
        }
    except Exception as e:
        warnings.warn(f"Failed to extract LAS metadata: {e}")
        return None
