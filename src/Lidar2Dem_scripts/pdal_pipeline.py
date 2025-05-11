"""
PDAL pipeline utilities for running point cloud processing pipelines.
"""

import json
import logging
import pdal
from typing import Dict, Any

logger = logging.getLogger(__name__)

def run_pdal_pipeline(pipeline_json: Dict[str, Any], tqdm_available: bool = True) -> None:
    """Run a PDAL pipeline with only error output.

    Args:
        pipeline_json (Dict[str, Any]): The PDAL pipeline configuration as a dictionary.
        tqdm_available (bool): Flag indicating if tqdm is available for progress tracking.

    Raises:
        Exception: If there is an error executing the pipeline or processing metadata.
    """
    pipeline_json_str = json.dumps(pipeline_json)
    pipeline = None
    try:
        pipeline = pdal.Pipeline(pipeline_json_str)
        pipeline.execute()
    except Exception as pipeline_error:
        logger.error(f"PDAL pipeline error: {pipeline_error}")
        raise
    if pipeline is not None:
        try:
            metadata_str = pipeline.metadata
            if isinstance(metadata_str, dict):
                metadata = metadata_str
            else:
                metadata = json.loads(metadata_str)
        except Exception as metadata_error:
            logger.error(f"Metadata error: {metadata_error}")
