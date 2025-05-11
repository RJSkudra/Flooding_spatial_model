"""
Georeferencing utilities for DEM and LAS files.
"""

import logging
import rasterio
from typing import Dict, Any

logger = logging.getLogger(__name__)

def ensure_georeferencing(dem_file: str, las_metadata: Dict[str, Any] = None) -> None:
    """
    Ensure DEM file has correct georeferencing (LKS-97). Optionally uses LAS metadata for bounds.

    Args:
        dem_file (str): Path to the DEM file.
        las_metadata (Dict[str, Any], optional): Metadata from LAS file containing bounds.
    """
    try:
        with rasterio.open(dem_file) as src:
            current_crs = src.crs
            logger.info(f"DEM faila pašreizējā CRS: {current_crs}")
            if current_crs is None or current_crs.to_string() != 'EPSG:3059':
                data = src.read(1)
                transform = src.transform
                if transform.is_identity:
                    if las_metadata and las_metadata.get('bounds'):
                        bounds = las_metadata['bounds']
                        transform = rasterio.transform.from_bounds(
                            bounds['minx'], bounds['miny'],
                            bounds['maxx'], bounds['maxy'],
                            src.width, src.height
                        )
                        logger.info(f"Izveidota jauna transformācija no LAS robežām: {transform}")
                profile = src.profile
                profile.update(crs='EPSG:3059', transform=transform)
                with rasterio.open(dem_file, 'w', **profile) as dst:
                    dst.write(data, 1)
                logger.info(f"Atjaunināta ģeoreference failam {dem_file} uz LKS-97 (EPSG:3059)")
            else:
                if src.transform.is_identity or src.transform.almost_equals(~src.transform):
                    logger.warning(f"Brīdinājums: DEM failam {dem_file} ir pareiza CRS, bet iespējams nepilnīga transformācija")
                    if las_metadata and las_metadata.get('bounds'):
                        bounds = las_metadata['bounds']
                        new_transform = rasterio.transform.from_bounds(
                            bounds['minx'], bounds['miny'],
                            bounds['maxx'], bounds['maxy'],
                            src.width, src.height
                        )
                        logger.info(f"Piemērojam precīzāku transformāciju no LAS robežām: {new_transform}")
                        profile = src.profile
                        profile.update(transform=new_transform)
                        with rasterio.open(dem_file, 'w', **profile) as dst:
                            dst.write(data, 1)
                        logger.info(f"Atjaunināta transformācija failam {dem_file}, saglabājot LKS-97 CRS")
                else:
                    logger.info(f"DEM fails {dem_file} jau ir korekti ģeoreferencēts uz LKS-97")
    except Exception as e:
        logger.error(f"Kļūda pārbaudot/atjauninot ģeoreferences informāciju: {e}")
