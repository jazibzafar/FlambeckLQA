import numpy as np
import cv2


def mask_to_polygons(mask):
    """Convert a binary mask to polygon segmentation."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    for contour in contours:
        # Polygon needs at least 3 points
        if contour.shape[0] < 3:
            continue
        polygon = contour.reshape(-1, 2).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def pixel_to_geo(polygon, transform):
    """
    Convert COCO-style polygon coordinates from pixel to map coordinates.
    """
    polygon = np.asarray(polygon).reshape(-1, 2)
    geo_coords = [list(transform * (float(x), float(y))) for x, y in polygon]
    return geo_coords
