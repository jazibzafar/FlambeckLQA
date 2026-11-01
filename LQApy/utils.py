import os
import json
import shapely
from shapely.ops import unary_union
import geopandas as gpd



class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_json(file_path):
    assert file_path.endswith(".json"), "not a valid json file"
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_preds_json(output_preds, save_path):
    if not save_path.endswith(".json"):
        save_path = os.path.join(save_path, "output.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_preds, f, indent=2)


def export_predictions_geo(pred_list, output_dir, file_format="gpkg"):
    """
        Export predictions to one GeoPackage or GeoJSON per tile.
        Parameters
        ----------
        pred_list : list Output preds
        output_dir : str Directory where files will be saved.
        file_format : str Either 'gpkg' or 'geojson'.

        Returns
        -------
        list: Paths to exported files.
    """
    file_format = file_format.lower().replace(".", "")
    if file_format not in ("gpkg", "geojson"):
        raise ValueError("file_format must be 'gpkg' or 'geojson'")

    exported_files = []
    for tile_data in pred_list:
        tile_name = tile_data["tile"]
        tile_name, tile_ext = tile_name.split(".")
        crs = tile_data["crs"]
        annotations = tile_data["annotations"]
        features = []
        for ann in annotations:
            polygons = []
            for coords in ann["segmentation_geo"]:
                if len(coords) < 3:
                    continue
                polygon = shapely.geometry.Polygon(coords)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if not polygon.is_empty:
                    polygons.append(polygon)
            if not polygons:
                continue

            # combine disconnected components
            geometry = unary_union(polygons)
            features.append({
                "id": int(ann["id"]),
                "label": int(ann["label"]),
                "score": float(ann["score"]),
                # Pixel coordinates stored as JSON strings
                "box_px": json.dumps(ann["box"]),
                "seg_px": json.dumps(ann["segmentation"]),
                # Geographic bounding box
                "bbox_geo": json.dumps(ann.get("bbox_geo")),
                "area_px": float(ann["area"]),
                "geometry": geometry,
                "transform": json.dumps(tile_data["transform"]),
                "tile": tile_data["tile"]
            })
        if not features:
            print(f"Skipping {tile_name}: no valid geometries.")
            continue
        gdf = gpd.GeoDataFrame(
            features,
            geometry="geometry",
            crs=crs,
        )
        output_path = os.path.join(output_dir, f"{tile_name}.{file_format}")
        if file_format == "gpkg":
            gdf.to_file(
                output_path,
                driver="GPKG",
                layer="predictions",
                index=False,
            )
        else:
            gdf.to_file(
                output_path,
                driver="GeoJSON",
                index=False,
            )
        exported_files.append(output_path)
        print(f"Exported {tile_name}: {len(gdf)} detections")
    return exported_files


def load_preds_geo(input_file_path):
    """
        Load GeoPackage/GeoJSON predictions and reconstruct pred_list.

        Parameters
        ----------
        input_path : str A single file

        Returns
        -------
        dict: Reconstructed prediction dictionaries, compatible with the output structure of collect_preds().
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(input_file_path)

    gdf = gpd.read_file(input_file_path)
    annotations = []
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if isinstance(geometry, shapely.geometry.Polygon):
            geometries = [geometry]
        elif isinstance(geometry, shapely.geometry.MultiPolygon):
            geometries = list(geometry.geoms)
        else:
            continue

        segmentation_geo = []
        for polygon in geometries:
            coords = [[float(x), float(y)] for x, y in polygon.exterior.coords]
            segmentation_geo.append(coords)

        box = json.loads(row["box_px"])
        segmentation = json.loads(row["seg_px"])
        bbox_geo = json.loads(row["bbox_geo"])
        annotations.append({
            "id": int(row["id"]),
            "label": int(row["label"]),
            "score": float(row["score"]),
            "box": box,
            "segmentation": segmentation,
            "bbox_geo": bbox_geo,
            "segmentation_geo": segmentation_geo,
            "area": float(row["area_px"]),
        })
    return {
        "tile": input_file_path,
        "crs": gdf.crs.to_string() if gdf.crs else None,
        "num_detections": len(annotations),
        "annotations": annotations,
    }
