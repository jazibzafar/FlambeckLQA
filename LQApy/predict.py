import torch
import tqdm
from pyproj import CRS
from LQApy.geo_utils import mask_to_polygons, pixel_to_geo


def collect_preds_geo(outputs, metadata, score_threshold=0.5, mask_threshold=0.5, filter_tree_groups=True):
    """
        Filter Mask R-CNN predictions and create georeferenced annotations.

        Parameters
        ----------
        meta : dict
            Geospatial metadata containing:
                - tile_names
                - transform (or tranform)
                - crs
            provided as lists containing one entry per tile.
        outputs : list
            Mask R-CNN prediction outputs.
        score_threshold : float
            Minimum confidence score to retain an annotation.
        filter_tree_groups : bool
            If True, filter out tree group predictions (label 1).
        mask_threshold : float
            Threshold for converting predicted mask probabilities to binary masks.
        Returns
        -------
        list
            Dictionary for each tile containing georeferenced annotations.
    """
    assert len(metadata) == len(outputs), "len(meta) /= len(outputs)"
    batch_size = len(outputs)

    pred_list = []
    for i, (current_meta, current_output) in enumerate(zip(metadata, outputs)):
        tile_name = current_meta["tile_name"]
        transform = current_meta["transform"]
        crs = CRS.from_user_input(current_meta["crs"])

        boxes = current_output["boxes"]
        labels = current_output["labels"]
        scores = current_output["scores"]
        masks = current_output["masks"]

        annotations = []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            # Convert tensors to Python/NumPy objects
            label = int(label.detach().cpu().item())
            score = float(score.detach().cpu().item())

            # Filter tree groups
            if filter_tree_groups and label == 1:
                continue

            # Filter low-confidence predictions
            if score < score_threshold:
                continue

            box = box.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()[0]  # Mask shape: [1, H, W]
            mask_binary = (mask >= mask_threshold).astype(np.uint8)
            # Extract polygons in pixel coordinates
            segmentation = mask_to_polygons(mask_binary)
            if len(segmentation) == 0:
                continue

            # Convert segmentation polygons to geographic coordinates
            segmentation_geo = [pixel_to_geo(poly, transform) for poly in segmentation]
            x1, y1, x2, y2 = box
            bbox_geo = [
                list(transform * (x1, y1)),
                list(transform * (x2, y1)),
                list(transform * (x2, y2)),
                list(transform * (x1, y2)),
            ]
            # Pixel area
            area = float(mask_binary.sum())
            annotations.append({
                "id": len(annotations) + 1,
                "label": label,
                "score": score,
                # Original pixel coordinates
                "box": box.tolist(),
                "segmentation": segmentation,
                # Geographic coordinates
                "bbox_geo": bbox_geo,
                "segmentation_geo": segmentation_geo,
                "area": area,
            })
        pred_list.append({
            "tile": tile_name,
            "crs": crs.to_string(),
            "transform": list(transform)[:6],
            "num_detections": len(annotations),
            "annotations": annotations,
        })
    return pred_list


def predict(pred_model, pred_loader, score_threshold=0.5, mask_threshold=0.5, filter_tree_groups=True, device="cuda"):
    """
    Predict the box and mask labels for the desired dataset.
    :param pred_model: prediction model loaded
    :param pred_loader: dataloader for prediction
    :param score_threshold: default=0.5; confidence filter to accept/reject model prediction. based on box score.
    :param mask_threshold: default=0.5; confidence filter for binarizing the mask.
    :param filter_tree_groups: trigger it true to filter out tree group preds and keep only single tree preds
    :param device: "cuda" if available, else "cpu"
    :return: preds: box and mask predictions as a list
    """
    pred_model.to(device)
    pred_model.eval()

    pbar = tqdm(pred_loader, desc="Predicting")
    preds = []
    with torch.no_grad():
        for batch in pbar:
            # get the predictions
            images, metadata = batch
            images = [image.to(device) for image in images]
            outputs = pred_model(images)
            # collect all predictions
            collected_preds = collect_preds_geo(outputs, metadata, score_threshold, mask_threshold, filter_tree_groups)
            preds.extend(collected_preds)
    return preds
