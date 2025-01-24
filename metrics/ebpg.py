import numpy as np

def energy_based_pointing_game(saliency_map, bbox):
    """
    Calculate the EBPG score for a single or multiple bounding boxes.
    Parameters:
        - saliency_map: 2D-array in range [0, 1]
        - bbox: Single bounding box (torch.Size([7])) or multiple (torch.Size([N, 7]))
    Returns: EBPG scores for the saliency map.
    """
    # Check if bbox is single or multiple
    if len(bbox.shape) == 1:  # Single bounding box (torch.Size([7]))
        bbox = bbox.unsqueeze(0)  # Convert to torch.Size([1, 7])

    scores = []
    for box in bbox:  # Iterate over all bounding boxes -> consider all detected objects in the image
        x_min, y_min, x_max, y_max = box[:4]
        x_min, y_min, x_max, y_max = map(lambda x: max(int(x), 0), [x_min, y_min, x_max, y_max])

        # Create bounding box mask
        mask = np.zeros_like(saliency_map)
        mask[y_min:y_max, x_min:x_max] = 1 # y=rows, x=columns

        # Normalize saliency map if needed
        if saliency_map.max() > 1.0:
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        # Calculate energy
        energy_bbox = np.sum(saliency_map * mask)
        energy_whole = np.sum(saliency_map)

        # Calculate EBPG score
        score = energy_bbox / energy_whole if energy_whole > 0 else 0
        scores.append(score)

    return scores if len(scores) > 1 else scores[0]
