import numpy as np

def pointing_game(saliency_map, bbox):
    """
    Calculate the Pointing Game score for a single or multiple bounding boxes.
    Parameters:
        - saliency_map: 2D-array in range [0, 1]
        - bbox: Single bounding box (torch.Size([7])) or multiple (torch.Size([N, 7])).
    Returns: PG scores for the saliency map.
    """
    # Check if bbox is single or multiple
    if len(bbox.shape) == 1:  # Single bounding box (torch.Size([7]))
        bbox = bbox.unsqueeze(0)  # Convert to torch.Size([1, 7])

    scores = []
    for box in bbox:  # Iterate over all bounding boxes
        x_min, y_min, x_max, y_max = box[:4]
        x_min, y_min, x_max, y_max = map(lambda x: max(int(x), 0), [x_min, y_min, x_max, y_max])

        # Find the maximum saliency point
        max_saliency = np.max(saliency_map)
        point_y, point_x = np.where(saliency_map == max_saliency)

        # Check for hits and misses
        hit = 0
        miss = 0
        for px, py in zip(point_x, point_y):
            if x_min <= px < x_max and y_min <= py < y_max:
                hit += 1
            else:
                miss += 1

        # Calculate PG score
        total = hit + miss
        score = hit / total if total > 0 else 0
        scores.append(score)

    return scores if len(scores) > 1 else scores[0]
