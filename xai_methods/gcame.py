import copy
import cv2
import math
import numpy as np
from math import floor
from tqdm import tqdm
import torch
from YOLOX.yolox.utils import postprocess


def create_heatmap(output_width, output_height, p_x, p_y, sigma):
    """
    Parameters:
      - output_width, output_height: The kernel size of Gaussian mask
      - p_x, p_y: The center of Gaussian mask
      - sigma: The standard deviation of Gaussian mask
    Returns:
      - mask: The 2D-array Gaussian mask in range [0, 1]
    """
    X1 = np.linspace(1, output_width, output_width)
    Y1 = np.linspace(1, output_height, output_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - floor(p_x)
    Y = Y - floor(p_y)
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma**2
    Exponent = D2 / E2
    mask = np.exp(-Exponent)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


class GCAME(object):
    def __init__(self, model, target_layers, arch="yolox", img_size=(640, 640), **kwargs):
        """
        Parameters:
          - model: The model in nn.Module() to analyze
          - target_layers: List of names of the target layers in model.named_modules()
          - arch: Architecture type ("yolox" or "fasterrcnn")
          - img_size: The size of image in tuple
        """
        self.model = model.eval()
        self.arch = arch
        self.img_size = img_size
        self.gradients = dict()
        self.activations = dict()
        self.target_layers = target_layers
        self.handlers = []

        # Save gradient values and activation maps
        def save_grads(key):
            def backward_hook(module, grad_inp, grad_out):
                g = grad_out[0].detach()
                self.gradients[key] = g
            return backward_hook

        def save_fmaps(key):
            def forward_hook(module, inp, output):
                if self.arch == "fasterrcnn":
                    self.activations[key] = output
                else:
                    self.activations[key] = output.detach()
            return forward_hook

        # Register hooks based on architecture
        if self.arch == "fasterrcnn":
            for name, module in list(self.model.named_modules())[1:]:
                if name in self.target_layers:
                    self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                    self.handlers.append(module.register_backward_hook(save_grads(name)))
        else:  # yolox
            for name, module in self.model.named_modules():
                if name in self.target_layers:
                    self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                    self.handlers.append(module.register_backward_hook(save_grads(name)))

    def remove_hook(self):
        """Remove all the forward/backward hook functions"""
        for handle in self.handlers:
            handle.remove()

    def __call__(self, img, box, obj_idx=None):
        if self.arch == "fasterrcnn":
            return self.forward_fasterrcnn(img, box, obj_idx)
        else:
            return self.forward_yolox(img, box, obj_idx)

    def forward_yolox(self, img, box, obj_idx=None):
        """Forward function for YOLOX architecture"""
        eps = 1e-7
        b, c, h, w = img.shape
        self.model.zero_grad()
        
        # Get the prediction of the model and the index of each predicted bounding box
        pred = self.model(img)
        _, index = postprocess(pred, 80, 0.25, 0.45, True)

        target_cls = box[6].int()
        self.model.zero_grad()
        # Do backward
        pred[0][index[obj_idx]][target_cls + 5].backward(retain_graph=True)

        # Create the saliency map
        score_saliency_map = np.zeros((self.img_size[0], self.img_size[1]))

        for key in self.activations.keys():
            map = self.activations[key]
            grad = self.gradients[key]

            # Select the branch that the target comes out
            if grad.max().item() == 0 and grad.min().item() == 0:
                continue

            map = map.squeeze().cpu().numpy()
            grad = grad.squeeze().cpu().numpy()

            # Calculate the proportion between the input image and the gradient map
            stride = self.img_size[0] / grad.shape[1]
            for j in range(map.shape[0]):
                weighted_map = map[j]
                mean_grad = np.mean(grad[j])

                # Get the center of the Gaussian mask
                id_x, id_y = (grad[j] != 0).nonzero()
                if len(id_x) == 0 or len(id_y) == 0:
                    continue

                id_x = id_x[0]
                id_y = id_y[0]

                # Weight the feature map
                weighted_map = abs(mean_grad) * map[j]
                kn_size = (
                    math.floor(math.sqrt(grad.shape[1] * grad.shape[2]) - 1) / 2 / 3
                )
                sigma = (np.log(abs(mean_grad)) / kn_size) * np.log(stride)
                mask = create_heatmap(
                    grad[j].shape[1], grad[j].shape[0], id_y, id_x, abs(sigma)
                )
                weighted_map *= mask
                weighted_map = cv2.resize(
                    weighted_map, (self.img_size[1], self.img_size[0])
                )
                
                if mean_grad > 0:
                    score_saliency_map += weighted_map
                else:
                    score_saliency_map -= weighted_map

        score_saliency_map[score_saliency_map < 0.0] = 0
        score_saliency_map = (score_saliency_map - score_saliency_map.min()) / (
            score_saliency_map.max() - score_saliency_map.min() + eps
        )

        return score_saliency_map

    def forward_fasterrcnn(self, img, box, index=None):
        """Forward function for FasterRCNN architecture"""
        eps = 1e-7
        c, h, w = img.size()
        org_size = (h, w)

        # Get input image size after transformation
        transform_img = self.model.transform([img])[0]
        self.img_size = transform_img.image_sizes[0]
        self.model.zero_grad()

        # Get prediction
        output = self.model([img])
        output[0]['scores'][index].backward(retain_graph=True)

        # Create saliency map
        score_saliency_map = np.zeros(org_size)

        for target_layer in self.target_layers:
            if target_layer not in self.activations or target_layer not in self.gradients:
                continue
                
            map = self.activations[target_layer]
            grad = self.gradients[target_layer]

            # Select the branch that the target comes out
            if grad.max() == 0 and grad.min() == 0:
                continue

            map = map.squeeze().detach().cpu().numpy()
            grad = grad.squeeze().detach().cpu().numpy()

            # Resize gradient to match activation map spatial dimensions
            if map.shape != grad.shape:
                grad_resized = np.zeros_like(map)
                min_channels = min(map.shape[0], grad.shape[0])
                for ch in range(min_channels):
                    grad_resized[ch] = cv2.resize(grad[ch], (map.shape[2], map.shape[1]))
                grad = grad_resized

            # Calculate the proportion between the input image and the gradient map
            stride = math.sqrt((self.img_size[0] * self.img_size[1]) / (grad.shape[1] * grad.shape[2]))

            for j in range(map.shape[0]):
                new_map = copy.deepcopy(map[j])
                pos_grad = copy.deepcopy(grad[j])
                neg_grad = copy.deepcopy(grad[j])

                # Get the positive part of gradient map
                pos_grad[pos_grad < 0] = 0
                mean_pos_grad = np.mean(pos_grad)
                max_grad = pos_grad.max()
                idx, idy = (pos_grad == max_grad).nonzero()
                if len(idx) == 0 or len(idy) == 0 or mean_pos_grad == 0:
                    continue

                idx = idx[0]
                idy = idy[0]
                kn_size = math.floor((math.sqrt(grad.shape[1] * grad.shape[2]) - 1) / 2) / 3
                pos_sigma = (np.log(abs(mean_pos_grad)) / kn_size) * np.log(stride)
                pos_sigma = max(abs(pos_sigma), 1.)
                pos_mask = create_heatmap(grad[j].shape[1], grad[j].shape[0], idy, idx, pos_sigma)
                pos_weighted_map = (new_map * mean_pos_grad) * pos_mask

                # Get the negative part of gradient map
                neg_grad[neg_grad > 0] = 0
                mean_neg_grad = np.mean(neg_grad)
                if mean_neg_grad == 0:
                    neg_weighted_map = np.zeros_like(pos_weighted_map)
                else:
                    min_grad = np.unique(neg_grad[neg_grad != 0])[-1]
                    idx_, idy_ = (neg_grad == min_grad).nonzero()
                    if len(idx_) == 0 or len(idy_) == 0:
                        neg_weighted_map = np.zeros_like(pos_weighted_map)
                    else:
                        idx_ = idx_[0]
                        idy_ = idy_[0]
                        neg_sigma = (np.log(abs(mean_neg_grad)) / kn_size) * np.log(stride)
                        neg_sigma = max(abs(neg_sigma), 1.)
                        neg_mask = create_heatmap(grad[j].shape[1], grad[j].shape[0], idy_, idx_, neg_sigma)
                        neg_mask = cv2.resize(neg_mask, (new_map.shape[1], new_map.shape[0]))
                        neg_weighted_map = (new_map * mean_neg_grad) * neg_mask
                
                # Sum up the weighted feature map
                weighted_map = pos_weighted_map - neg_weighted_map
                weighted_map = cv2.resize(weighted_map, (org_size[1], org_size[0]))
                weighted_map[weighted_map < 0.] = 0.
                score_saliency_map += weighted_map

        score_saliency_map = (score_saliency_map - score_saliency_map.min()) / (
            score_saliency_map.max() - score_saliency_map.min() + eps
        )
        return score_saliency_map