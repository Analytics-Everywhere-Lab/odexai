import copy
import cv2
import math
import numpy as np
from math import floor
from tqdm import tqdm
import torchvision
from PIL import Image
from torch.nn import functional as F
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
    def __init__(self, model, target_layers, img_size=(640, 640), **kwargs):
        """
        Parameters:
          - model: The model in nn.Module() to analyze
          - target_layers: List of names of the target layers in model.named_modules()
          - img_size: The size of image in tuple
        Variables:
          - self.gradients, self.activations: Dictionary to save the value when
            do forward/backward in format {'name_layer': activation_map/gradient}
          - self.handlers: List of hook functions
        """
        self.model = model.eval()
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
                self.activations[key] = output.detach()

            return forward_hook

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def __call__(self, img, box, obj_idx=None):
        return self.forward(img, box, obj_idx)

    def forward(self, img, box, obj_idx=None):
        """
        Parameters:
          - img: Input image in Tensor[1, 3, H, W]
          - box: The bounding box to analyze Tensor([xmin, ymin, xmax, ymax, p_obj, p_cls, cls])
          - obj_index: The index of target bounding box in int
        Returns:
          - score_saliency_map: The saliency map of target object
        """

        eps = 1e-7
        b, c, h, w = img.shape
        self.model.zero_grad()
        # Get the prediction of the model and the index of each predicted bouding box
        pred = self.model(img)
        _, index = postprocess(pred, 80, 0.25, 0.45, True)

        # num_objs = box.shape[0]
        # target_box = box[:4]
        # target_score = box[4] * box[5]
        # pred_all_score = pred[0][index[obj_idx]][5:]

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
                # weighted_map[weighted_map < 0.] = 0.
                if mean_grad > 0:
                    score_saliency_map += weighted_map
                else:
                    score_saliency_map -= weighted_map

        score_saliency_map[score_saliency_map < 0.0] = 0
        score_saliency_map = (score_saliency_map - score_saliency_map.min()) / (
            score_saliency_map.max() - score_saliency_map.min() + eps
        )

        return score_saliency_map
