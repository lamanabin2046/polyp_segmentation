import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for segmentation models (UNet++, EfficientNet encoder)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register forward hook
        self.fwd_hook = target_layer.register_forward_hook(self.forward_hook_fn)

        # Register backward hook
        self.bwd_hook = target_layer.register_full_backward_hook(self.backward_hook_fn)

    def forward_hook_fn(self, module, input, output):
        self.activations = output.detach()

    def backward_hook_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        """
        Generates Grad-CAM heatmap for input tensor x
        x: Tensor of shape [1, 3, H, W]
        """
        # Forward pass
        preds = self.model(x)
        score = preds[:, :, :, :].sum()  # segmentation = sum over all pixels

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Mean of gradients across channels
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU to keep only positive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam
