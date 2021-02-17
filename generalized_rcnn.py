import warnings
from collections import OrderedDict

import torch
import numpy as np
from torch import Tensor, nn
from torch._C import clear_autocast_cache
from torch.jit.annotations import Dict, List, Optional, Tuple
from PIL import Image

def tensor_to_img(tensor) -> Image.Image:
    from PIL import Image
    import numpy as np
    arr_img = (tensor.numpy()*255).astype(np.uint8).transpose(1,2,0)
    return Image.fromarray(arr_img)

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN
            and computes detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs
            to feed into the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections, evaluating):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]])
        # -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if evaluating:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        # -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        imgs = images
        tars = targets
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training or targets is not None:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            "Expected target boxes to be a tensor"
                            "of shape [N, 4], got {:}.".format(boxes.shape)
                        )
                else:
                    raise ValueError(
                        "Expected target boxes to be of type "
                        "Tensor, got {:}.".format(type(boxes))
                    )

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function

        try:
            if targets is not None:
                for target_idx, target in enumerate(targets):
                    boxes = target["boxes"]
                    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                    if degenerate_boxes.any():
                        # print the first degenrate box
                        bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                        
                        # get index of all degenerated item:
                        degen_flag = np.array([torch.sum(degenerate_boxes[i]).numpy() for i in range(len(degenerate_boxes))])
                        degen_idx = np.where(degen_flag >= 1)[0]

                        # remove all degenerated item
                        for key in target.keys():
                            for i in range(len(degen_idx)):
                                idx = degen_idx[i] - i 
                                print("Found 1 dengenerated item. Removing...")
                                targets[target_idx][key] = torch.cat((target[key][:idx], target[key][idx+1:]))
                        # degen_bb: List[float] = boxes[bb_idx].tolist()
                        # raise Exception(
                        #     "All bounding boxes should have positive height and width."
                        #     " Found invaid box {} for target at index {}.".format(
                        #         degen_bb, target_idx
                        #     )
                        # )
        except Exception as ex:
            print(ex)
            print(ex.__traceback__.tb_lineno)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in scripting"
                )
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections, targets is not None)
