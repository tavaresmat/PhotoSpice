import collections
import cv2, numpy as np, matplotlib.pyplot as plt

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import torchvision.models as models
import torch

class KeypointsNet:

    def __init__(self, weights="models/diode keypoints.pth"):

        WEIGHTS_PATH = weights

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
        self.model = models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                            pretrained_backbone=False,
                                                            num_keypoints=2,
                                                            num_classes = 2, # Background is the first class, object is the second class
                                                            rpn_anchor_generator=anchor_generator,
                                                            )

        checkpoint: collections.OrderedDict = torch.load(WEIGHTS_PATH, map_location=self.device)

        '''
        # correcting weights names
        list1 = ["backbone.fpn.inner_blocks.0.weight", 
        "backbone.fpn.inner_blocks.0.bias", "backbone.fpn.inner_blocks.1.weight", 
        "backbone.fpn.inner_blocks.1.bias", "backbone.fpn.inner_blocks.2.weight", 
        "backbone.fpn.inner_blocks.2.bias", "backbone.fpn.inner_blocks.3.weight", 
        "backbone.fpn.inner_blocks.3.bias", "backbone.fpn.layer_blocks.0.weight", "backbone.fpn.layer_blocks.0.bias", "backbone.fpn.layer_blocks.1.weight", 
        "backbone.fpn.layer_blocks.1.bias", "backbone.fpn.layer_blocks.2.weight", "backbone.fpn.layer_blocks.2.bias", "backbone.fpn.layer_blocks.3.weight", 
        "backbone.fpn.layer_blocks.3.bias", "rpn.head.conv.weight", "rpn.head.conv.bias"
        ]
        list2 = [
            "backbone.fpn.inner_blocks.0.0.weight", "backbone.fpn.inner_blocks.0.0.bias", "backbone.fpn.inner_blocks.1.0.weight", 
            "backbone.fpn.inner_blocks.1.0.bias", "backbone.fpn.inner_blocks.2.0.weight", "backbone.fpn.inner_blocks.2.0.bias", "backbone.fpn.inner_blocks.3.0.weight", 
            "backbone.fpn.inner_blocks.3.0.bias", "backbone.fpn.layer_blocks.0.0.weight", "backbone.fpn.layer_blocks.0.0.bias", "backbone.fpn.layer_blocks.1.0.weight", 
            "backbone.fpn.layer_blocks.1.0.bias", "backbone.fpn.layer_blocks.2.0.weight", "backbone.fpn.layer_blocks.2.0.bias", 
            "backbone.fpn.layer_blocks.3.0.weight", "backbone.fpn.layer_blocks.3.0.bias", "rpn.head.conv.0.0.weight", "rpn.head.conv.0.0.bias"
        ]

        for i in range(len(list1)):
            checkpoint[list1[i]] = checkpoint[list2[i]]
            del checkpoint[list2[i]]'''

        #mounting model
        self.model.load_state_dict(checkpoint)  
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image=None, imagepath=None) -> list[np.ndarray]:
        if image is None:
            image:np.ndarray = cv2.imread(imagepath)

        img_tensor = F.to_tensor(image)
        output = self.model([img_tensor])[0]

        debug_image = image.copy()

        # check if output is not empty
        score = output['scores'][0]
        box = output['boxes'][0].detach().numpy().astype(int)
        keypoints = output['keypoints'].squeeze()[0].detach().numpy().astype(int)
        keypoint1, keypoint2 = None, None
        try:
            keypoint1 = keypoints[0, :-1:-1]
            keypoint2 = keypoints[1, :-1:-1]
        finally:
            return [keypoint1 or np.array([]), keypoint2 or np.array([])]

    def predict_and_plot(self, image=None, imagepath=None) -> None:
        if image is None:
            image:np.ndarray = cv2.imread(imagepath)
            
        img_tensor = F.to_tensor(image)
        output = self.model([img_tensor])[0]

        RED = (255,0,0)
        BLUE =  (0,0,255)
        GREEN = (0,150,0)

        try:
            debug_image = cv2.cvtColor( image.copy(), cv2.COLOR_GRAY2RGB )
        except:
            debug_image = image
        for index, score in enumerate(output['scores']):
            if score.item() > 0:
                    box = output['boxes'][index].detach().numpy().astype(int)
                    keypoints = output['keypoints'].squeeze()[index].detach().numpy().astype(int)
                    cv2.rectangle (debug_image, tuple(box[:2]), tuple(box[2:]), GREEN, 2) # bounding box
                    cv2.circle (debug_image, tuple(keypoints[0,:2]), 5, RED, 2) # keypoint 0
                    cv2.circle (debug_image, tuple(keypoints[1,:2]), 5,BLUE, 2) # keypoint 1
                    break

        plt.imshow(debug_image)
        plt.show()