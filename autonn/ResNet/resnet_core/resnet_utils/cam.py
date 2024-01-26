"""autonn/ResNet/resnet_core/resnet_utils/cam.py
This code not used in the project.
"""
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet

from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from models import densenet_1ch


def resnet200(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> resnet.ResNet:
    return resnet._resnet(
        "resnet200", resnet.Bottleneck, [3, 24, 36, 3], pretrained, progress, **kwargs
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '-i',
        '--image_path',
        type=str,
        default='./examples/',
        help='Input image folder path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument(
        '-p',
        '--pt_path',
        type=str,
        default='./cxr_densenet_n.pt',
        help='Input pt path')

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='densenet121',
        help='Select model')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def make_dir():
    for i in [f'./cam_result/']:
        if not os.path.exists(i):
            os.makedirs(i)
    
    n = 1
    while os.path.exists(f'./cam_result/exp{n}'):
        n += 1
    os.makedirs(f'./cam_result/exp{n}')

    return n

def make_cam_img(args, target_img, path_num):
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}
    
    models = {
        "resnet50": resnet.resnet50(num_classes=2),
        "resnet101": resnet.resnet101(num_classes=2),
        "resnet152": resnet.resnet152(num_classes=2),
        "resnet200": resnet200(num_classes=2),
        "densenet121": densenet_1ch.densenet121(num_classes=2),
        "densenet201": densenet_1ch.densenet201(num_classes=2),
        "densenet121_2048": densenet_1ch.densenet121_2048(num_classes=2),


    }

    model = models[args.model]

    if args.model[0:6] == "resnet":
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    # model = models.resnet50(pretrained=True)
    # model = torch.load("../Test_bench/cxr_densenet.pt")
    pt = torch.load(args.pt_path)
    model.load_state_dict(pt['model_state_dict'])


    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    # print(model)
    

    if args.model[0:6] == "resnet":
        target_layers = [model.layer4]
    elif args.model[0:8] == "densenet":
        target_layers = [model.features[-1]]



    # img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    img = cv2.imread(target_img, 0)
    img = np.float32(img) / 255
    preprocessing = Compose([
                    ToTensor(),
                    Normalize(mean=[0.5,], std=[0.5,])
    ])
    input_tensor = preprocessing(img.copy()).unsqueeze(0)
    # print(input_tensor)

    # input_tensor = preprocess_image(rgb_img,
    #                                 mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        _img = cv2.imread(target_img, 1)
        _img = np.float32(_img) / 255
        # _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)

        cam_image = show_cam_on_image(_img, grayscale_cam, use_rgb=False)
    # print(cam_image)
        

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)
    # cam_gb = cam_mask * gb
    # gb = gb
    _temp = target_img.split("/")[-1].split(".")[0]

    cv2.imwrite(f'./cam_result/exp{path_num}/{_temp}_{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'./cam_result/exp{path_num}/{_temp}_{args.method}_gb.jpg', gb)
    cv2.imwrite(f'./cam_result/exp{path_num}/{_temp}_{args.method}_cam_gb.jpg', cam_gb)

    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    
    'pt_path' example : "./cxr_densenet_n.pt"
    
    """



    args = get_args()

    img_list = os.listdir(args.image_path)

    n = make_dir()

    count = 0
    for i in img_list:
        count += 1
        print("\r"+"processing : ", f"{count} / {len(img_list)}", end="")
        make_cam_img(args, target_img=args.image_path + "/" + i, path_num=n)
    
    print(f"\n CAM image saved 'exp{n}'")

