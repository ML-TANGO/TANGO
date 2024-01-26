"""autonn/ResNet/resnet_core/resnet_utils/img_resize.py
This code not used in the project.
"""
import albumentations as A
import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='image resize')
parser.add_argument("-p", "--data_path", required=True)
parser.add_argument("-l", "--length", default="224")
# parser.add_argument("-s","--save_folder", default='')
args = parser.parse_args()

# path = "D:/resize_test/s_data"

f_list = os.listdir(args.data_path)
save_path = args.data_path + "(resize)"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    for f in f_list:
        os.mkdir(save_path + "/" + f)

transform = A.Compose(
    [
        A.Resize(height=224, width=224),
    ]
)
for folder in f_list:
    img_list = os.listdir(args.data_path + "/" + folder)
    print("processing", folder, "folder...")
    for i, img in enumerate(img_list):
        image = cv2.imread(args.data_path + "/" + folder + "/" + img, cv2.IMREAD_GRAYSCALE)

        transformed = transform(image=image)
        transformed_image = transformed["image"]
        result = transformed_image

        cv2.imwrite(save_path + "/" + folder + "/" + img, result)

        print("image resized & saved:", i+1, end='\r')
        print("done")