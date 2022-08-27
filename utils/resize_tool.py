import argparse
import cv2
import imageio
import os
import tqdm

def resize_blender(dataset_path:str, ratio:float) -> None:
    # just need to resize images, no need to resize camera files
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        for image_name in tqdm.tqdm(os.listdir(split_path)):
            image_path = os.path.join(split_path, image_name)
            image = imageio.imread(image_path)
            image = cv2.resize(image, (int(ratio * image.shape[1]), int(ratio * image.shape[0])), interpolation=cv2.INTER_AREA)
            imageio.imwrite(image_path, image)


def resize_colmap(dataset_path:str, ratio:float) -> None:
    for image_name in tqdm.tqdm(os.listdir(os.path.join(dataset_path, "images"))):
        image_path = os.path.join(dataset_path, "images", image_name)
        image = imageio.imread(image_path)
        image = cv2.resize(image, (int(ratio * image.shape[1]), int(ratio * image.shape[0])), interpolation=cv2.INTER_AREA)
        imageio.imwrite(image_path, image)


# resize in-place
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['blender', 'colmap'], help='dataset type')
    parser.add_argument('--dataset_path', type=str, help='dataset path')
    parser.add_argument('--ratio', type=float, help='ratio')
    args = parser.parse_args()

    if args.dataset_type == 'blender':
        resize_blender(args.dataset_path, args.ratio)
    elif args.dataset_type == 'colmap':
        resize_colmap(args.dataset_path, args.ratio)
    else:
        raise NotImplementedError
