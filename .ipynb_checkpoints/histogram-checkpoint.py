import glob
from multiprocessing import Process
import os
import re
import numpy as np
import PIL
from PIL import Image


original_path = './Dataset/original_data/Other/AtopicDermatitis/arm'
base_path = '.'


def natural_key(string_):
    """
    Define sort key that is integer-aware
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_norm = img_ybr.convert('RGB')

    return img_norm


def prep_images(paths, out_dir):
    """
    Preprocess images

    Reads images in paths, and writes to out_dir

    """
    for count, path in enumerate(paths):
        if count % 100 == 0:
            print(path)
        img = Image.open(path)
        img_norm = norm_image(img)
        basename = os.path.basename(path)
        path_out = os.path.join(out_dir, basename)
        img_norm.save(path_out)


def main():
    original_all = sorted(glob.glob(os.path.join(original_path, '*.jpg')), key=natural_key)

    # Make the output directories
    base_out = os.path.join(base_path, 'norm_vt_')
    os.makedirs(base_out, exist_ok=True)

    procs = dict()

    procs[1] = Process(target=prep_images, args=(original_all, base_out))
    procs[1].start()

    procs[1].join()


if __name__ == '__main__':
    main()