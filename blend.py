import argparse
import cv2
import numpy as np
from os import path
from PIL import Image


class Blend:
    def __init__(self):
        pass

    def get_mask(self):
        pass

    def blend(self):
        pass


class CutPaste(Blend):
    def __init__(self, params):
        super().__init__()
        self.kernel = params['kernel']

    def get_mask(self):
        mask = 0.5
        return mask

    def blend(self, src, tgt, mask=None):
        self.mask = self.get_mask() if mask is None else mask
        src, tgt = src.astype(float), tgt.astype(float)
        mask = cv2.blur(mask, tuple(self.kernel))
        mask = mask.astype(np.float32) / 255.0
        res = mask*tgt + (1-mask)*src
        res = res.astype(np.uint8)
        return res


def create_blending_technique(technique_name, params):
    if technique_name == 'CutPaste':
        return CutPaste(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create blending technique')
    parser.add_argument('technique', type=str,
                        help='Name of blending technique')
    parser.add_argument('--kernel', nargs='+', type=int,
                        help='Kernel for the CutPaste blending technique')
    parser.add_argument('--mode', type=float,
                        help='Mode for the Poisson blending technique')

    args = parser.parse_args()
    params = {}
    for param_name in ['kernel', 'mode']:
        param_value = getattr(args, param_name)
        if param_value is not None:
            params[param_name] = param_value

    blending_technique = create_blending_technique(args.technique, params)

    dir = 'D:/PDFs/4th year-2nd sem/CV/project/blending_stiching/test/'
    source_path = path.join(dir, "source.png")
    destination_path = path.join(dir, "target.png")
    mask_path = path.join(dir, "mask.png")
    src_img = np.array(Image.open(source_path))
    dest_img = np.array(Image.open(destination_path))

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask,  cv2.COLOR_BGR2RGB)
    blended_data = blending_technique.blend(src_img, dest_img, mask_path)
