import argparse
import cv2
import numpy as np

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
      mask = cv2.blur(mask, self.kernel)
      mask = mask.astype(np.float32) / 255.0
      res = mask*tgt + (1-mask)*src
      res = res.astype(np.uint8)
      return res


class Technique2(Blend):
    def __init__(self, params):
        super().__init__()
        self.param3 = params['param3']
        self.param4 = params['param4']

    def helper_function2(self):
        # Helper function specific to Technique2
        pass

    def blend(self, data):
        # Blending logic specific to Technique2
        pass


def create_blending_technique(technique_name, params):
    if technique_name == 'Technique1':
        return Technique1(params)
    elif technique_name == 'Technique2':
        return Technique2(params)
    else:
        raise ValueError("Invalid technique name")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create blending technique')
    parser.add_argument('technique', type=str,
                        help='Name of blending technique')
    parser.add_argument('--param1', type=float,
                        help='Parameter 1 for the blending technique')
    parser.add_argument('--param2', type=float,
                        help='Parameter 2 for the blending technique')
    parser.add_argument('--param3', type=float,
                        help='Parameter 3 for the blending technique')
    parser.add_argument('--param4', type=float,
                        help='Parameter 4 for the blending technique')

    args = parser.parse_args()

    params = {}
    for param_name in ['param1', 'param2', 'param3', 'param4']:
        param_value = getattr(args, param_name)
        if param_value is not None:
            params[param_name] = param_value

    blending_technique = create_blending_technique(args.technique, params)

    # Now you can use the created blending technique object as before
    blended_data = blending_technique.blend(data)
