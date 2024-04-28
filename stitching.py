from feature_selection import get_features
from feature_matching import get_match
from RANSAC import ransac,getHomography
from blend import create_blending_technique
import cv2
from os import path
import os

class Stitch:
    def __init__(self, method, match, alpha, ransac_iter, blend_technique, blend_params, ransac_thresh=5.0):
        self.method=method
        self.match=match
        self.alpha=alpha
        self.ransac_iter=ransac_iter
        self.blend_technique=blend_technique
        self.blend_params = blend_params
        self.ransac_thresh=ransac_thresh

    def stitch2(self, img1, img2):
        kps1, ftrs1 = get_features(img1, self.method)
        kps2, ftrs2 = get_features(img2, technique_name=self.method)
        matches = get_match(img1, img2, kps1, kps2, ftrs1, ftrs2, self.method, self.match)
        matches = sorted(matches, key = lambda x:x.distance)
        if (len(matches) > 100):
            matches = matches[:int(len(matches)*self.alpha)]
        # mask = ransac(matches,kps1, kps2,self.ransac_thresh,self.ransac_iter)
        mask = getHomography(kps1, kps2,matches,self.ransac_thresh)
        if mask is not None:
            # (mask,M) = mask # ransac
            (M,mask) = mask ## homography
        else:
            print(f"Need atleast 4 matches, found {len(matches)}")
        # print(f"img1 {img1.shape} M {H.shape} img2 {img2.T.shape}")
        res = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
        return res
    
    @staticmethod
    def get_img(path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def stitch_folder(self, folder_path):
        files = os.listdir(folder_path)
        img1 = self.get_img(os.path.join(folder_path, files[0]))
        for i in range(1, len(files)):
            img2 = self.get_img(os.path.join(folder_path, files[i]))
            res = self.stitch2(img1, img2)
            print(res.shape)
            # cv2.imwrite(f'tt_{i}.jpg', res)
            blendtech = create_blending_technique(self.blend_technique, self.blend_params)
            res_blended = blendtech.blend(res, img2)
            img1 = res_blended

        img_map = cv2.cvtColor(res_blended, cv2.COLOR_BGR2RGB)
        cv2.imwrite('ppt.jpg', img_map)
    
s = Stitch("SIFT", 'FLANN', 0.5, 20, 'CutPaste', {'mode': 'alpha', 'alpha': 0.5, 'solver': 'spsolve'})

# dir=r'C:\Users\priya\Desktop\Sem7_8\Sem8\CV_project\computer_vision_project\test'
# out_dir = r'C:\Users\priya\Desktop\Sem7_8\Sem8\CV_project\computer_vision_project\output'
# source_path = path.join(dir, "source.png")
# destination_path = path.join(dir, "target.png")
# mask_path = path.join(dir, "mask.png")
# src_img = np.array(Image.open(source_path))
# dest_img = np.array(Image.open(destination_path))

s.stitch_folder(r'FISB-Image-Stitching\fisb_dataset\sub\scene_2')




