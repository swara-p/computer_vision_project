import cv2
import numpy as np
import math

class FREAK_create():

    def __descriptor_help(self,image, ksize, pts, p):
        out = []
        for pt in pts:
            if pt[0] + p[0] - ksize < 0 or pt[0] + p[0] + ksize >= image.shape[1] or \
                    pt[1] + p[1] - ksize < 0 or pt[1] + p[1] + ksize >= image.shape[0]:
                return -1
            src = image[pt[1] + p[1] - ksize:pt[1] + p[1] + ksize + 1, pt[0] + p[0] - ksize:pt[0] + p[0] + ksize + 1]
            gauss = cv2.GaussianBlur(src, (2*ksize + 1, 2*ksize + 1), 2, borderType=cv2.BORDER_CONSTANT)
            out.append(gauss[ksize, ksize])

        return out

    def compute_orientation(self,pts, intensities):
        M = 42
        pairs = [(0, 2), (1, 3), (2, 4), (3, 5), (0, 4), (1, 5)]
        sum = np.zeros((2, ))
        for i in range(7):
            rank_pts = pts[6 * i: 6 * i + 6]
            rank_i = intensities[6 * i: 6 * i + 6]
            for j in range(6):
                first_pt = np.array(rank_pts[pairs[j][0]])
                second_pt = np.array(rank_pts[pairs[j][1]])
                vec = first_pt - second_pt
                norm = np.linalg.norm(vec)
                vec = vec / norm
                sum += (rank_i[pairs[j][0]] - rank_i[pairs[j][1]]) * vec

        gradients = sum / M
        return math.atan(gradients[1] / gradients[0])


    def __calculate_intensities(self,image, pts, p):
        intensities = np.array([], dtype='int64')
        radii = [18, 13, 9, 6, 4, 3, 2, 1]
        smoothed_intensity = -1

        for i in range(8):
            if i == 7:
                smoothed_intensity = self.__descriptor_help(image, radii[i], [pts[42]], p)
            else:
                smoothed_intensity = self.__descriptor_help(image, radii[i], pts[6 * i: 6 * i + 6], p)

            if smoothed_intensity == -1:
                return -1

            intensities = np.concatenate((intensities, smoothed_intensity))

        return intensities


    def select_pairs(self,images, keypoints_arr, corr_thresh):
        descriptors = []
        sums = np.array([0 for i in range(903)], dtype='int64')
        new_kps = [[] for i in range(1)]

        # Receptive field centres
        pts = [(33, 0), (17, -30), (-17, -30), (-33, 0), (-17, 30), (17, 30),
            (22, 13), (22, -13), (0, -26), (-22, -13), (-22, 13), (0, 26),
            (18, 0), (9, -17), (-9, -17), (-18, 0), (-9, 17), (9, 17),
            (11, 7), (11, -7), (0, -13), (-11, -7), (-11, 7), (0, 13),
            (8, 0), (4, -8), (-4, -8), (-8, 0), (-4, 8), (4, 8),
            (5, 3), (5, -3), (0, -6), (-5, -3), (-5, 3), (0, 6),
            (4, 0), (2, -4), (-2, -4), (-4, 0), (-2, 4), (2, 4),
            (0, 0)]

        for x in range(len(images)):
            image = images[x]
            keypoints = keypoints_arr[x]
            for kp in keypoints:
                p = [int(kp.pt[0]), int(kp.pt[1])]

                # Calculate intensities from the sampling pattern
                intensities = self.__calculate_intensities(image, pts, p)

                if isinstance(intensities, int) and intensities == -1:
                    continue

                # Calculate orientation and rotate sampling pattern
                orientation = self.compute_orientation(pts, intensities)
                cosine = math.cos(orientation)
                sine = math.sin(orientation)
                rotation_matrix = np.array([[cosine, -sine], [sine, cosine]])
                new_pts = []
                for pt in pts:
                    new_pt = np.matmul(rotation_matrix, np.array(pt))
                    new_pts.append((int(new_pt[0]), int(new_pt[1])))

                # Recalculate intensities
                intensities = self.__calculate_intensities(image, new_pts, p)
                if isinstance(intensities, int) and intensities == -1:
                    continue

                # Compute large descriptors and find column sums
                t = 0
                des = []
                for i in range(43):
                    for j in range(43):
                        if i > j:
                            if intensities[i] - intensities[j] > 0:
                                sums[t] += 1
                                des.append(1)
                            else:
                                des.append(0)
                            t += 1

                new_kps[x].append(kp)

                descriptors.append(des)

        # Form initial mask by ordering columns by high variance (mean = 0.5)
        num_kps = sum([len(keypoints) for keypoints in new_kps])
        sums = np.divide(sums, num_kps)
        sums = np.add(sums, -0.5)
        sums = np.abs(sums)
        mask = np.argsort(sums)

        # Correlation check
        corr_mask = [0 for i in range(903)]
        corr_mask[0] = 1
        descriptors = np.array(descriptors)

        good_pairs = [0]
        for i in range(1, 903):
            for j in range(len(good_pairs)):
                corrcoeff = np.corrcoef(descriptors[:, mask[i]], descriptors[:, mask[good_pairs[j]]])
                if abs(corrcoeff[0][1]) >= corr_thresh:
                    break

                if j == len(good_pairs) - 1:
                    corr_mask[i] = 1
                    good_pairs.append(i)

        out_mask = []
        for i in range(903):
            if corr_mask[i] == 1:
                out_mask.append(mask[i])

        for i in range(903):
            if corr_mask[i] == 0:
                out_mask.append(mask[i])

        return descriptors, np.array(out_mask), new_kps
    
    def compute_descriptor(self,descriptors, mask):
        for i in range(len(descriptors)):
            descriptors[i] = descriptors[i][mask]

        descriptors = np.packbits(descriptors, axis=-1)

        return descriptors[:512]
    
    def detectAndCompute(self,image, kps):
        if kps==None:
            orb_obj = cv2.ORB_create()
            kps = orb_obj.detect(image)
        image_gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) ] 
        large_des, mask, kps = self.select_pairs(image_gray, [kps], 0.2)
        descriptors = self.compute_descriptor(large_des[:len(kps[0]), :], mask)
        features = np.uint8(descriptors)
        return (tuple(kps[0]),features)
        

def get_features(image,technique_name):
    assert technique_name is not None, "You need to define a feature detection method. Values are: 'FREAK', 'ORB','SIFT', 'AKAZE', 'BRISK' "
    
    # detect and extract features from the image
    if technique_name == 'SIFT':
        descriptor = cv2.SIFT_create()
    elif technique_name == 'AKAZE':
        descriptor = cv2.AKAZE_create()
    elif technique_name == 'BRISK':
        descriptor = cv2.BRISK_create()
    elif technique_name == 'ORB':
        descriptor = cv2.ORB_create()
    elif technique_name == 'FREAK':
        descriptor = FREAK_create()
    else:
        raise ValueError("Invalid technique name")
    
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)