import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class Match:
    def __init__(self,method):
        self.method=method

    def make_matcher(self,check=True):
        if self.method == 'SIFT':
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=check)
        else:
            # self.method == 'ORB' or self.method == 'BRISK' or self.method == 'AKAZE' or self.method=="FREAK":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=check)
        
    @staticmethod
    def check_distance(raw_matches, ratio):
        matches = []
        for m,n in raw_matches:
            if m.distance < n.distance * ratio:
                matches.append(m)
        return matches
    
    @staticmethod
    def show_feature_match(img1, kpsA, img2, kpsB, matches):
        return cv2.drawMatches(img1,kpsA,img2,kpsB,np.random.choice(matches,100), None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    @staticmethod
    def show_kps(img1, kpsA, img2, kpsB):
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
        ax1.imshow(cv2.drawKeypoints(img1,kpsA,None,color=(0,255,0)))
        ax1.set_xlabel("(a)", fontsize=14)
        ax2.imshow(cv2.drawKeypoints(img2,kpsB,None,color=(0,255,0)))
        ax2.set_xlabel("(b)", fontsize=14)
        plt.title('Keypoints')
        plt.savefig('../output/keypoints.png')
        plt.close()

class BruteForce(Match):
    def __init__(self, method):
        super().__init__(method)

    def matchKP(self, featuresA, featuresB):
        bf = self.make_matcher(check=True)
        best_matches = bf.match(featuresA,featuresB)
        rawMatches = sorted(best_matches, key = lambda x:x.distance)
        "Raw matches (bruteforce):", len(rawMatches)
        return rawMatches
    
    @staticmethod
    def show_feature_match(img1, kpsA, img2, kpsB, matches):
        return cv2.drawMatches(img1,kpsA,img2,kpsB,matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


class KNN(Match):
    def __init__(self, method, ratio):
        super().__init__(method)
        self.ratio=ratio

    def matchKP(self, featuresA, featuresB):
        bf = self.make_matcher(check=True)
        rawMatches = bf.knnMatch(featuresA, featuresB, 2,mask=None)
        print("Raw matches (knn):", len(rawMatches))
        return self.check_distance(rawMatches, self.ratio)

class FLANN(Match):
    def __init__(self, method, ratio):
        super().__init__(method)
        self.ratio=ratio

    def matchKP(self, featuresA, featuresB):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        rawMatches = flann.knnMatch(featuresA,featuresB,k=2)
        print("Raw matches (flann):", len(rawMatches))
        return self.check_distance(rawMatches, self.ratio)
    
def get_match(img1, img2, kpsA, kpsB, featuresA, featuresB, method, match):

    if match == 'Brute':
        l=BruteForce(method=method)
    elif match == 'KNN':
        l=KNN(method=method, ratio=1)
    elif match == 'FLANN':
        l=FLANN(method=method, ratio=1)
 
    matches = l.matchKP(featuresA, featuresB)
    res = l.show_feature_match(img1, kpsA, img2, kpsB, matches)
    return matches