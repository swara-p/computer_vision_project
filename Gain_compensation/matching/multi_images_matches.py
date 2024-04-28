import cv2

from images import Image
from matching.pair_match import PairMatch
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
    


class MultiImageMatches:
    def __init__(self, images: list[Image],match_method,method, ratio: float = 1) -> None:
        """
        Create a new MultiImageMatches object.

        Args:
            images: images to compare
            ratio: ratio used for the Lowe's ratio test
        """
        self.images = images
        self.matches = {image.path: {} for image in images}
        self.ratio = ratio
        self.match_method=match_method
        self.method=method

    def get_matches(self, image_a: Image, image_b: Image) -> list:
        """
        Get matches for the given images.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            matches: List of matches between the two images
        """
        if image_b.path not in self.matches[image_a.path]:
            matches = self.compute_matches(image_a, image_b)
            self.matches[image_a.path][image_b.path] = matches

        return self.matches[image_a.path][image_b.path]

    def get_pair_matches(self, max_images: int = 6) -> list[PairMatch]:
        """
        Get the pair matches for the given images.

        Args:
            max_images: Number of matches maximum for each image

        Returns:
            pair_matches: List of pair matches
        """
        pair_matches = []
        for i, image_a in enumerate(self.images):
            possible_matches = sorted(
                self.images[:i] + self.images[i + 1 :],
                key=lambda image, ref=image_a: len(self.get_matches(ref, image)),
                reverse=True,
            )[:max_images]
            for image_b in possible_matches:
                if self.images.index(image_b) > i:
                    pair_match = PairMatch(image_a, image_b, self.get_matches(image_a, image_b))
                    if pair_match.is_valid():
                        pair_matches.append(pair_match)
        return pair_matches

    def compute_matches(self, image_a: Image, image_b: Image) -> list:
        """
        Compute matches between image_a and image_b.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            matches: Matches between image_a and image_b
        """
        if self.match_method == 'Brute':
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            raw_matches = matcher.knnMatch(image_a.features, image_b.features, 2)
            matches = []

            for m, n in raw_matches:
                # ensure the distance is within a certain ratio of each
                # other (i.e. Lowe's ratio test)
                if m.distance < n.distance * self.ratio:
                    matches.append(m)
        elif self.match_method == 'KNN':
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            raw_matches = matcher.knnMatch(image_a.features, image_b.features, 2)
            matches = []

            for m, n in raw_matches:
                # ensure the distance is within a certain ratio of each
                # other (i.e. Lowe's ratio test)
                if m.distance < n.distance * self.ratio:
                    matches.append(m)

        elif self.match_method == 'FLANN':
            l=FLANN(method=self.method, ratio=1)
            matches = l.matchKP(image_a.features, image_b.features)
    
        # matches = l.matchKP(image_a.features, image_b.features)
        return matches
        # matcher = cv2.DescriptorMatcher_create("BruteForce")

        # raw_matches = matcher.knnMatch(image_a.features, image_b.features, 2)
        # matches = []

        # for m, n in raw_matches:
        #     # ensure the distance is within a certain ratio of each
        #     # other (i.e. Lowe's ratio test)
        #     if m.distance < n.distance * self.ratio:
        #         matches.append(m)

        # return matches
# def get_match(img1, img2, kpsA, kpsB, featuresA, featuresB, method, match):

#     if match == 'Brute':
#         l=BruteForce(method=method)
#     elif match == 'KNN':
#         l=KNN(method=method, ratio=1)
#     elif match == 'FLANN':
#         l=FLANN(method=method, ratio=1)
 
#     matches = l.matchKP(featuresA, featuresB)
#     res = l.show_feature_match(img1, kpsA, img2, kpsB, matches)
#     return matches