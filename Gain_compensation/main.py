"""
Main script.
Create panoramas from a set of images.
"""

import argparse
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import numpy as np

from images import Image
from matching import (
    MultiImageMatches,
    PairMatch,
    build_homographies,
    find_connected_components,
)
from rendering import multi_band_blending, set_gain_compensations, simple_blending

# parser = argparse.ArgumentParser(
#     description="Create panoramas from a set of images. \
#                  All the images must be in the same directory. \
#                  Multiple panoramas can be created at once"
# )

# parser.add_argument(dest="data_dir", type=Path, help="directory containing the images")
# parser.add_argument(
#     "-mbb",
#     "--multi-band-blending",
#     action="store_true",
#     help="use multi-band blending instead of simple blending",
# )
# parser.add_argument(
#     "--size", type=int, help="maximum dimension to resize the images to"
# )
# parser.add_argument(
#     "--num-bands", type=int, default=5, help="number of bands for multi-band blending"
# )
# parser.add_argument(
#     "--mbb-sigma", type=float, default=1, help="sigma for multi-band blending"
# )

# parser.add_argument(
#     "--gain-sigma-n", type=float, default=10, help="sigma_n for gain compensation"
# )
# parser.add_argument(
#     "--gain-sigma-g", type=float, default=0.1, help="sigma_g for gain compensation"
# )

# parser.add_argument(
#     "-v", "--verbose", action="store_true", help="increase output verbosity"
# )

# args = vars(parser.parse_args())
args={"verbose":2,"data_dir":"image-stitching\samples\mountain","size":None,"gain_sigma_n":10,"gain_sigma_g":0.1,"mbb_sigma":1,"num_bands":5}
if args["verbose"]:
    logging.basicConfig(level=logging.INFO)

print("Gathering images...")

valid_images_extensions = {".jpg", ".png", ".bmp", ".jpeg"}

data_dir_path = Path(args["data_dir"])
image_paths = [
    str(filepath)
    for filepath in data_dir_path.iterdir()
    if filepath.suffix.lower() in valid_images_extensions
]

#images = [Image(path, args.get("size")) for path in image_paths]
images = [Image(path,args.get('size')) for path in image_paths]

print("Found %d images", len(images))
print("Computing features...")
print("Type method out of - SIFT, AKAZE, BRISK, ORB, FREAK")
method=input()
for image in images:
    image.compute_features("SIFT")

print("Matching images with features...")
print("Choose match method out of - Brute, KNN, FLANN")
match_method=input()
matcher = MultiImageMatches(images,match_method=match_method,method=method)
pair_matches: list[PairMatch] = matcher.get_pair_matches()
pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True)

print("Finding connected components...")

connected_components = find_connected_components(pair_matches)

print("Found %d connected components", len(connected_components))
print("Building homographies...")

build_homographies(connected_components, pair_matches)

time.sleep(0.1)


for connected_component in connected_components:
    component_matches = [
        pair_match
        for pair_match in pair_matches
        if pair_match.image_a in connected_component
    ]

    set_gain_compensations(
        connected_component,
        component_matches,
        sigma_n=args["gain_sigma_n"],
        sigma_g=args["gain_sigma_g"],
    )

time.sleep(0.1)
print(len(images))
# print("Do you want gain compensantion?- Y/N")
# answer=input()
# if answer=='Y':
# print("Computing gain compensations...")
# for image in images:
#     image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)

# print(images[0])
# print(images[0].shape)
results = []
print("Choose blending technique out of - Simple, CutPaste, Alpha, Multiband, Poisson-Scratch, Poisson-Inbuilt, Deepblend")

technique=input()
print(f"Applying {technique} blending ---")
if technique=="Deepblend" or technique=="Multiband":
    results = [
        multi_band_blending(
            connected_component,
            num_bands=args["num_bands"],
            sigma=args["mbb_sigma"],
        )
        for connected_component in connected_components
    ]
else:

    results = [
        simple_blending(connected_component)
        for connected_component in connected_components
    ]

# logging.info("Saving results to %s", args["data_dir"] + "\results")

# (args["data_dir"] +"\results").mkdir(exist_ok=True, parents=True)
import cv2
import os

# Assuming 'images' is a list of NumPy arrays representing the images
# and 'output_folder' is the path to the folder where you want to save the images
print(len(results))
cv2.imwrite(f'aa2\{method}_{match_method}_{technique}.png',results[0])

