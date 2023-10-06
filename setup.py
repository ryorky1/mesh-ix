from setuptools import setup, find_packages
import os
import sys

_args  = {
    "name":"image-crop","version":0.01,"author":"Ryan O'Neal","email":"ryorky1@gmail.com",
    "packages":["imagecrop"],
    "install_requires":[
        "torch","torchvision","Pillow","scikit-image","tqdm","opencv-python","trimesh","PyOpenGL","ffmpeg"
    ],
    "include_package_data":True,
    "package_data":{'':['checkpoint_iter.*']}
}