import numpy as np


VIDEO1_PATH = "videos\\Stream1Recording.avi"
VIDEO2_PATH = "videos\\Stream2Recording.avi"
VIDEO3_PATH = "videos\\Stream3Recording.avi"

IP1 = "http://10.130.8.187:8080/video"
IP2 = "http://10.130.8.187:8080/video"
IP3 = "http://10.130.8.187:8080/video"
# IP2 = "http://10.130.132.243:8080/video"
# IP3 = "http://10.130.13.170:8080/video"

MAX_DIST = 86
MIN_DIST = 10

HOMO_SIZE = (738, 409)

A = np.array(
    [
        [-3.18197280e-01, -3.05873793e00, 1.42293676e03],
        [2.12185144e-01, -1.83588790e00, 7.20665027e02],
        [-4.35247998e-05, -4.44636552e-03, 1.00000000e00],
    ]
)


B = np.array(
    [
        [7.76665778e-02, -1.36798585e00, 5.64848247e02],
        [6.66388825e-02, 1.69389517e-02, -2.25449838e02],
        [-9.73617871e-05, -2.36999031e-03, 1.00000000e00],
    ]
)

C = np.array(
    [
        [1.32579294e-01, -4.24335478e-01, -4.34272363e01],
        [-8.48398008e-02, -1.40574971e-01, 3.72730750e-01],
        [4.70467843e-05, -2.13970404e-03, 1.00000000e00],
    ]
)

A2 = np.array(
    [
        [-1.23554027e-01, -3.85322224e00, 1.43825150e03],
        [2.24694038e-01, -2.26655077e00, 7.40911265e02],
        [2.79068272e-04, -5.28196793e-03, 1.00000000e00],
    ]
)

B2 = np.array(
    [
        [6.33597306e-02, -1.79967013e00, 6.34938030e02],
        [6.49471813e-02, 1.85487418e-01, -3.14975632e02],
        [-1.28869885e-04, -2.88646525e-03, 1.00000000e00],
    ]
)

C2 = np.array(
    [
        [4.98464634e-02, -4.31779385e-01, 9.91183543e01],
        [-4.99976284e-02, -1.70373928e-01, 5.75077114e01],
        [-9.15103086e-06, -2.03096591e-03, 1.00000000e00],
    ]
)


def menu():
    MENU = """********************************************
    Press 1: Record video
    Press 2: Livestream from webcam
    Press 3: Object Detection on Pre-Recorded Video
    Press 4: Object Detection on Live Stream
    Press 5: Top View Projection of Pre-Recorded Videos
    Press 6: Top View Projection of Live Videos
    Press 7: Top View Object Detection on Pre-Recorded Videos
             ...
    Press 9: SOP Violation on Pre-recorded Videos         
             ...
    Press 11: Heat map 1 on Pre-recrded Videos
             ...
    Press 13: Heat map 1 on Pre-recrded Videos
             ...
    Press 15: Heat map 1 on Pre-recrded Videos
********************************************"""
    print(MENU)
    return input("> ")
