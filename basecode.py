import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import queue

VIDEO1 = "vids new\\Stream1Recording.avi"
VIDEO2 = "vids new\\Stream2Recording.avi"
VIDEO3 = "vids new\\Stream3Recording.avi"


dst = "imgs/dst.jpg", -1
src1 = "imgs/src1.jpg", -1
src2 = "imgs/src2.jpg", -1
src3 = "imgs/src3.jpg", -1


def getHomography():
    homographies = []
    drawing = False  # true if mouse is pressed
    src_x, src_y = -1, -1
    dst_x, dst_y = -1, -1

    src_list = []
    dst_list = []

    # mouse callback function
    def select_points_src(event, x, y, flags, param):
        global src_x, src_y, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            src_x, src_y = x, y
            cv2.circle(src_copy, (x, y), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    # mouse callback function
    def select_points_dst(event, x, y, flags, param):
        global dst_x, dst_y, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            dst_x, dst_y = x, y
            cv2.circle(dst_copy, (x, y), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def get_plan_view(src, dst):
        src_pts = np.array(src_list).reshape(-1, 1, 2)
        dst_pts = np.array(dst_list).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print("H:")
        print(H)
        plan_view = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
        return plan_view, H

    def merge_views(src, dst):
        plan_view = get_plan_view(src, dst)
        for i in range(0, dst.shape[0]):
            for j in range(0, dst.shape[1]):
                if (
                    plan_view.item(i, j, 0) == 0
                    and plan_view.item(i, j, 1) == 0
                    and plan_view.item(i, j, 2) == 0
                ):
                    plan_view.itemset((i, j, 0), dst.item(i, j, 0))
                    plan_view.itemset((i, j, 1), dst.item(i, j, 1))
                    plan_view.itemset((i, j, 2), dst.item(i, j, 2))
        return plan_view

    src = cv2.imread("imgs/src.jpg", -1)

    src_copy = src.copy()
    cv2.namedWindow("src")
    src_copy = cv2.resize(src_copy, (960, 540))
    # cv2.resizeWindow('src',src.shape[0], src.shape[1])
    cv2.resizeWindow("src", 960, 540)
    cv2.moveWindow("src", 80, 80)
    cv2.setMouseCallback("src", select_points_src)

    dst = cv2.imread("imgs/dst.jpg", -1)
    dst_copy = dst.copy()
    cv2.namedWindow("dst")
    cv2.moveWindow("dst", 780, 80)
    cv2.setMouseCallback("dst", select_points_dst)

    while 1:
        cv2.imshow("src", src_copy)
        cv2.imshow("dst", dst_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            print("save points")
            cv2.circle(src_copy, (src_x, src_y), 5, (0, 255, 0), -1)
            cv2.circle(dst_copy, (dst_x, dst_y), 5, (0, 255, 0), -1)
            src_list.append([src_x, src_y])
            dst_list.append([dst_x, dst_y])
            print("src points:")
            print(src_list)
            print("dst points:")
            print(dst_list)
        elif k == ord("h"):
            print("create plan view")
            plan_view = get_plan_view(src, dst)
            homographies.append(plan_view)
            break
    cv2.destroyAllWindows()


def main():

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        "D:\\Fall2021\\Computer visions\\project\\best.pt",
    )

    # print("Press 1 for pre-recorded videos, 2 for live stream: ")

    menu = """********************************************
    Press 1: Record video
    Press 2: Livestream from webcam
    Press 3: Run on Recorded Video
    press 4: Homographic Top view
    press 5: Homographic Top view on live feed (Assuming Camera in same location)
    press 6: SOP violation
    Press 7: Static Heatmap
    Press 8: Animated Heatmap
    press 9: SOP heatmap
    ********************************************"""

    print(menu)

    homo1 = np.float32(
        [
            [-3.18197280e-01, -3.05873793e00, 1.42293676e03],
            [2.12185144e-01, -1.83588790e00, 7.20665027e02],
            [-4.35247998e-05, -4.44636552e-03, 1.00000000e00],
        ]
    )

    homo2 = np.float32(
        [
            [7.76665778e-02, -1.36798585e00, 5.64848247e02],
            [6.66388825e-02, 1.69389517e-02, -2.25449838e02],
            [-9.73617871e-05, -2.36999031e-03, 1.00000000e00],
        ]
    )

    homo3 = np.float32(
        [
            [1.20643191e-01, -4.22760079e-01, -1.46117703e01],
            [-7.40937092e-02, -1.84339641e-01, 3.82869967e01],
            [3.25874717e-05, -2.06228604e-03, 1.00000000e00],
        ]
    )

    homogay = np.array(
        [
            [-1.23554027e-01, -3.85322224e00, 1.43825150e03],
            [2.24694038e-01, -2.26655077e00, 7.40911265e02],
            [2.79068272e-04, -5.28196793e-03, 1.00000000e00],
        ]
    )

    homogay2 = np.array(
        [
            [6.33597306e-02, -1.79967013e00, 6.34938030e02],
            [6.49471813e-02, 1.85487418e-01, -3.14975632e02],
            [-1.28869885e-04, -2.88646525e-03, 1.00000000e00],
        ]
    )

    homogay3 = np.array(
        [
            [4.98464634e-02, -4.31779385e-01, 9.91183543e01],
            [-4.99976284e-02, -1.70373928e-01, 5.75077114e01],
            [-9.15103086e-06, -2.03096591e-03, 1.00000000e00],
        ]
    )

    option = int(input())

    if option == 1:
        # Record video
        windowName = "Sample Feed from Camera 1"
        windowName2 = "Feed from 2"
        windowName3 = "Feed from 3"
        cv2.namedWindow(windowName)

        # capture1 = cv2.VideoCapture(0)  # laptop's camera
        capture1 = cv2.VideoCapture(
            "http://10.130.8.187:8080/video"
        )  # sample code for mobile camera video capture using IP camera
        capture2 = cv2.VideoCapture("http://10.130.138.70:8080/video")
        capture3 = cv2.VideoCapture("http://10.130.23.3:8080/video")
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # define size for recorded video frame for video 1
        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)

        # for 2nd camera
        width2 = int(capture2.get(3))
        height2 = int(capture2.get(4))
        size2 = (width2, height2)

        # for 3nd camera

        width3 = int(capture3.get(3))
        height3 = int(capture3.get(4))
        size3 = (width3, height3)
        # --------------------------------------------------------------------------------------------------------------------------------------
        # frame of size is being created and stored in .avi file
        optputFile1 = cv2.VideoWriter(
            "Stream1Recording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size1
        )

        # for 2nd camera
        optputFile2 = cv2.VideoWriter(
            "Stream2Recording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size2
        )

        # for 3rd camera
        optputFile3 = cv2.VideoWriter(
            "Stream3Recording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size3
        )
        # -------------------------------------------------------------------------------------------------------------------------------------
        # check if feed exists or not for camera 1
        if capture1.isOpened():
            ret1, frame1 = capture1.read()
        else:
            ret1 = False

        # for 2nd camera
        if capture2.isOpened():
            ret2, frame2 = capture2.read()
        else:
            ret2 = False

        # for 3nd camera
        if capture3.isOpened():
            ret3, frame3 = capture3.read()
        else:
            ret3 = False
        # ---------------------------------------------------------------------------------------------------------------------------------
        while ret1:
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            # sample feed display from camera 1
            cv2.imshow(windowName, frame1)
            cv2.imshow(windowName2, frame2)
            cv2.imshow(windowName3, frame3)
            # cv2.imshow(windowName, frame2)
            # saves the frame from camera 1
            optputFile1.write(frame1)
            optputFile2.write(frame2)
            optputFile3.write(frame3)
            # escape key (27) to exit
            if cv2.waitKey(1) == 27:
                break
        # ----------------------------------------------------------------------------------------------------------------------------------
        capture1.release()
        optputFile1.release()
        # cv2.destroyAllWindows()

        # for 2nd camera
        capture2.release()
        optputFile2.release()
        # cv2.destroyAllWindows()

        # for 3nd camera
        capture3.release()
        optputFile3.release()
        cv2.destroyAllWindows()
    # ------------------------------------------------------------------------------------------------------------------------------------

    elif option == 2:
        # live stream
        windowName1 = "Live Stream Camera 1"
        windowName2 = "Feed from 2"
        windowName3 = "Feed from 3"
        cv2.namedWindow(windowName1)

        capture1 = cv2.VideoCapture("http://10.130.8.187:8080/video")  # laptop's camera
        # capture2=cv2.VideoCapture("http://10.130.8.187:8080/video")
        capture2 = cv2.VideoCapture(0)
        capture3 = cv2.VideoCapture("http://10.130.8.187:8080/video")
        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False
        if capture2.isOpened():  # check if feed exists or not for camera 1
            ret2, frame2 = capture2.read()
        else:
            ret1 = False
        if capture3.isOpened():  # check if feed exists or not for camera 1
            ret3, frame3 = capture3.read()
        else:
            ret1 = False

        while ret1:
            ret1, frame1 = capture1.read()

            result1 = model(frame1)
            result1.render()
            frame1 = result1.imgs[0]

            cv2.imshow(windowName1, frame1)

            ret2, frame2 = capture2.read()

            result2 = model(frame2)
            result2.render()
            frame2 = result2.imgs[0]

            cv2.imshow(windowName2, frame2)

            ret3, frame3 = capture3.read()

            result3 = model(frame3)
            result3.render()

            frame3 = result3.imgs[0]

            cv2.imshow(windowName3, frame3)

            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        cv2.destroyAllWindows()

    elif option == 3:
        windowName1 = "Pre-recorded video"
        windowName2 = "Feed from 2"
        windowName3 = "Feed from 3"
        cv2.namedWindow(windowName1)

        capture1 = cv2.VideoCapture(VIDEO1)
        capture2 = cv2.VideoCapture(VIDEO2)
        capture3 = cv2.VideoCapture(VIDEO3)

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False

        while ret1:
            ret1, frame1 = capture1.read()

            result1 = model(frame1)
            result1.render()
            frame1 = result1.imgs[0]

            cv2.imshow(windowName1, frame1)

            ret2, frame2 = capture2.read()

            result2 = model(frame2)
            result2.render()
            frame2 = result2.imgs[0]

            cv2.imshow(windowName2, frame2)

            ret3, frame3 = capture3.read()

            result3 = model(frame3)
            result3.render()
            frame3 = result3.imgs[0]

            cv2.imshow(windowName3, frame3)

            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        cv2.destroyAllWindows()

    elif option == 4:
        windowName1 = "Pre-recorded video"
        windowName2 = "Feed from 2"
        windowName3 = "Feed from 3"

        width1 = int(1000)
        height1 = int(1000)
        size1 = (width1, height1)

        optputFile1 = cv2.VideoWriter(
            "HomographicRecording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size1
        )
        cv2.namedWindow("feed")

        capture1 = cv2.VideoCapture(VIDEO1)
        capture2 = cv2.VideoCapture(VIDEO2)
        capture3 = cv2.VideoCapture(VIDEO3)

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False
        count = 0
        while ret1:
            result1 = model(frame1)

            ret2, frame2 = capture2.read()
            result2 = model(frame2)
            ret3, frame3 = capture3.read()
            result3 = model(frame3)

            dst1 = cv2.warpPerspective(
                frame1, homo1, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst2 = cv2.warpPerspective(
                frame2, homo2, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst3 = cv2.warpPerspective(
                frame3, homo3, (1000, 800), flags=cv2.INTER_LINEAR
            )

            vis = cv2.addWeighted(dst1, 0.5, dst2, 0.5, 0)
            vis = cv2.addWeighted(vis, 0.8, dst3, 0.2, 0)

            labels, cord_thres1 = (
                result1.xyxyn[0][:, -1].numpy(),
                result1.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres1.shape[0]):
                loc = cord_thres1[a][0] * 1920
                loc2 = cord_thres1[a][1] * 1080
                centre1 = np.array([int(loc), int(loc2), 1])
                centre1 = np.matmul(homogay, centre1)
                centre1 = (centre1[0] // centre1[2], centre1[1] // centre1[2])
                centre1 = (int(centre1[0]), int(centre1[1]))
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre1, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre1, 5, (0, 0, 255), -1)

            labels, cord_thres2 = (
                result2.xyxyn[0][:, -1].numpy(),
                result2.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres2.shape[0]):
                loc = cord_thres2[a][0] * 1920
                loc2 = cord_thres2[a][1] * 1080
                centre2 = np.array([int(loc), int(loc2), 1])
                centre2 = np.matmul(homogay2, centre2)
                centre2 = (centre2[0] // centre2[2], centre2[1] // centre2[2])
                centre2 = (int(centre2[0]), int(centre2[1]))
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre2, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre2, 5, (0, 0, 255), -1)

            labels, cord_thres3 = (
                result3.xyxyn[0][:, -1].numpy(),
                result3.xyxyn[0][:, :-1].numpy(),
            )

            for a in range(cord_thres3.shape[0]):
                loc = cord_thres3[a][0] * 1920
                loc2 = cord_thres3[a][1] * 1080
                centre3 = np.array([int(loc), int(loc2), 1])
                centre3 = np.matmul(homogay3, centre3)
                centre3 = (centre3[0] // centre3[2], centre3[1] // centre3[2])
                centre3 = (int(centre3[0]), int(centre3[1]))
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre3, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre3, 5, (0, 0, 255), -1)

            imgName = "img" + str(count) + ".jpg"
            count += 1
            cv2.imwrite(
                "D:\\Fall2021\\Computer visions\\project\\Top view Images from recorded Files\\"
                + imgName,
                vis,
            )
            cv2.imshow("feed", vis)
            optputFile1.write(vis)
            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        optputFile1.release()
        cv2.destroyAllWindows()

    elif option == 5:

        cv2.namedWindow("feed")

        capture1 = cv2.VideoCapture("http://10.130.8.187:8080/video")
        capture2 = cv2.VideoCapture("http://10.130.8.187:8080/video")
        # capture2 = cv2.VideoCapture(0)      # laptop's camera
        capture3 = cv2.VideoCapture("http://10.130.8.187:8080/video")

        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)

        optputFile1 = cv2.VideoWriter(
            "LiveHomographicRecording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size1
        )

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False
        count = 0
        while ret1:
            result1 = model(frame1)

            ret2, frame2 = capture2.read()
            result2 = model(frame2)
            ret3, frame3 = capture3.read()
            result3 = model(frame3)

            dst1 = cv2.warpPerspective(
                frame1, homo1, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst2 = cv2.warpPerspective(
                frame2, homo2, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst3 = cv2.warpPerspective(
                frame3, homo3, (1000, 800), flags=cv2.INTER_LINEAR
            )

            vis = cv2.addWeighted(dst1, 0.5, dst2, 0.5, 0)
            vis = cv2.addWeighted(vis, 0.8, dst3, 0.2, 0)

            labels, cord_thres1 = (
                result1.xyxyn[0][:, -1].numpy(),
                result1.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres1.shape[0]):
                loc = cord_thres1[a][0] * 1920
                loc2 = cord_thres1[a][1] * 1080
                centre1 = np.array([int(loc), int(loc2), 1])
                centre1 = np.matmul(homogay, centre1)
                centre1 = (centre1[0] // centre1[2], centre1[1] // centre1[2])
                centre1 = (int(centre1[0]), int(centre1[1]))
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre1, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre1, 5, (0, 0, 255), -1)

            labels, cord_thres2 = (
                result2.xyxyn[0][:, -1].numpy(),
                result2.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres2.shape[0]):
                loc = cord_thres2[a][0] * 1920
                loc2 = cord_thres2[a][1] * 1080
                centre2 = np.array([int(loc), int(loc2), 1])
                centre2 = np.matmul(homogay2, centre2)
                centre2 = (centre2[0] // centre2[2], centre2[1] // centre2[2])
                centre2 = (int(centre2[0]), int(centre2[1]))
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre2, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre2, 5, (0, 0, 255), -1)

            labels, cord_thres3 = (
                result3.xyxyn[0][:, -1].numpy(),
                result3.xyxyn[0][:, :-1].numpy(),
            )

            for a in range(cord_thres3.shape[0]):
                loc = cord_thres3[a][0] * 1920
                loc2 = cord_thres3[a][1] * 1080
                centre3 = np.array([int(loc), int(loc2), 1])
                centre3 = np.matmul(homogay3, centre3)
                centre3 = (centre3[0] // centre3[2], centre3[1] // centre3[2])
                centre3 = (int(centre3[0]), int(centre3[1]))
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre3, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre3, 5, (0, 0, 255), -1)
            imgName = "img" + str(count) + ".jpg"
            cv2.imwrite(
                "D:\\Fall2021\\Computer visions\\Top vie images from live feed\\"
                + imgName,
                vis,
            )
            cv2.imshow("feed", vis)
            # optputFile1.write(vis)
            if cv2.waitKey(1) == 27:
                break
        capture1.release()
        optputFile1.release()

    elif option == 6:
        windowName1 = "Pre-recorded video"
        windowName2 = "Feed from 2"
        windowName3 = "Feed from 3"

        width1 = int(1000)
        height1 = int(1000)
        size1 = (width1, height1)

        optputFile1 = cv2.VideoWriter(
            "HomographicRecording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size1
        )
        cv2.namedWindow("feed")

        capture1 = cv2.VideoCapture(VIDEO1)
        capture2 = cv2.VideoCapture(VIDEO2)
        capture3 = cv2.VideoCapture(VIDEO3)

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False
        count = 0
        while ret1:
            ret1, frame1 = capture1.read()
            result1 = model(frame1)

            ret2, frame2 = capture2.read()
            result2 = model(frame2)
            ret3, frame3 = capture3.read()
            result3 = model(frame3)

            dst1 = cv2.warpPerspective(
                frame1, homo1, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst2 = cv2.warpPerspective(
                frame2, homo2, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst3 = cv2.warpPerspective(
                frame3, homo3, (1000, 800), flags=cv2.INTER_LINEAR
            )

            vis = cv2.addWeighted(dst1, 0.5, dst2, 0.5, 0)
            vis = cv2.addWeighted(vis, 0.8, dst3, 0.2, 0)

            labels, cord_thres1 = (
                result1.xyxyn[0][:, -1].numpy(),
                result1.xyxyn[0][:, :-1].numpy(),
            )
            points = []
            for a in range(cord_thres1.shape[0]):
                loc = cord_thres1[a][0] * 1920
                loc2 = cord_thres1[a][1] * 1080
                centre1 = np.array([int(loc), int(loc2), 1])
                centre1 = np.matmul(homogay, centre1)
                centre1 = (centre1[0] // centre1[2], centre1[1] // centre1[2])
                centre1 = (int(centre1[0]), int(centre1[1]))
                points.append(centre1)
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre1, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre1, 5, (0, 0, 255), -1)

            labels, cord_thres2 = (
                result2.xyxyn[0][:, -1].numpy(),
                result2.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres2.shape[0]):
                loc = cord_thres2[a][0] * 1920
                loc2 = cord_thres2[a][1] * 1080
                centre2 = np.array([int(loc), int(loc2), 1])
                centre2 = np.matmul(homogay2, centre2)
                centre2 = (centre2[0] // centre2[2], centre2[1] // centre2[2])
                centre2 = (int(centre2[0]), int(centre2[1]))
                points.append(centre2)
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre2, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre2, 5, (0, 0, 255), -1)

            labels, cord_thres3 = (
                result3.xyxyn[0][:, -1].numpy(),
                result3.xyxyn[0][:, :-1].numpy(),
            )

            for a in range(cord_thres3.shape[0]):
                loc = cord_thres3[a][0] * 1920
                loc2 = cord_thres3[a][1] * 1080
                centre3 = np.array([int(loc), int(loc2), 1])
                centre3 = np.matmul(homogay3, centre3)
                centre3 = (centre3[0] // centre3[2], centre3[1] // centre3[2])
                centre3 = (int(centre3[0]), int(centre3[1]))
                points.append(centre3)
                if labels[a] == 1:
                    vis = cv2.circle(vis, centre3, 5, (255, 0, 0), -1)
                else:
                    vis = cv2.circle(vis, centre3, 5, (0, 0, 255), -1)

            for i in range(len(points)):
                for j in range(i, len(points)):
                    p1 = np.array([[points[i][0], points[i][1]]])
                    p2 = np.array([[points[j][0], points[j][1]]])
                    diff = p1 - p2
                    if np.linalg.norm(diff) < 30 and np.linalg.norm(diff) > 10:
                        vis = cv2.putText(
                            vis,
                            "SOP VIOLATION",
                            points[i],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        vis = cv2.line(vis, points[i], points[j], (0, 255, 255), 3)

            cv2.imshow("feed", vis)
            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        optputFile1.release()
        cv2.destroyAllWindows()

    elif option == 7:
        k = 15
        gauss = cv2.getGaussianKernel(k, np.sqrt(64))
        gauss = gauss * gauss.T
        gauss = gauss / gauss[int(k / 2), int(k / 2)]

        width1 = int(1000)
        height1 = int(1000)
        size1 = (width1, height1)

        optputFile1 = cv2.VideoWriter(
            "HomographicRecording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size1
        )
        cv2.namedWindow("feed")

        capture1 = cv2.VideoCapture(VIDEO1)
        capture2 = cv2.VideoCapture(VIDEO2)
        capture3 = cv2.VideoCapture(VIDEO3)

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False
        count = 0
        point = []
        while ret1:
            ret1, frame1 = capture1.read()
            result1 = model(frame1)

            ret2, frame2 = capture2.read()
            result2 = model(frame2)
            ret3, frame3 = capture3.read()
            result3 = model(frame3)

            dst1 = cv2.warpPerspective(
                frame1, homo1, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst2 = cv2.warpPerspective(
                frame2, homo2, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst3 = cv2.warpPerspective(
                frame3, homo3, (1000, 800), flags=cv2.INTER_LINEAR
            )

            vis = cv2.addWeighted(dst1, 0.5, dst2, 0.5, 0)
            vis = cv2.addWeighted(vis, 0.8, dst3, 0.2, 0)

            labels, cord_thres1 = (
                result1.xyxyn[0][:, -1].numpy(),
                result1.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres1.shape[0]):
                loc = cord_thres1[a][0] * 1920
                loc2 = cord_thres1[a][1] * 1080
                centre1 = np.array([int(loc), int(loc2), 1])
                centre1 = np.matmul(homogay, centre1)
                centre1 = (centre1[0] // centre1[2], centre1[1] // centre1[2])
                centre1 = (int(centre1[0]), int(centre1[1]))
                point.append(centre1)

            labels, cord_thres2 = (
                result2.xyxyn[0][:, -1].numpy(),
                result2.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres2.shape[0]):
                loc = cord_thres2[a][0] * 1920
                loc2 = cord_thres2[a][1] * 1080
                centre2 = np.array([int(loc), int(loc2), 1])
                centre2 = np.matmul(homogay2, centre2)
                centre2 = (centre2[0] // centre2[2], centre2[1] // centre2[2])
                centre2 = (int(centre2[0]), int(centre2[1]))
                point.append(centre2)

            labels, cord_thres3 = (
                result3.xyxyn[0][:, -1].numpy(),
                result3.xyxyn[0][:, :-1].numpy(),
            )

            for a in range(cord_thres3.shape[0]):
                loc = cord_thres3[a][0] * 1920
                loc2 = cord_thres3[a][1] * 1080
                centre3 = np.array([int(loc), int(loc2), 1])
                centre3 = np.matmul(homogay3, centre3)
                centre3 = (centre3[0] // centre3[2], centre3[1] // centre3[2])
                centre3 = (int(centre3[0]), int(centre3[1]))
                point.append(centre3)

            vis = vis.astype(float) / 255
            imgH = np.zeros((vis.shape[0], vis.shape[1], 3)).astype(np.float32)
            j = (
                cv2.cvtColor(
                    cv2.applyColorMap(
                        ((1 - gauss) * 255).astype(np.uint8), cv2.COLORMAP_JET
                    ),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255
            )

            heatmap = None
            for p in point:
                b = imgH[
                    p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                    p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                    :,
                ]
                c = j + b
                imgH[
                    p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                    p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                    :,
                ] = c

                g = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY)
                mask = np.where(g > 0.2, 1, 0).astype(np.float32)
                mask_3 = (
                    np.ones((vis.shape[0], vis.shape[1], 3)) * (1 - mask)[:, :, None]
                )
                mask_4 = imgH * (mask)[:, :, None]
                new_top = mask_3 * vis
                heatmap = new_top + mask_4

            cv2.imshow("feed", heatmap)
            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        optputFile1.release()
        cv2.destroyAllWindows()

    elif option == 8:
        k = 15
        gauss = cv2.getGaussianKernel(k, np.sqrt(64))
        gauss = gauss * gauss.T
        gauss = gauss / gauss[int(k / 2), int(k / 2)]

        width1 = int(1000)
        height1 = int(1000)
        size1 = (width1, height1)

        optputFile1 = cv2.VideoWriter(
            "HomographicRecording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size1
        )
        cv2.namedWindow("feed")

        capture1 = cv2.VideoCapture(VIDEO1)
        capture2 = cv2.VideoCapture(VIDEO2)
        capture3 = cv2.VideoCapture(VIDEO3)

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False
        count = 0
        point = []
        count = 0
        l1 = []
        while ret1:

            if count > 1:
                count = 0
                print(len(point))
                for i in point:
                    if i in l1:
                        point.remove(i)
                l1 = []
                print(len(point))

            ret1, frame1 = capture1.read()
            result1 = model(frame1)

            ret2, frame2 = capture2.read()
            result2 = model(frame2)
            ret3, frame3 = capture3.read()
            result3 = model(frame3)

            dst1 = cv2.warpPerspective(
                frame1, homo1, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst2 = cv2.warpPerspective(
                frame2, homo2, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst3 = cv2.warpPerspective(
                frame3, homo3, (1000, 800), flags=cv2.INTER_LINEAR
            )

            vis = cv2.addWeighted(dst1, 0.5, dst2, 0.5, 0)
            vis = cv2.addWeighted(vis, 0.8, dst3, 0.2, 0)

            labels, cord_thres1 = (
                result1.xyxyn[0][:, -1].numpy(),
                result1.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres1.shape[0]):
                loc = cord_thres1[a][0] * 1920
                loc2 = cord_thres1[a][1] * 1080
                centre1 = np.array([int(loc), int(loc2), 1])
                centre1 = np.matmul(homogay, centre1)
                centre1 = (centre1[0] // centre1[2], centre1[1] // centre1[2])
                centre1 = (int(centre1[0]), int(centre1[1]))
                point.append(centre1)
                l1.append(centre1)

            labels, cord_thres2 = (
                result2.xyxyn[0][:, -1].numpy(),
                result2.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres2.shape[0]):
                loc = cord_thres2[a][0] * 1920
                loc2 = cord_thres2[a][1] * 1080
                centre2 = np.array([int(loc), int(loc2), 1])
                centre2 = np.matmul(homogay2, centre2)
                centre2 = (centre2[0] // centre2[2], centre2[1] // centre2[2])
                centre2 = (int(centre2[0]), int(centre2[1]))
                point.append(centre2)
                l1.append(centre2)

            labels, cord_thres3 = (
                result3.xyxyn[0][:, -1].numpy(),
                result3.xyxyn[0][:, :-1].numpy(),
            )

            for a in range(cord_thres3.shape[0]):
                loc = cord_thres3[a][0] * 1920
                loc2 = cord_thres3[a][1] * 1080
                centre3 = np.array([int(loc), int(loc2), 1])
                centre3 = np.matmul(homogay3, centre3)
                centre3 = (centre3[0] // centre3[2], centre3[1] // centre3[2])
                centre3 = (int(centre3[0]), int(centre3[1]))
                point.append(centre3)
                l1.append(centre3)

            vis = vis.astype(float) / 255
            imgH = np.zeros((vis.shape[0], vis.shape[1], 3)).astype(np.float32)
            j = (
                cv2.cvtColor(
                    cv2.applyColorMap(
                        ((1 - gauss) * 255).astype(np.uint8), cv2.COLORMAP_JET
                    ),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255
            )

            heatmap = None
            for p in point:
                b = imgH[
                    p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                    p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                    :,
                ]
                c = j + b
                imgH[
                    p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                    p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                    :,
                ] = c

                g = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY)
                mask = np.where(g > 0.2, 1, 0).astype(np.float32)
                mask_3 = (
                    np.ones((vis.shape[0], vis.shape[1], 3)) * (1 - mask)[:, :, None]
                )
                mask_4 = imgH * (mask)[:, :, None]
                new_top = mask_3 * vis
                heatmap = new_top + mask_4

            count += 1

            cv2.imshow("feed", heatmap)
            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        optputFile1.release()
        cv2.destroyAllWindows()

    elif option == 9:
        k = 15
        gauss = cv2.getGaussianKernel(k, np.sqrt(64))
        gauss = gauss * gauss.T
        gauss = gauss / gauss[int(k / 2), int(k / 2)]

        width1 = int(1000)
        height1 = int(1000)
        size1 = (width1, height1)

        optputFile1 = cv2.VideoWriter(
            "HomographicRecording.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size1
        )
        cv2.namedWindow("feed")

        capture1 = cv2.VideoCapture(VIDEO1)
        capture2 = cv2.VideoCapture(VIDEO2)
        capture3 = cv2.VideoCapture(VIDEO3)

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False
        count = 0
        point = []
        count = 0

        sop = []
        while ret1:

            ret1, frame1 = capture1.read()
            result1 = model(frame1)

            ret2, frame2 = capture2.read()
            result2 = model(frame2)
            ret3, frame3 = capture3.read()
            result3 = model(frame3)

            dst1 = cv2.warpPerspective(
                frame1, homo1, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst2 = cv2.warpPerspective(
                frame2, homo2, (1000, 800), flags=cv2.INTER_LINEAR
            )
            dst3 = cv2.warpPerspective(
                frame3, homo3, (1000, 800), flags=cv2.INTER_LINEAR
            )

            vis = cv2.addWeighted(dst1, 0.5, dst2, 0.5, 0)
            vis = cv2.addWeighted(vis, 0.8, dst3, 0.2, 0)

            labels, cord_thres1 = (
                result1.xyxyn[0][:, -1].numpy(),
                result1.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres1.shape[0]):
                loc = cord_thres1[a][0] * 1920
                loc2 = cord_thres1[a][1] * 1080
                centre1 = np.array([int(loc), int(loc2), 1])
                centre1 = np.matmul(homogay, centre1)
                centre1 = (centre1[0] // centre1[2], centre1[1] // centre1[2])
                centre1 = (int(centre1[0]), int(centre1[1]))
                point.append(centre1)

            labels, cord_thres2 = (
                result2.xyxyn[0][:, -1].numpy(),
                result2.xyxyn[0][:, :-1].numpy(),
            )
            for a in range(cord_thres2.shape[0]):
                loc = cord_thres2[a][0] * 1920
                loc2 = cord_thres2[a][1] * 1080
                centre2 = np.array([int(loc), int(loc2), 1])
                centre2 = np.matmul(homogay2, centre2)
                centre2 = (centre2[0] // centre2[2], centre2[1] // centre2[2])
                centre2 = (int(centre2[0]), int(centre2[1]))
                point.append(centre2)

            labels, cord_thres3 = (
                result3.xyxyn[0][:, -1].numpy(),
                result3.xyxyn[0][:, :-1].numpy(),
            )

            for a in range(cord_thres3.shape[0]):
                loc = cord_thres3[a][0] * 1920
                loc2 = cord_thres3[a][1] * 1080
                centre3 = np.array([int(loc), int(loc2), 1])
                centre3 = np.matmul(homogay3, centre3)
                centre3 = (centre3[0] // centre3[2], centre3[1] // centre3[2])
                centre3 = (int(centre3[0]), int(centre3[1]))
                point.append(centre3)

            for i in range(len(point)):
                for j in range(i, len(point)):
                    p1 = np.array([[point[i][0], point[i][1]]])
                    p2 = np.array([[point[j][0], point[j][1]]])
                    diff = p1 - p2
                    if np.linalg.norm(diff) < 30 and np.linalg.norm(diff) > 10:
                        sop.append(point[j])
                        sop.append(point[i])

            vis = vis.astype(float) / 255
            imgH = np.zeros((vis.shape[0], vis.shape[1], 3)).astype(np.float32)
            j = (
                cv2.cvtColor(
                    cv2.applyColorMap(
                        ((1 - gauss) * 255).astype(np.uint8), cv2.COLORMAP_JET
                    ),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255
            )

            heatmap = None
            if len(sop) == 0:
                heatmap = vis
            for p in sop:
                b = imgH[
                    p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                    p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                    :,
                ]
                c = j + b
                imgH[
                    p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                    p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                    :,
                ] = c

                g = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY)
                mask = np.where(g > 0.2, 1, 0).astype(np.float32)
                mask_3 = (
                    np.ones((vis.shape[0], vis.shape[1], 3)) * (1 - mask)[:, :, None]
                )
                mask_4 = imgH * (mask)[:, :, None]
                new_top = mask_3 * vis
                heatmap = new_top + mask_4

            count += 1

            cv2.imshow("feed", heatmap)
            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        optputFile1.release()
        cv2.destroyAllWindows()

    else:
        print("Invalid option entered. Exiting...")


main()
