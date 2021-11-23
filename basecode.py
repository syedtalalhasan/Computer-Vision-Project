import cv2
import torch

VIDEO1 = "videos\\Stream1Recording.avi"
VIDEO2 = "videos\\Stream2Recording.avi"
VIDEO3 = "videos\\Stream3Recording.avi"


def main():

    model = torch.hub.load("ultralytics/yolov5", "custom", " best.pt")
    # image = "images/img-a5.jpg"

    torch.nn.Module.dump_patches = True

    # print("Press 1 for pre-recorded videos, 2 for live stream: ")

    menu = """********************************************
    Press 1: Record video
    Press 2: Livestream from webcam
    Press 3: Run on Recorded Video
********************************************"""

    print(menu)

    option = int(input())

    if option == 1:
        # Record video
        windowName = "Sample Feed from Camera 1"
        cv2.namedWindow(windowName)

        # capture1 = cv2.VideoCapture(0)  # laptop's camera
        capture1 = cv2.VideoCapture(
            "http://10.130.16.238:8080/video"
        )  # sample code for mobile camera video capture using IP camera
        capture2 = cv2.VideoCapture("http://10.130.132.243:8080/video")
        capture3 = cv2.VideoCapture("http://10.130.13.170:8080/video")
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
        cv2.namedWindow(windowName1)

        capture1 = cv2.VideoCapture(0)  # laptop's camera

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False

        while ret1:
            ret1, frame1 = capture1.read()

            results = model(frame1)
            results.render()
            frame1 = results.imgs[0]

            cv2.imshow(windowName1, frame1)

            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        cv2.destroyAllWindows()

    elif option == 3:
        windowName1 = "Pre-recorded video"
        cv2.namedWindow(windowName1)

        capture1 = cv2.VideoCapture(VIDEO3)

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
        else:
            ret1 = False

        while ret1:
            ret1, frame1 = capture1.read()
            results = model(frame1)
            results.render()
            frame1 = results.imgs[0]

            cv2.imshow(windowName1, frame1)

            if cv2.waitKey(1) == 27:
                break

    else:
        print("Invalid option entered. Exiting...")


main()
