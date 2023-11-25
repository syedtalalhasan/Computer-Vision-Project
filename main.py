import cv2
import torch
import numpy as np
import util
from video_reader import VideoReader, VideoShower
import homography_computation
import time
import sys
from matplotlib import pyplot as plt


model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    "weights.pt",
    force_reload=True,  # uncomment out when changing weights
)


def main(path_list, M_list, m_list, dst_shape):

    option = int(util.menu())

    if option == 1:
        object_detection(path_list)

    elif option == 2:
        topview(path_list, M_list, dst_shape)

    elif option == 3:
        topview_object_detection(path_list, M_list, m_list, dst_shape)

    elif option == 4:
        SOP_violation(path_list, M_list, m_list, dst_shape)

    elif option == 5:
        heatmap1(path_list, M_list, m_list, dst_shape)

    elif option == 6:
        heatmap2(path_list, M_list, m_list, dst_shape)

    elif option == 7:
        heatmap3(path_list, M_list, m_list, dst_shape)

    else:
        print("why am I here? ")
        print("just to suffer")


def save_videos():
    # Record video
    window_name = "Sample Feed from Camera 1"
    cv2.namedWindow(window_name)

    # capture1 = cv2.VideoCapture(0)  # laptop's camera
    capture1 = cv2.VideoCapture(
        util.IP1
    )  # sample code for mobile camera video capture using IP camera
    capture2 = cv2.VideoCapture(util.IP2)
    capture3 = cv2.VideoCapture(util.IP3)
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
        cv2.imshow(window_name, frame1)
        # cv2.imshow(window_name, frame2)
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


def live_stream():
    window_name = "Live Stream Camera 1"

    reader = VideoReader(0).start()
    shower = VideoShower(window_name).start()

    while True:
        if reader.stopped or shower.stopped:
            reader.stop()
            shower.stop()
            break
        # shower.q.put(reader.frame)

        frame = reader.frame
        time.sleep(0.05)
        shower.q.put(frame)

        # cv2.imshow("video", reader.frame)
        # if cv2.waitKey(1) == 27:
        #     break

    # cv2.destroyAllWindows()


def live_object_detection():

    # * Function does not open three windows from the webcame
    # * uncomment out the code and replace 0 with IP address
    # * of webcam to be able to open three livestream windows
    # * at the same time

    reader1 = VideoReader(0).start()
    shower1 = VideoShower("Video1").start()
    # reader2 = VideoReader(0).start()
    # shower2 = VideoShower("Video1").start()
    # reader3 = VideoReader(0).start()
    # shower3 = VideoShower("Video1").start()

    while True:
        if (
            reader1.stopped
            or shower1.stopped
            # or reader2.stopped
            # or shower2.stopped
            # or reader3.stopped
            # or shower3.stopped
        ):
            reader1.stop()
            shower1.stop()
            # reader2.stop()
            # shower2.stop()
            # reader3.stop()
            # shower3.stop()
            break

        results1 = model(reader1.frame)
        results1.render()
        frame1 = results1.imgs[0]

        # results2 = model(reader2.frame)
        # results2.render()
        # frame2 = results2.imgs[0]

        # results3 = model(reader3.frame)
        # results3.render()
        # frame3 = results3.imgs[0]

        shower1.q.put(frame1)
        # shower2.frame = frame2
        # shower3.frame = frame3


def object_detection(path_list):
    reader1 = VideoReader(path_list[0]).start()
    reader2 = VideoReader(path_list[1]).start()
    reader3 = VideoReader(path_list[2]).start()

    shower1 = VideoShower("Video1").start()
    shower2 = VideoShower("Video2").start()
    shower3 = VideoShower("Video3").start()

    shape = reader1.frame.shape
    # reshape to smaller size ot make video easer to view
    output_shape = (int(shape[1] / 2), int(shape[0] / 2))

    while True:
        if (
            reader1.stopped
            or shower1.stopped
            or reader2.stopped
            or shower2.stopped
            or reader3.stopped
            or shower3.stopped
        ):
            reader1.stop()
            shower1.stop()
            reader2.stop()
            shower2.stop()
            reader3.stop()
            shower3.stop()
            break

        results1 = model(reader1.frame)
        results2 = model(reader2.frame)
        results3 = model(reader3.frame)

        results1.render()
        results2.render()
        results3.render()

        frame1 = results1.imgs[0]
        frame2 = results2.imgs[0]
        frame3 = results3.imgs[0]

        frame1 = cv2.resize(frame1, output_shape)
        frame2 = cv2.resize(frame2, output_shape)
        frame3 = cv2.resize(frame3, output_shape)

        shower1.q.put(frame1)
        shower2.q.put(frame2)
        shower3.q.put(frame3)


def topview(vids, M_list, dst_shape):
    reader1 = VideoReader(vids[0]).start()
    reader2 = VideoReader(vids[1]).start()
    reader3 = VideoReader(vids[2]).start()

    shower1 = VideoShower("Video1").start()
    shower2 = VideoShower("Video2").start()
    shower3 = VideoShower("Video3").start()

    final_shower = VideoShower("video").start()

    while True:
        if (
            reader1.stopped
            or shower1.stopped
            or reader2.stopped
            or shower2.stopped
            or reader3.stopped
            or shower3.stopped
            or final_shower.stopped
        ):
            reader1.stop()
            reader2.stop()
            reader3.stop()
            shower1.stop()
            shower2.stop()
            shower3.stop()
            final_shower.stop()
            break

        frame1 = cv2.warpPerspective(reader1.frame, M_list[0], dst_shape)
        frame2 = cv2.warpPerspective(reader2.frame, M_list[1], dst_shape)
        frame3 = cv2.warpPerspective(reader3.frame, M_list[2], dst_shape)

        temp = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        final = cv2.addWeighted(temp, 0.67, frame3, 0.33, 0)

        shower1.q.put(frame1)
        shower2.q.put(frame2)
        shower3.q.put(frame3)
        final_shower.q.put(final)


def topview_object_detection(vids, M_list, m_list, dst_shape):
    
    reader1 = VideoReader(vids[0]).start()
    reader2 = VideoReader(vids[1]).start()
    reader3 = VideoReader(vids[2]).start()

    shower = VideoShower("video").start()

    while True:
        if reader1.stopped or reader2.stopped or reader3.stopped or shower.stopped:
            reader1.stop()
            reader2.stop()
            reader3.stop()
            shower.stop()
            break

        frame1 = cv2.warpPerspective(reader1.frame, M_list[0], dst_shape)
        frame2 = cv2.warpPerspective(reader2.frame, M_list[1], dst_shape)
        frame3 = cv2.warpPerspective(reader3.frame, M_list[2], dst_shape)

        temp = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        final = cv2.addWeighted(temp, 0.67, frame3, 0.33, 0)

        results1 = model(reader1.frame)
        results2 = model(reader2.frame)
        results3 = model(reader3.frame)
        
        if save_counter < 500:
            try:
            
                cv2.imwrite(f'E:\\D drive\\Fall 2021\\Computer visions\\project\\github version\\masked_images2\\img_{save_counter}.jpg',results1.render()[0])
                cv2.imwrite(f'E:\\D drive\\Fall 2021\\Computer visions\\project\\github version\\masked_images2\\img_{save_counter+1}.jpg',results2.render()[0])
                cv2.imwrite(f'E:\\D drive\\Fall 2021\\Computer visions\\project\\github version\\masked_images2\\img_{save_counter+2}.jpg',results3.render()[0])
                
                save_counter+=3
            except:
                continue


        labels1, cord_thres1 = (
            results1.xyxyn[0][:, -1].cpu().numpy(),
            results1.xyxyn[0][:, :-1].cpu().numpy(),
        )

        labels2, cord_thres2 = (
            results2.xyxyn[0][:, -1].cpu().numpy(),
            results2.xyxyn[0][:, :-1].cpu().numpy(),
        )

        labels3, cord_thres3 = (
            results3.xyxyn[0][:, -1].cpu().numpy(),
            results3.xyxyn[0][:, :-1].cpu().numpy(),
        )

        for i, j in zip(labels1, cord_thres1):
            if i:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[0], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)

            final = cv2.circle(final, (new_coords[0], new_coords[1]), 5, color, -1)

        for i, j in zip(labels2, cord_thres2):
            if i:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[1], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)

            final = cv2.circle(final, (new_coords[0], new_coords[1]), 5, color, -1)

        for i, j in zip(labels3, cord_thres3):
            if i:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[2], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)

            final = cv2.circle(final, (new_coords[0], new_coords[1]), 7, color, -1)

        shower.q.put(final)


def SOP_violation(vids, M_list, m_list, dst_shape):
    reader1 = VideoReader(vids[0]).start()
    reader2 = VideoReader(vids[1]).start()
    reader3 = VideoReader(vids[2]).start()

    shower = VideoShower("video").start()

    while True:
        if reader1.stopped or reader2.stopped or reader3.stopped or shower.stopped:
            reader1.stop()
            reader2.stop()
            reader3.stop()
            shower.stop()
            break

        frame1 = cv2.warpPerspective(reader1.frame, M_list[0], dst_shape)
        frame2 = cv2.warpPerspective(reader2.frame, M_list[1], dst_shape)
        frame3 = cv2.warpPerspective(reader3.frame, M_list[2], dst_shape)

        temp = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        final = cv2.addWeighted(temp, 0.67, frame3, 0.33, 0)

        results1 = model(reader1.frame)
        results2 = model(reader2.frame)
        results3 = model(reader3.frame)

        labels1, cord_thres1 = (
            results1.xyxyn[0][:, -1].cpu().numpy(),
            results1.xyxyn[0][:, :-1].cpu().numpy(),
        )

        labels2, cord_thres2 = (
            results2.xyxyn[0][:, -1].cpu().numpy(),
            results2.xyxyn[0][:, :-1].cpu().numpy(),
        )

        labels3, cord_thres3 = (
            results3.xyxyn[0][:, -1].cpu().numpy(),
            results3.xyxyn[0][:, :-1].cpu().numpy(),
        )

        points = []

        for i, j in zip(labels1, cord_thres1):
            if i:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[0], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

            final = cv2.circle(final, (new_coords[0], new_coords[1]), 5, color, -1)

        for i, j in zip(labels2, cord_thres2):
            if i:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[1], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

            final = cv2.circle(final, (new_coords[0], new_coords[1]), 5, color, -1)

        for i, j in zip(labels3, cord_thres3):
            if i:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[2], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

            final = cv2.circle(final, (new_coords[0], new_coords[1]), 7, color, -1)

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1 = points[i]
                p2 = points[j]
                dist = np.linalg.norm(p1 - p2)
                if dist < util.MAX_DIST and dist > util.MIN_DIST:
                    print(p1, p2, dist)
                    final = cv2.putText(
                        final,
                        "SOP VIOLATION",
                        points[i],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    final = cv2.line(final, p1, p2, (0, 255, 255), 3)

        shower.q.put(final)


def heatmap1(vids, M_list, m_list, dst_shape):
    k = 15
    gauss = cv2.getGaussianKernel(k, np.sqrt(64))
    gauss = gauss * gauss.T
    gauss = gauss / gauss[int(k / 2), int(k / 2)]

    reader1 = VideoReader(vids[0]).start()
    reader2 = VideoReader(vids[1]).start()
    reader3 = VideoReader(vids[2]).start()

    shower = VideoShower("video").start()

    points = []
    while True:

        if reader1.stopped or reader2.stopped or reader3.stopped or shower.stopped:
            reader1.stop()
            reader2.stop()
            reader3.stop()
            shower.stop()
            break

        frame1 = cv2.warpPerspective(reader1.frame, M_list[0], dst_shape)
        frame2 = cv2.warpPerspective(reader2.frame, M_list[1], dst_shape)
        frame3 = cv2.warpPerspective(reader3.frame, M_list[2], dst_shape)

        temp = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        final = cv2.addWeighted(temp, 0.67, frame3, 0.33, 0)

        try:
            results1 = model(reader1.frame)
            results2 = model(reader2.frame)
            results3 = model(reader3.frame)
        except AttributeError:
            break

        cord_thres1 = results1.xyxyn[0][:, :-1].cpu().numpy()
        cord_thres2 = results2.xyxyn[0][:, :-1].cpu().numpy()
        cord_thres3 = results3.xyxyn[0][:, :-1].cpu().numpy()

        for j in cord_thres1:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[0], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

        for j in cord_thres2:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[1], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

        for j in cord_thres3:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[2], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

        vis = final.astype(np.float64) / 255
        imgH = np.zeros((final.shape[0], final.shape[1], 3)).astype(np.float32)

        equation = ((1 - gauss) * 255).astype(np.uint8)

        j = (
            cv2.cvtColor(
                cv2.applyColorMap(equation, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
            ).astype(np.float32)
            / 255
        )

        heatmap = None

        for p in points:
            b = imgH[
                p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                :,
            ]
            try:
                c = j + b
            except ValueError:
                continue
            imgH[
                p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                :,
            ] = c

            g = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY)
            mask = np.where(g > 0.2, 1, 0).astype(np.float32)
            mask_3 = np.ones((vis.shape[0], vis.shape[1], 3)) * (1 - mask)[:, :, None]
            mask_4 = imgH * (mask)[:, :, None]
            new_top = mask_3 * vis
            heatmap = new_top + mask_4

        # shower.frame = heatmap
        if heatmap is None:
            shower.q.put(vis)
        else:
            shower.q.put(heatmap)


def heatmap2(vids, M_list, m_list, dst_shape):
    k = 15
    timer = 0
    gauss = cv2.getGaussianKernel(k, np.sqrt(64))
    gauss = gauss * gauss.T
    gauss = gauss / gauss[int(k / 2), int(k / 2)]

    reader1 = VideoReader(vids[0]).start()
    reader2 = VideoReader(vids[1]).start()
    reader3 = VideoReader(vids[2]).start()

    shower = VideoShower("video").start()

    points = []
    count = 0
    while True:

        if count > timer:
            count -= 1
            try:
                points.pop(0)
            except IndexError:
                continue

        if reader1.stopped or reader2.stopped or reader3.stopped or shower.stopped:
            reader1.stop()
            reader2.stop()
            reader3.stop()
            shower.stop()
            break

        frame1 = cv2.warpPerspective(reader1.frame, M_list[0], dst_shape)
        frame2 = cv2.warpPerspective(reader2.frame, M_list[1], dst_shape)
        frame3 = cv2.warpPerspective(reader3.frame, M_list[2], dst_shape)

        temp = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        final = cv2.addWeighted(temp, 0.67, frame3, 0.33, 0)

        try:
            results1 = model(reader1.frame)
            results2 = model(reader2.frame)
            results3 = model(reader3.frame)
        except AttributeError:
            break

        cord_thres1 = results1.xyxyn[0][:, :-1].cpu().numpy()
        cord_thres2 = results2.xyxyn[0][:, :-1].cpu().numpy()
        cord_thres3 = results3.xyxyn[0][:, :-1].cpu().numpy()

        for j in cord_thres1:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[0], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append((new_coords[0], new_coords[1]))

        for j in cord_thres2:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[1], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append((new_coords[0], new_coords[1]))

        for j in cord_thres3:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[2], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append((new_coords[0], new_coords[1]))

        vis = final.astype(np.float64) / 255
        imgH = np.zeros((final.shape[0], final.shape[1], 3)).astype(np.float32)

        equation = ((1 - gauss) * 255).astype(np.uint8)

        j = (
            cv2.cvtColor(
                cv2.applyColorMap(equation, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
            ).astype(np.float32)
            / 255
        )

        heatmap = None

        for p in points:
            b = imgH[
                p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                :,
            ]
            try:
                c = j + b
            except ValueError:
                continue
            imgH[
                p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                :,
            ] = c

            g = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY)
            mask = np.where(g > 0.2, 1, 0).astype(np.float32)
            mask_3 = np.ones((vis.shape[0], vis.shape[1], 3)) * (1 - mask)[:, :, None]
            mask_4 = imgH * (mask)[:, :, None]
            new_top = mask_3 * vis
            heatmap = new_top + mask_4

        count += 1

        if heatmap is None:
            shower.q.put(vis)
        else:
            shower.q.put(heatmap)


def heatmap3(vids, M_list, m_list, dst_shape):
    k = 15
    timer = 0
    gauss = cv2.getGaussianKernel(k, np.sqrt(64))
    gauss = gauss * gauss.T
    gauss = gauss / gauss[int(k / 2), int(k / 2)]

    reader1 = VideoReader(vids[0]).start()
    reader2 = VideoReader(vids[1]).start()
    reader3 = VideoReader(vids[2]).start()

    shower = VideoShower("video").start()

    heatmap_points = []
    count = 0
    while True:

        if count > timer:
            count -= 1
            try:
                heatmap_points.pop(0)
            except IndexError:
                continue

        if reader1.stopped or reader2.stopped or reader3.stopped or shower.stopped:
            reader1.stop()
            reader2.stop()
            reader3.stop()
            shower.stop()
            break

        frame1 = cv2.warpPerspective(reader1.frame, M_list[0], dst_shape)
        frame2 = cv2.warpPerspective(reader2.frame, M_list[1], dst_shape)
        frame3 = cv2.warpPerspective(reader3.frame, M_list[2], dst_shape)

        temp = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        final = cv2.addWeighted(temp, 0.67, frame3, 0.33, 0)

        try:
            results1 = model(reader1.frame)
            results2 = model(reader2.frame)
            results3 = model(reader3.frame)
        except AttributeError:
            break

        cord_thres1 = results1.xyxyn[0][:, :-1].cpu().numpy()
        cord_thres2 = results2.xyxyn[0][:, :-1].cpu().numpy()
        cord_thres3 = results3.xyxyn[0][:, :-1].cpu().numpy()

        points = []

        for j in cord_thres1:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[0], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

        for j in cord_thres2:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[1], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

        for j in cord_thres3:
            coords = j[0:2]
            coords *= np.array([1920, 1080])
            coords = np.append(coords, [1])
            new_coords = np.matmul(m_list[2], coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)
            points.append(new_coords)

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1 = points[i]
                p2 = points[j]
                dist = np.linalg.norm(p1 - p2)
                if dist < util.MAX_DIST and dist > util.MIN_DIST:
                    heatmap_points.append(p1)
                    heatmap_points.append(p2)

        vis = final.astype(np.float64) / 255
        imgH = np.zeros((final.shape[0], final.shape[1], 3)).astype(np.float32)

        equation = ((1 - gauss) * 255).astype(np.uint8)

        j = (
            cv2.cvtColor(
                cv2.applyColorMap(equation, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
            ).astype(np.float32)
            / 255
        )

        heatmap = None

        for p in heatmap_points:
            b = imgH[
                p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                :,
            ]
            try:
                c = j + b
            except ValueError:
                continue
            imgH[
                p[1] - int(k / 2) : p[1] + int(k / 2) + 1,
                p[0] - int(k / 2) : p[0] + int(k / 2) + 1,
                :,
            ] = c

            g = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY)
            mask = np.where(g > 0.2, 1, 0).astype(np.float32)
            mask_3 = np.ones((vis.shape[0], vis.shape[1], 3)) * (1 - mask)[:, :, None]
            mask_4 = imgH * (mask)[:, :, None]
            new_top = mask_3 * vis
            heatmap = new_top + mask_4

        count += 1

        if heatmap is None:
            shower.q.put(vis)
        else:
            shower.q.put(heatmap)


def homography_generator(path_list, dst_path):
    dst = cv2.imread(dst_path)
    M_list = []
    temp_stream = cv2.VideoCapture(path_list[0])
    status, frame = temp_stream.read()
    temp_stream.release()
    if not status:
        print("error")
        return None
    M_list.append(homography_computation.main(frame, dst))

    temp_stream = cv2.VideoCapture(path_list[1])
    status, frame = temp_stream.read()
    temp_stream.release()

    if not status:
        print("error")
        return None
    M_list.append(homography_computation.main(frame, dst))

    temp_stream = cv2.VideoCapture(path_list[2])
    status, frame = temp_stream.read()
    temp_stream.release()

    if not status:
        print("error")
        return None
    M_list.append(homography_computation.main(frame, dst))

    return M_list, dst.shape[:2]


if __name__ == "__main__":

    if len(sys.argv) == 1:
        argument = "pre"
    else:
        argument = sys.argv[1]

    if argument == "pre":
        path_list = [util.VIDEO1_PATH, util.VIDEO2_PATH, util.VIDEO3_PATH]
        M_list = [util.A, util.B, util.C]
        m_list = [util.A2, util.B2, util.C2]
        main(path_list, M_list, m_list, util.HOMO_SIZE)
    elif argument == "save":
        save_videos()
    elif argument == "stream":
        live_stream()
    elif argument == "live":
        path_list = [util.IP1, util.IP2, util.IP3]
        # path_list = [0, 0, 0]
        M_list, dst_shape = homography_generator(path_list, util.TOP_VIEW)
        main(path_list, M_list, M_list, dst_shape)
    else:
        print("Error: Invalid Argument")
