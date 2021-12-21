import threading
import cv2
import util
from queue import Empty, Queue
import numpy as np
import torch
import time


class VideoReader:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.status, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self

    def stop(self):
        self.stopped = True

    def get(self):
        while not self.stopped:
            if not self.status:
                self.stop()
            else:
                self.status, self.frame = self.stream.read()


class VideoShower:
    def __init__(self, window_name, frame=None):
        self.window_name = window_name
        self.frame = frame
        self.q = Queue()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.show, args=())
        t.daemon = True
        t.start()
        return self

    def stop(self):
        self.stopped = True

    def show(self):
        while not self.stopped:
            # if self.frame is None:
            #     continue
            try:
                # time.sleep(0.01)
                frame = self.q.get(block=True, timeout=10)
                cv2.imshow(self.window_name, frame)
            except Empty as e:
                print(e)
                self.stopped = True
                break
            # cv2.imshow(self.window_name, self.frame)

            if cv2.waitKey(1) == 27:
                self.stopped = True


if __name__ == "__main__":
    reader1 = VideoReader(util.VIDEO1_PATH).start()
    reader2 = VideoReader(util.VIDEO2_PATH).start()
    reader3 = VideoReader(util.VIDEO3_PATH).start()

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        "weights.pt",
        # force_reload=True,  # uncomment out when changing weights
    )

    shower = VideoShower("video").start()

    while True:
        if reader1.stopped or reader2.stopped or reader3.stopped or shower.stopped:
            reader1.stop()
            reader2.stop()
            reader3.stop()
            shower.stop()
            break

        results1 = model(reader1.frame)
        results2 = model(reader2.frame)
        results3 = model(reader3.frame)

        frame1 = cv2.warpPerspective(reader1.frame, util.A, util.HOMO_SIZE)
        frame2 = cv2.warpPerspective(reader2.frame, util.B, util.HOMO_SIZE)
        frame3 = cv2.warpPerspective(reader3.frame, util.C, util.HOMO_SIZE)

        temp = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        final = cv2.addWeighted(temp, 0.67, frame3, 0.33, 0)

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
            new_coords = np.matmul(util.A2, coords)
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
            new_coords = np.matmul(util.B2, coords)
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
            new_coords = np.matmul(util.C2, coords)
            new_coords = (new_coords[:2] / new_coords[2]).astype(int)

            final = cv2.circle(final, (new_coords[0], new_coords[1]), 5, color, -1)

        shower.q.put(final)
