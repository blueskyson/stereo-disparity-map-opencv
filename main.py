import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Stereo Disparity Map")
        button = [None] * 2
        button[0] = QPushButton("1. Stereo Disparity Map", self)
        button[0].clicked.connect(self.button1_click)
        button[1] = QPushButton("2. Checking the Disparity Value", self)
        button[1].clicked.connect(self.button2_click)
        vbox = QVBoxLayout()
        vbox.addWidget(button[0])
        vbox.addWidget(button[1])
        vbox.addStretch(1)
        self.setLayout(vbox)
        self.prepare_data()
        self.win_name = "match"

    def prepare_data(self):
        self.imgL = cv2.imread("imL.png")
        self.imgR = cv2.imread("imR.png")
        grayL = cv2.imread("imL.png", cv2.IMREAD_GRAYSCALE)
        grayR = cv2.imread("imR.png", cv2.IMREAD_GRAYSCALE)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = stereo.compute(grayL, grayR)
        self.disp_norm = cv2.normalize(
            self.disparity,
            self.disparity,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

    def draw_match(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x = x * 4
            y = y * 4
            if x > self.disparity.shape[1]:
                return

            disp = int(self.disparity[y][x] / 16)
            if disp <= 0:
                return

            print(x, y, disp)
            imgL = self.imgL.copy()
            imgR = self.imgR.copy()
            point = (x - disp, y)
            imgR = cv2.circle(imgR, point, 10, (0, 255, 0), -1)
            horizontal = np.hstack((imgL, imgR))
            horizontal = cv2.resize(horizontal, (1398, 476))
            cv2.imshow(self.win_name, horizontal)

    def button1_click(self):
        cv2.imshow("disparity", cv2.resize(self.disp_norm, (1049, 714)))

    def button2_click(self):
        horizontal = np.hstack((self.imgL, self.imgR))
        horizontal = cv2.resize(horizontal, (1398, 476))
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.draw_match)
        cv2.imshow(self.win_name, horizontal)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
