import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.exposure
import cv2
import math


class Aligning:
    def __init__(self, data, images, z_values, PPI, offset_x, offset_y):
        self.images = images
        self.z_values = z_values
        self.data = data
        self.PPI = PPI
        self.offset_x = offset_x
        self.offset_y = offset_y

    def matching(self):
        x = np.array(self.data['pos_x'])
        y = np.array(self.data['pos_y'])
        z = np.array(self.data['pos_z'])
        label = np.empty(0, dtype=float)
        # print(x,x.shape)

        pixel_x = x * (self.PPI / 25.4) + self.offset_x
        pixel_y = y * (self.PPI / 25.4) + self.offset_y

        for i in range(self.images.shape[0]):
            print('Layer: ', i)
            pore_img = self.extract_object(self.images[i])
            xy_coordinate = np.flip(np.column_stack(np.where(pore_img == 1.0)), axis=1)
            # print(xy_coordinate, xy_coordinate.shape)
            threshold = int(3000 / 35)
            pore_x = []
            pore_y = []

            px = pixel_x[z == self.z_values[i]]
            py = pixel_y[z == self.z_values[i]]
            # print(px, len(px))

            label_i = np.zeros(len(px))
            if len(xy_coordinate) >= 1:
                flag = np.zeros((len(px)))

                for p in range(len(px)):
                    dx = (px[p] - xy_coordinate[:, 0]) ** 2
                    dy = (py[p] - xy_coordinate[:, 1]) ** 2

                    for d in range(len(xy_coordinate)):
                        distance = math.sqrt(dx[d] + dy[d])
                        if distance <= threshold and flag[p] == 0:
                            pore_x.append(px[p])
                            pore_y.append(py[p])
                            label_i[p] = 1
                            flag[p] = 1
            # print(label_i, len(label_i))
            # exit()
            # fig = plt.figure(figsize=(4, 6))
            # ax = fig.add_subplot(111)
            # ax.imshow(self.images[i], cmap='gray')
            # ax.scatter(px, py, c='b', s=10)
            # ax.scatter(pore_x, pore_y, c='r', s=10)
            # for idx, txt in enumerate(label_i[label_i == 1]):
            #     ax.annotate('1', (pore_x[idx], pore_y[idx]), c='k')
            # plt.show()
            # exit()

            label = np.append(label, label_i, axis=0)
        print(label, label.shape)
        self.data['Label'] = label.tolist()
        self.data['pos_x'] = pixel_x
        self.data['pos_y'] = pixel_y
        return self.data, pixel_x, pixel_y

    def extract_object(self, image):
        # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        thresh = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        contour = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0] if len(contour) == 2 else contour[1]
        # print(contour.type)
        # plt.imshow(morph)
        # plt.show()
        if contour == ():
            return morph
        else:
            big_contour = max(contour, key=cv2.contourArea)

            filled_contour = np.zeros_like(image)
            cv2.drawContours(filled_contour, [big_contour], 0, 255, -1)

            blur = cv2.GaussianBlur(filled_contour, (5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

            mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255))

            new_arr = -1 * (np.ones((morph.shape[0], morph.shape[1])))
            # print(new_arr,new_arr.shape)
            for i in range(new_arr.shape[0]):
                for j in range(new_arr.shape[1]):
                    if mask[i, j] or thresh[i, j] == 1:
                        new_arr[i, j] = thresh[i, j]
            # print(new_arr.shape)
            # plt.imshow(new_arr[1500:1600,270:380],cmap='gray')
            # plt.show()
            # exit()
            pore = np.zeros((morph.shape[0], morph.shape[1]))
            for i in range(new_arr.shape[0]):
                for j in range(new_arr.shape[1]):
                    if new_arr[i, j] == 0:
                        pore[i, j] = 1

            # print(pore.shape)
            # plt.imshow(pore,cmap='gray')
            # plt.show()
            # exit()
        return pore
