import cv2
import scipy.io as sio
import numpy as np
import glob
import statistics


class CreatCT:

    def ct_image(self, path_img, end, path_new_img, values_z):
        count = len(values_z)
        # number of read image from last to first. n_read = [distance last, ... , distance first]
        n_read = []
        for j in range(count - 1, 0, -1):
            dis = values_z[j] - values_z[j - 1]
            n_read.append(np.round((dis * 1000) / 35))
        last = statistics.mode(n_read)

        n_read.append(last)
        print('number of read:', n_read, len(n_read))
        print('count:', count)

        # Read image from last index to count of z and then calculate min of 7(e.g) image(in row and column).
        idx = end
        new_img_array = []
        for i in range(count):
            distance = int(n_read[i])  # last to first
            print(idx)
            imgs_dis = []
            for j in range(distance):
                # print(idx)
                img = cv2.imread(path_img[idx], cv2.IMREAD_GRAYSCALE)
                imgs_dis.append(img)
                idx = idx - 1

            imgs_dis = np.array(imgs_dis)
            new_img = np.zeros((img.shape[0], img.shape[1]))
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    new_img[row, col] = np.min(imgs_dis[:, row, col])

            # print(new_img,new_img.shape)
            new_img_array.append(new_img)

            # plt.imshow(new_img,cmap='gray')
            # plt.show()

            # exit()
        new_img_array = np.array(new_img_array)
        np.save('new image.npy', new_img_array)
        print(new_img_array.shape)

        # Save the new images as png files.
        for c in range(count - 1, -1, -1):
            cv2.imwrite(f'{path_new_img}/Aligned {(count - c):02}.png', new_img_array[c])
