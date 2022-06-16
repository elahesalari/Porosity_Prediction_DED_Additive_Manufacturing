import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt
import cv2
import imutils
import random
from nltk import flatten


def get_neighborhood(data, features, y1, y2, image, dictionary):
    feature_in_layer = []
    for name in features:
        globals()[name] = np.array(data[name])
        feature_in_layer.append(name + '_layer')

    value_z = np.unique(pos_z)
    len_layer = len(value_z)
    min_x, max_x, avg_x = np.min(pos_x), np.max(pos_x), np.average(pos_x)
    min_y = np.min(pos_y)

    for i in range(len_layer):
        print('Layer:', i)
        # All feature extracted from i_th layer
        # ...i...
        for name in feature_in_layer:
            globals()[name] = globals()[name[:-6]][pos_z == value_z[i]]

        indecies_points_on_object = []
        for idx, (x_val, y_val) in enumerate(zip(pos_x_layer, pos_y_layer)):
            if min_x < x_val < avg_x and min_y < y_val < y1:
                indecies_points_on_object.append(idx)
            elif avg_x < x_val < max_x and min_y < y_val < y2:
                indecies_points_on_object.append(idx)
        # print(len(indecies_points_on_object))

        for name in feature_in_layer:
            globals()[name] = globals()[name][indecies_points_on_object]
            dictionary[name[:-6]].append(globals()[name].tolist())

        pore_x = pos_x_layer[Label_layer == 1]
        pore_y = pos_y_layer[Label_layer == 1]

        plt.imshow(image[i], cmap='gray')
        plt.scatter(pos_x_layer, pos_y_layer, s=4, c='b')
        plt.scatter(pore_x, pore_y, s=4, c='r')
        plt.show()
    return dictionary


def preprocess(flag):
    if flag == 'Sample 4':
        path_img = 'New images/Sample 4'
        data = pd.read_csv('New Datasets - Pixel/Pyrometer Data 4.csv')
        angle = 10
        y1 = 750
        y2 = 1250

    elif flag == 'Sample 5':
        path_img = 'New images/Sample 5'
        data = pd.read_csv('New Datasets - Pixel/Pyrometer Data 5.csv')
        angle = 100
        y1 = 700
        y2 = 1150

    elif flag == 'Sample 6':
        path_img = 'New images/Sample 6'
        data = pd.read_csv('New Datasets - Pixel/Pyrometer Data 6.csv')
        angle = -85
        y1 = 700
        y2 = 1000

    elif flag == 'Sample 8':
        path_img = 'New images/Sample 8'
        data = pd.read_csv('New Datasets - Pixel/Pyrometer Data 8.csv')

        angle = 105
        y1 = 650
        y2 = 1100

    images = []
    for filename in os.listdir(path_img):
        # print(filename)
        img = cv2.imread(os.path.join(path_img, filename), cv2.IMREAD_GRAYSCALE)
        rot = imutils.rotate_bound(img, angle=angle)
        images.append(rot)

    head_name = list(data.head())
    # print(head_name)
    dictionary = {}
    for h in head_name:
        dictionary[h] = []

    dic = get_neighborhood(data, head_name, y1, y2, images, dictionary)

    for h in head_name:
        dic[h] = list(flatten(dic[h]))

    return dic


flag = 'Sample 4'
dic = preprocess(flag)

df = pd.DataFrame.from_dict(dic)
print(df)
df.to_csv (r'DataPoints On Object/DataPoints Sample 4.csv', index = False, header=True)
