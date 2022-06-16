import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import cv2
import glob
import os
import imutils
from CreateNewCTimages import CreatCT
from Alignment import Aligning
from PlotData import Plot


def read_data(flag):
    if flag == 'sample4':
        mat = sio.loadmat('For Shima and Elahe/fin_p550_2.mat')  # load mat-file
        mdata = mat['fin_p550_2']  # variable in mat file
        ndata = {n: mdata[n][0, 0] for n in mdata.dtype.names}
        Columns = [n for n, v in ndata.items() if v.size == 8199 or v.size == 8198]
        d = dict((c, list(ndata[c][:, 0])) for c in Columns)
        del d['distcum']
        d['avg_temp'].append(np.nan)

        path_img = glob.glob('D:/aaa Master/3D printer/code/CT images/Sera Sample 4 Aligned/*.tiff')
        end = 700
        path_new_img = 'New images/Sample 4'
        angle = 10

        PPI = 730
        offset_x = 165
        offset_y = 286

    elif flag == 'sample5':
        mat = sio.loadmat('For Shima and Elahe/fin_p550_3.mat')  # load mat-file
        mdata = mat['fin_p550_3']  # variable in mat file
        ndata = {n: mdata[n][0, 0] for n in mdata.dtype.names}
        Columns = [n for n, v in ndata.items() if v.size == 8592 or v.size == 8591]
        d = dict((c, list(ndata[c][:, 0])) for c in Columns)
        del d['distcum']
        d['avg_temp'].append(np.nan)

        path_img = glob.glob('D:/aaa Master/3D printer/code/CT images/Sera Sample 5 Aligned/*.tiff')
        end = 290
        path_new_img = 'New images/Sample 5'
        angle = 100

        PPI = 725
        offset_x = 30
        offset_y = 185


    elif flag == 'sample6':
        mat = sio.loadmat('For Shima and Elahe/fin_p450_1.mat')  # load mat-file
        mdata = mat['fin_p450_1']  # variable in mat file
        ndata = {n: mdata[n][0, 0] for n in mdata.dtype.names}
        Columns = [n for n, v in ndata.items() if v.size == 7004]
        d = dict((c, list(ndata[c][:, 0])) for c in Columns)

        path_img = glob.glob('D:/aaa Master/3D printer/code/CT images/Sera Sample 6 Aligned/*.tiff')
        end = 650
        path_new_img = 'New images/Sample 6'
        angle = -85

        PPI = 725
        offset_x = 90
        offset_y = 140


    elif flag == 'sample8':
        mat = sio.loadmat('For Shima and Elahe/fin_p450_3.mat')  # load mat-file
        mdata = mat['fin_p450_3']  # variable in mat file
        ndata = {n: mdata[n][0, 0] for n in mdata.dtype.names}
        Columns = [n for n, v in ndata.items() if v.size == 7004]
        d = dict((c, list(ndata[c][:, 0])) for c in Columns)

        path_img = glob.glob('D:/aaa Master/3D printer/code/CT images/Sera Sample 8 Aligned/*.tiff')
        end = 670
        path_new_img = 'New images/Sample 8'
        angle = 105

        PPI = 725
        offset_x = 150
        offset_y = 150

    # Remove int values from DataFrame
    df = pd.DataFrame.from_dict(d)
    m = df.loc[df['velo'] != np.inf, 'velo'].max()
    df['velo'].replace(np.inf,m,inplace=True)

    # Remove Null values from DataFrame
    df['velo'].fillna(value=df['velo'].mean(), inplace=True)
    df['avg_temp'].fillna(value=df['avg_temp'].mean(), inplace=True)

    # Remove rows that just one z value
    unique = np.unique(df['pos_z'], return_counts=True)
    rem_z = unique[0][[i == 1 for i in unique[1]]]
    # print(unique)
    # print(rem_z)
    # exit()
    for pz in rem_z:
        df = df.drop(df[df.pos_z == pz].index)

    z_values = np.unique(df['pos_z'])

    return df, path_img, end, path_new_img , z_values, angle, PPI, offset_x, offset_y


def read_image(path_new_img,angle):
    images = []
    for filename in os.listdir(path_new_img):
        # print(filename)
        img = cv2.imread(os.path.join(path_new_img, filename), cv2.IMREAD_GRAYSCALE)
        rot = imutils.rotate_bound(img, angle=angle)
        images.append(rot)
    return np.array(images)


if __name__=='__main__':
    flag = 'sample8'
    data, path_img, end, path_new_img, z_values, angle, PPI, offset_x, offset_y = read_data(flag)

    # Create new CT images
    # new_ct = CreatCT()
    # new_ct.ct_image(path_img, end, path_new_img, z_values)

    # Read new CT images
    images = read_image(path_new_img,angle)

    # Alignment CT images with pyrometer data and get label
    alg = Aligning(data,images,z_values,PPI, offset_x, offset_y)
    data , pixel_x, pixel_y = alg.matching()
    print(data)
    # plt.imshow(images[0])
    # plt.scatter(data['pos_x'], data['pos_y'])
    # plt.show()
    data.to_csv('New Datasets - Pixel/Pyrometer Data 8.csv', index=False)











