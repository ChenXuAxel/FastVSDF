# FastVSDF: An Efficient Spatiotemporal Data Fusion Method for Seamless Data Cube
# Author: Chen Xu
# Date:09/01/2024

import gdal
import numpy as np
import torch
import collections
from skimage.filters import gaussian
from skimage.measure import block_reduce
from skimage import feature
from skimage.morphology import square
from skimage.filters.rank import mean
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from guided_filter_pytorch.guided_filter import FastGuidedFilter


def read_img(path):
    src = gdal.Open(path)
    band_data = []
    for b in range(src.RasterCount):
        band_data.append(src.GetRasterBand(b + 1).ReadAsArray())
    img = np.dstack(band_data)
    return img.astype("float")


def save_img(array, path):
    driver = gdal.GetDriverByName("GTiff")
    if len(array.shape) == 2:
        dst = driver.Create(path, array.shape[1], array.shape[0], 1, 6)
        dst.GetRasterBand(1).WriteArray(array)
    else:
        n_band = array.shape[-1]
        dst = driver.Create(path, array.shape[1], array.shape[0], n_band, 6)
        for b in range(n_band):
            dst.GetRasterBand(b + 1).WriteArray(array[:, :, b])
    del dst


def rmse(a, b):
    a = a.astype("float")
    b = b.astype("float")
    band_num = a.shape[-1]
    r = []
    for band in range(band_num):
        r.append(np.sqrt(np.sum(((a[:, :, band] - b[:, :, band])) ** 2) / np.prod(a[:, :, band].shape)))
    return r


def downsample_small_size(array, factor):
    return block_reduce(array, block_size=(factor, factor, 1), func=np.mean)


def get_para_from_count(count, obj_list):
    para_list = []
    for obj in obj_list:
        para_list.append(count[obj])
    return para_list


def enlarge_size(array, factor):
    if len(array.shape) == 3:
        out = np.zeros(shape=(array.shape[0] * factor, array.shape[1] * factor, array.shape[-1])).astype(array.dtype)
    else:
        out = np.zeros(shape=(array.shape[0] * factor, array.shape[1] * factor)).astype(array.dtype)

    for x in range(array.shape[1]):
        for y in range(array.shape[0]):
            out[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor] = array[y, x]

    return out


def make_tensor(in_array):
    return torch.from_numpy(np.array([in_array.transpose(2, 0, 1)]))


def make_array(tensor):
    return tensor[0].detach().numpy().transpose([1, 2, 0])


def FastVSDF(F1_path, C1_path, C2_path, fastvsdf_path, base_cluster=5, P=20, max_value=10000, rri_k = 1):
    """
    Main function of FastVSDF
    :param F1_path: input fine image at T1 path. Tif format
    :param C1_path: input coarse image path at T1. Tif format, same size as F1_path
    :param C2_path: input coarse image path at T2 (predicting date). Tif format, same size as F1_path
    :param fastvsdf_path: output file path
    :param base_cluster: land cover types, e.g., 5
    :param P: scale factor between coarse image and fine image
    :param max_value: max value of input data, minimum is 0 as default
    :param rri_k: k in Eq.(11), 1 is recommended for Landsat and MODIS pairs
    :return: fastvsdf_path
    """
    """""""""""""""""""""""""""""""""""""""""""""
    Prepare the fusion
    """""""""""""""""""""""""""""""""""""""""""""
    # read input
    F1_img = read_img(F1_path)[:, :, :] / max_value
    C1_img = (100+read_img(C1_path))[:, :, :] / max_value
    C2_img = (100+read_img(C2_path))[:, :, :] / max_value

    # process invaild value
    C1_img[(C1_img<0) | (C2_img>1)] = 0
    C2_img[(C1_img<0) | (C2_img>1)] = 0

    # prepare for fusion
    x_f_num = F1_img.shape[1] # x size of fine image
    y_f_num = F1_img.shape[0] # y size of fine image
    x_c_num = int(x_f_num / P) # x size of coarse image
    y_c_num = int(y_f_num / P) # y size of coarse image
    band_num = F1_img.shape[-1]

    # transfer to Tensor
    C1_sr = make_tensor(C1_img)
    F1_sr = make_tensor(F1_img)

    # downsample
    F1_img_small = downsample_small_size(F1_img, P)
    C1_img_small = downsample_small_size(C1_img, P)
    C2_img_small = downsample_small_size(C2_img, P)
    delta_M_img_small = C2_img_small - C1_img_small

    # RRI
    rmse_F1_C1 = np.average(rmse(F1_img_small, C1_img_small))
    rmse_C2_C1 = np.average(rmse(C2_img_small, C1_img_small))
    RRI = rri_k * rmse_C2_C1 / rmse_F1_C1 # Eq. (11)

    # get edge pixels
    pca = PCA(n_components=1)
    x_reduction = pca.fit(F1_img_small.reshape(x_c_num * y_c_num, band_num)).transform(
        F1_img.reshape(x_f_num * y_f_num, band_num)).reshape(y_f_num, x_f_num)
    edges = feature.canny(x_reduction)
    edges_sum = np.where(mean(edges.astype("float"), square(3)) > 0, 1, 0) # pixels within the 3-pixel buffer around edges
    edges_cube_sum_2 = np.tile(edges_sum, [band_num * 2, 1, 1]).transpose(1, 2, 0) # enlarge the edge size from (x_f_num, y_f_num) to (band_num * 2, x_f_num, y_f_num)

    """""""""""""""""""""""""""""""""""""""""""""
    STEP 1 unmix with FAVC
    """""""""""""""""""""""""""""""""""""""""""""
    # fast guided temporal change
    delta_M_img_FGF_sr = FastGuidedFilter(P, 1e-7)(C1_sr, make_tensor(C2_img - C1_img), F1_sr) # Eq. (9)
    delta_M_img_GF = make_array(delta_M_img_FGF_sr)

    for band in range(delta_M_img_GF.shape[-1]):
        delta_M_img_GF[:, :, band] = delta_M_img_GF[:, :, band] - np.min(delta_M_img_GF[:, :, band])
        delta_M_img_GF[:, :, band] = delta_M_img_GF[:, :, band] / np.max(delta_M_img_GF[:, :, band])

    # fast guided temporal change + F1_img
    u = np.zeros([y_f_num, x_f_num, band_num * 2])
    u[:, :, :band_num] = delta_M_img_GF
    u[:, :, band_num:] = F1_img

    # define the clustering number
    n_clusters = max(int(round(base_cluster ** 2 * (1-1/RRI))), base_cluster) # Eq. (12)

    # select feature pixels randomly
    random_num = int(n_clusters * 100) # Eq. (10)
    edge_value = u[edges_cube_sum_2 == 1].reshape(int(np.sum(edges_cube_sum_2[:, :, 0])), band_num * 2)
    classifer = KMeans(n_clusters=n_clusters)

    # K-Means clustering for FAVC
    classifer.fit(edge_value[np.random.choice(edge_value.shape[0], random_num, replace=False), :])
    classes_img = classifer.predict(u.reshape(y_f_num * x_f_num, band_num * 2)).reshape([y_f_num, x_f_num])
    label_list = np.unique(classes_img.flatten())

    # unmix
    pre_delta_img = np.zeros(shape=F1_img.shape)
    para_list = []
    y_list = []
    for i in range(x_c_num):
        for j in range(y_c_num):
            para = get_para_from_count(collections.Counter(classes_img[j * P:(j + 1) * P, i * P:(i + 1) * P].flatten()),
                                       label_list)
            para_list.append(para)
            y_list.append(delta_M_img_small[j, i, :])

    para_mat = np.mat(para_list)
    y_mat = np.mat(y_list)

    for b in range(band_num):
        b_para = np.linalg.lstsq(para_mat, y_mat[:, b] * P * P, rcond=None)[0]
        for i in range(label_list.shape[0]):
            pre_delta_img[:, :, b][classes_img == label_list[i]] = b_para[i][0, 0]

    """""""""""""""""""""""""""""""""""""""""""""
    STEP 2 distribute global residuals
    """""""""""""""""""""""""""""""""""""""""""""
    difference_small = delta_M_img_small - downsample_small_size(pre_delta_img, P)
    difference = enlarge_size(difference_small, P)
    difference_FGF_sr = FastGuidedFilter(int(P / 2), 1e-3)(C1_sr, make_tensor(difference), F1_sr)
    difference_new = make_array(difference_FGF_sr)
    pre_delta_img = difference_new + pre_delta_img # Eq. (14)

    """""""""""""""""""""""""""""""""""""""""""""
    STEP 3 distribute local residuals
    """""""""""""""""""""""""""""""""""""""""""""
    # fuzzy classification
    random_num = int(base_cluster * 100) # base_cluster number is used here instead of FAVC cluster number
    classifer = KMeans(n_clusters=base_cluster)
    classifer.fit(
        F1_img[edges_cube_sum_2[:, :, :band_num] == 1].reshape(int(np.sum(edges_cube_sum_2[:, :, 0])), band_num)[
            np.random.choice(np.sum(edges_cube_sum_2[:, :, 0]), random_num, replace=False)])
    classes_img = classifer.predict(F1_img.reshape(y_f_num * x_f_num, band_num)).reshape(y_f_num, x_f_num)
    label_list = np.unique(classes_img.flatten())

    # in-class Gaussian weight function Eq. (15)&(16)
    for label in label_list:
        fix_positon = (classes_img == label) # filter pixels from the same classification
        fix_array = np.where(fix_positon, 1, np.nan)
        fix_array[np.isnan(fix_array)] = 0
        fix_gaussian = gaussian(fix_array, sigma=3.75) # smaller sigma get higher efficiency
        for band in range(band_num):
            class_array = np.where(fix_positon, pre_delta_img[:, :, band], 0)
            class_gaussian = gaussian(class_array, sigma=3.75)
            fixed_class_gaussian = class_gaussian / fix_gaussian
            pre_delta_img[:, :, band] = np.where(fix_positon,
                                                 fixed_class_gaussian,
                                                 pre_delta_img[:, :, band])


    """""""""""""""""""""""""""""""""""""""""""""
    output fusion
    """""""""""""""""""""""""""""""""""""""""""""
    pre_F2_img = pre_delta_img + F1_img
    pre_F2_img = np.where((pre_F2_img> 1) | (pre_F2_img < 0),
                                    0,
                                    pre_F2_img)

    save_img(pre_F2_img * max_value, fastvsdf_path)
    return fastvsdf_path
