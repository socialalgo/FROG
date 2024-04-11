# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def load_s3_data(path, num_fea_size):


    src = pd.read_csv(path, header=None, usecols=[1]).values
    dst = pd.read_csv(path, header=None, usecols=[2]).values
    label = pd.read_csv(path, header=None, usecols=[3]).values
    feature = pd.read_csv(path, header=None, usecols=[i + 4 for i in range(num_fea_size)]).values


    label = np.array(label)
    feature = np.array(feature)

    """remove rows which contain INF and NAN"""
    print("INF:", np.isinf(feature).any())
    if np.isinf(feature).any():
        try:
            dst = np.delete(dst, np.where(np.isinf(feature))[0], axis=0)
            src = np.delete(src, np.where(np.isinf(feature))[0], axis=0)
            label = np.delete(label, np.where(np.isinf(feature))[0], axis=0)
            feature = np.delete(feature, np.where(np.isinf(feature))[0], axis=0)
            print("delete inf done")
        except:
            print("delete inf error")
    print("NULL:", np.isnan(feature).any())
    if np.isnan(feature).any():
        try:
            dst = np.delete(dst, np.where(np.isnan(feature))[0], axis=0)
            src = np.delete(src, np.where(np.isnan(feature))[0], axis=0)
            label = np.delete(label, np.where(np.isnan(feature))[0], axis=0)
            feature = np.delete(feature, np.where(np.isnan(feature))[0], axis=0)
            print("delete nan done")
        except:
            print("delete null error")
    return feature, label, src, dst

