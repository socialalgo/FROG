#!/usr/bin/env python3#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.keras import losses, optimizers
from sklearn.metrics import accuracy_score
from utils.metrics import AverageMeter
from utils.metrics import AUC
from utils.loss import binary_focal_loss
from utils.loss import weighted_bce_loss
# from utils.utils import split_train_test_by_id
from utils.train_parse_args import parse_args
from utils.dataloader import load_s3_data
from model.model import  LR, MLP, FM, AutoInt, DeepFM,  DMF, AutoFIS, FROG, AutoFIS_search
# from tensorflow.keras.models import load_model
from utils.grda_tensorflow import GRDA



def train(args):
    batch_size = args.batch_size
    epochs = args.num_epochs
    lr = args.eta
    model_path = args.model_path
    lr_decay_steps = args.lr_decay_steps
    lr_decay_rate = args.lr_decay_rate


    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    num_fea_size = args.num_fea_size
    feature_train, label_train, src_train, dst_train = load_s3_data(train_path,num_fea_size)
    feature_val, label_val, src_val, dst_val = load_s3_data(val_path,num_fea_size)
    feature_test, label_test, src_test, dst_test = load_s3_data(test_path,num_fea_size)

    print("data_loading done!")


    # 2. scale feature
    train_max_value = []
    train_min_value = []
    for i in range(feature_train.shape[1]):
        train_max_value.append(max(feature_train[:, i]))
        train_min_value.append(min(feature_train[:, i]))
    train_max_value = np.array(train_max_value).reshape(1, -1)
    train_min_value = np.array(train_min_value).reshape(1, -1)
    scaler = np.r_[train_max_value, train_min_value]
    scaled_feature_train = (feature_train - train_min_value) / (train_max_value - train_min_value + 1e-7)
    scaled_feature_val = (feature_val - train_min_value) / (train_max_value - train_min_value + 1e-7)
    scaled_feature_test = (feature_test - train_min_value) / (train_max_value - train_min_value + 1e-7)

    scaled_feature_val[scaled_feature_val >= 1] = 0.99999
    scaled_feature_val[scaled_feature_val < 0] = 0
    scaled_feature_test[scaled_feature_test >= 1] = 0.99999
    scaled_feature_test[scaled_feature_test < 0] = 0
    print("data scaling done!")
    # np.savetxt(args.data_input+"scaler.csv", scaler, delimiter=',')
    # print("scaler saved!")

    # 3.build dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((scaled_feature_train, label_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((scaled_feature_val, label_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((scaled_feature_test, label_test))

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    mod_split_list = args.mod_split_str

    try:
        mod_split_str =args.mod_split_str.split(",")
        mod_split_list = []
        for mod_str in mod_split_str:
            str_tmp = mod_str.split("-")
            for i in range(len(str_tmp)):
                str_tmp[i] = int(str_tmp[i])
            mod_split_list.append(str_tmp)
    except:
        print("please input right mod_split_str")

    # 4.configure model
    if args.model_version == "AutoInt":
        model = AutoInt(mod_split_list=mod_split_list)
    elif args.model_version == "DeepFM":
        model = DeepFM(mod_split_list=mod_split_list)
    elif args.model_version == "LR":
        model = LR(mod_split_list=mod_split_list)
    elif args.model_version == "MLP":
        model = MLP(mod_split_list=mod_split_list)
    elif args.model_version == "FM":
        model = FM(mod_split_list=mod_split_list)
    elif args.model_version == "FROG":
        model = FROG(mod_split_list=mod_split_list)
    elif args.model_version == "DMF":
        model = DMF(mod_split_list=mod_split_list)
    elif args.model_version == "AutoFIS_search":
        model = AutoFIS_search(mod_split_list=mod_split_list)
    elif args.model_version == "AutoFIS":
        AutoFIS_mask = []
        for mask_label in args.AutoFIS_mask:
            AutoFIS_mask.append(int(mask_label))
        model = AutoFIS(mod_split_list=mod_split_list, mask=AutoFIS_mask)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=lr_decay_steps * len(label_train) // batch_size, decay_rate=lr_decay_rate)
    optimizer = optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=True)
    optimizer0 = GRDA(learning_rate=args.grda_lr, c=0.05, mu=0.51)


    # 5.train and validate
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    auc_meter = AverageMeter()
    val_auc_meter = AverageMeter()

    best_auc = 0
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, epochs + 1):
        tic = time.time()
        loss_meter.reset()
        acc_meter.reset()
        auc_meter.reset()
        val_loss_meter.reset()
        val_acc_meter.reset()
        val_auc_meter.reset()

        # training process
        loss_tmp = []
        for n_iter, data_batch in enumerate(train_dataset):
            x_train_batch, y_train_batch = data_batch[0], data_batch[1]
            with tf.GradientTape() as tape:
                predictions = model(x_train_batch, training=True)
                if args.model_version == "AutoFIS_search":
                    loss = tf.reduce_mean(losses.binary_crossentropy(y_train_batch, predictions))
                    grad = tape.gradient(loss, model.variables)
                    optimizer.apply_gradients(zip(grad[:-1], model.variables[:-1]))
                    optimizer0.apply_gradients(zip([grad[-1]], [model.variables[-1]]))
                else:
                    loss = tf.reduce_mean(binary_focal_loss(alpha=0.25, gamma=2)(y_train_batch, predictions))
                    grad = tape.gradient(loss, model.variables)
                    optimizer.apply_gradients(zip(grad, model.variables))


            pre = [1 if x > 0.5 else 0 for x in predictions]
            acc = accuracy_score(y_train_batch, pre)
            auc = AUC(y_train_batch, predictions)
            loss_meter.update(loss, y_train_batch.shape[0])
            acc_meter.update(acc, 1)
            auc_meter.update(auc, 1)
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.5f}, Acc: {:.5f}, AUC:{:.5f}"
                  .format(epoch, (n_iter + 1), len(train_dataset), loss_meter.avg, acc_meter.avg, auc_meter.avg))
            loss_tmp.append(loss_meter.avg)
        train_loss_list.append(np.mean(loss_tmp))
        toc = time.time()
        train_time = toc - tic
        print("training_time for one epoch:",train_time)

        # validation process
        for n_iter, data_batch in enumerate(val_dataset):
            x_val_batch, y_val_batch = data_batch[0], data_batch[1]
            predictions = model(x_val_batch, training=False)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_val_batch, predictions))

            pre = [1 if x > 0.5 else 0 for x in predictions]
            acc = accuracy_score(y_val_batch, pre)
            auc = AUC(y_val_batch, predictions)
            val_loss_meter.update(loss, y_val_batch.shape[0])
            val_acc_meter.update(acc, 1)
            val_auc_meter.update(auc, 1)
        print("val: Epoch[{}] Loss: {:.5f}, Acc: {:.5f},AUC:{:.5f}"
              .format(epoch, val_loss_meter.avg, val_acc_meter.avg, val_auc_meter.avg))

        val_loss_list.append(val_loss_meter.avg)
        if val_auc_meter.avg > best_auc:
            if args.model_version == "AutoFIS_search":
                fis_mask = np.array(model.variables[-1].numpy())
                # np.savetxt("autofis/fis_mask.csv", fis_mask, delimiter=',')
                print("mask:", model.variables[-1])
            best_auc = val_auc_meter.avg
            # checkpoint_path = args.bestmodel_name
            # model.save(checkpoint_path, save_format="tf")
            # checkpoint_path = "best_weights.h5"
            # model.save_weights(checkpoint_path)

    model.load_weights("best_weights.h5")
    pred = []
    label = []
    for n_iter, data_batch in enumerate(test_dataset):
        x_test_batch, y_test_batch = data_batch[0], data_batch[1]
        predictions = model(x_test_batch, training=False)
        pred.extend([x for x in predictions])
        label.extend([x for x in y_test_batch])
    print("test_pre finish!")
    # data_matrix = np.c_[pred,label]
    print('test_auc:',AUC(label, pred))

    print("train_loss:",np.array(train_loss_list))
    print("val_loss:",np.array(val_loss_list))


if __name__ == '__main__':
    args = parse_args()
    tf.config.experimental_run_functions_eagerly(True)

    args.model_version = "FROG"
    args.train_path = 'data/train.csv'
    args.val_path = 'data/val.csv'
    args.test_path = 'data/test.csv'
    args.num_epochs = 10
    args.num_fea_size = 512
    args.mod_split_str = '1-128,129-384,385-512'   # 1-128 test(interact)_emb; 129-384 profile_emb; 385-512 img_emb
    args.imp_mod_num = 2
    args.batch_size = 1024
    train(args)

