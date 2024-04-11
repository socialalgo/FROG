# -*- coding: utf-8 -*-

import itertools
import tensorflow as tf

from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


from model.layer import DenseLayer, MultiHeadAttention
from model.layer import Attention_embedding_layer, MLP_embedding_layer

from utils.utils import concat_func, reduce_sum, reduce_mean, generate_pairs
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D


class FROG(Model):

    def __init__(self, mod_split_list, imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat"):
        super(FROG, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units,weight_share=True,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=True,
                                                         use_coattention=True, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=True, use_coattention=True,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.linear = DenseLayer([32, 16], activation, dnn_dropout)
        self.out_layer1 = Dense(16, activation="relu")
        self.out_layer2 = Dense(1, activation=None)
        self.decision_vec = self.add_weight(name='decision_vec',
                                      shape=[16],
                                      initializer=tf.keras.initializers.RandomUniform,
                                      trainable=True)

    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))


        l_input = tf.transpose(mod_mapping, [1, 0, 2])
        l_input = tf.reshape(l_input, shape=(-1, l_input.shape[1] * l_input.shape[2]))
        l_output = self.linear(l_input)

        decision_vec_project = []
        for mod_emb in mod_mapping:
            decision_vec_project.append(K.sum(mod_emb * self.decision_vec, axis=-1))
        level_2_matrix = tf.transpose(decision_vec_project, [1, 0])
        fm_out = K.sum(level_2_matrix, axis=-1)
        fm_out = tf.expand_dims(fm_out, axis=1) * self.decision_vec
        output = self.out_layer1(K.concatenate([l_output,fm_out],axis=-1))
        output = self.out_layer2(output)
        return tf.nn.sigmoid(output)



class AutoFIS(Model):

    def __init__(self, mod_split_list, imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat", mask=[1 for _ in range(64)]):
        super(AutoFIS, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units,weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False, use_coattention=False,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.linear = DenseLayer([32, 16], activation, dnn_dropout)
        self.out_layer = Dense(1, activation=None)
        self.cols, self.rows = generate_pairs(range(16*len(mod_split_list)))
        self.mask = mask
    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))

        l_input = tf.transpose(mod_mapping, [1, 0, 2])
        l_input = tf.reshape(l_input, shape=(-1, l_input.shape[1] * l_input.shape[2]))
        l_output = self.linear(l_input)
        l_output = self.out_layer(l_output)
        # print(l_output)
        left = tf.gather(l_input, self.rows, axis=-1)
        right = tf.gather(l_input, self.cols, axis=-1)
        level_2_matrix = tf.multiply(left, right)

        level_2_matrix *= self.mask
        # fm_out = tf.reduce_sum(level_2_matrix, axis=-1)
        fm_out = K.sum(level_2_matrix, axis=-1)
        fm_out = tf.expand_dims(fm_out, axis=1)
        return tf.nn.sigmoid(l_output + fm_out)




class AutoFIS_search(Model):

    def __init__(self, mod_split_list, imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat",weight_base=0.6):
        super(AutoFIS_search, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units,weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False, use_coattention=False,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.linear = DenseLayer([32, 16], activation, dnn_dropout)
        self.out_layer = Dense(1, activation=None)
        self.cols, self.rows = generate_pairs(range(16*len(mod_split_list)))
        self.edge_weights = self.add_weight(name='weights',
                                      shape=[len(self.cols)],
                                      initializer=tf.keras.initializers.RandomUniform(
                                          minval=weight_base-0.001, maxval=weight_base+0.001),
                                      trainable=True)
    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))

        l_input = tf.transpose(mod_mapping, [1, 0, 2])
        l_input = tf.reshape(l_input, shape=(-1, l_input.shape[1] * l_input.shape[2]))
        l_output = self.linear(l_input)
        l_output = self.out_layer(l_output)
        # print(l_output)
        left = tf.gather(l_input, self.rows, axis=-1)
        right = tf.gather(l_input, self.cols, axis=-1)
        level_2_matrix = tf.multiply(left, right)

        mask = tf.expand_dims(self.edge_weights, axis=0)
        level_2_matrix *= mask
        # fm_out = tf.reduce_sum(level_2_matrix, axis=-1)
        fm_out = K.sum(level_2_matrix, axis=-1)
        fm_out = tf.expand_dims(fm_out, axis=1)
        return tf.nn.sigmoid(fm_out + l_output)



class AutoInt(Model):

    def __init__(self, mod_split_list,  imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat"):
        super(AutoInt, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False, use_coattention=False,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.linear = DenseLayer([32, 16], activation, dnn_dropout)
        self.out_layer = Dense(1, activation=None)
        self.dense_emb_layers = [Dense(embed_dim, activation=None) for _ in range(16*len(mod_split_list))]
        self.multi_head_att = MultiHeadAttention(n_heads=2, head_dim=4, dropout=0.1)
    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))

        input = tf.transpose(mod_mapping, [1, 0, 2])
        input = tf.reshape(input, shape=(-1, input.shape[1] * input.shape[2]))

        emb = [layer(tf.reshape(input[:, i], shape=(-1, 1))) for i, layer in enumerate(self.dense_emb_layers)]
        emb = tf.transpose(emb, [1, 0, 2])

        # DNN
        dnn_input = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))
        dnn_out = self.linear(dnn_input)

        # AutoInt
        att_out = emb
        att_out_res = att_out
        att_out = self.multi_head_att([att_out, att_out, att_out])
        att_out = att_out + att_out_res
        att_out = tf.nn.relu(att_out)
        att_out = tf.reshape(att_out, [-1, att_out.shape[1] * att_out.shape[2]])  # [None, 39*k]

        # output
        x = tf.concat([dnn_out, att_out], axis=-1)
        x = self.out_layer(x)

        return tf.nn.sigmoid(x)

class DMF(Model):

    def __init__(self, mod_split_list,  imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat"):
        super(DMF, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False, use_coattention=False,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.out_layer = Dense(1, activation=None)
        self.dense_layer_1st = [Dense(8, activation=activation) for _ in range(len(mod_split_list))]
        self.dense_layer_2st = [Dense(4, activation=activation) for _ in range(len(mod_split_list))]
        self.dense_layer_s1 = Dense(4, activation=activation)

    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))

        h1 = []
        for i in range(len(mod_mapping)):
            h1.append(self.dense_layer_1st[i](mod_mapping[i]))
        s1 = sum(h1)
        h2 = []
        for i in range(len(self.mod_split_list)):
            h2.append(self.dense_layer_2st[i](h1[i]))
        h2.append(self.dense_layer_s1(s1))
        s2 = sum(h2)
        dnn_out = self.out_layer(s2)
        return tf.nn.sigmoid(dnn_out)



class FM_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FM_layer,self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, self.k)
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)

        output = linear_part + inter_part
        return output


class DeepFM(Model):

    def __init__(self, mod_split_list,  imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat"):
        super(DeepFM, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False, use_coattention=False,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.linear = DenseLayer([32, 16], activation, dnn_dropout)
        self.out_layer = Dense(1, activation=None)
        self.fm = FM_layer(k=8, w_reg=1e-4, v_reg=1e-4)

    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))
        l_input = tf.transpose(mod_mapping, [1, 0, 2])
        l_input = tf.reshape(l_input, shape=(-1, l_input.shape[1] * l_input.shape[2]))
        l_output = self.linear(l_input)
        l_output = self.out_layer(l_output)
        fm_output = self.fm(l_input)
        return tf.nn.sigmoid(l_output+fm_output)


class FM(Model):

    def __init__(self, mod_split_list,  imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat"):
        super(FM, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False, use_coattention=False,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.fm = FM_layer(k=8, w_reg=1e-4, v_reg=1e-4)

    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))
        l_input = tf.transpose(mod_mapping, [1, 0, 2])
        l_input = tf.reshape(l_input, shape=(-1, l_input.shape[1] * l_input.shape[2]))
        output = self.fm(l_input)
        return tf.nn.sigmoid(output)


class LR(Model):

    def __init__(self, mod_split_list,  imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat"):
        super(LR, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False, use_coattention=False,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.out_layer = Dense(1, activation=None)

    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))
        l_input = tf.transpose(mod_mapping, [1, 0, 2])
        l_input = tf.reshape(l_input, shape=(-1, l_input.shape[1] * l_input.shape[2]))
        l_output = self.out_layer(l_input)
        return tf.nn.sigmoid(l_output)



class MLP(Model):

    def __init__(self, mod_split_list,  imp_mod_num=2, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16],
                embed_dim=8, co_mix="concat"):
        super(MLP, self).__init__()
        self.mod_split_list = mod_split_list
        self.imp_mod_num = imp_mod_num
        self.hidden_units = hidden_units
        self.fea_num_list = [(mod_split_list[i][1] - mod_split_list[i][0] + 1) for i in range(len(mod_split_list))]
        self.mod_emb_iteract = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[0], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_profile = Attention_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False,
                                                         use_coattention=False, num_fea_size=self.fea_num_list[1], emb_dim=embed_dim, co_mix=co_mix)
        self.mod_emb_oth = [MLP_embedding_layer(activation=activation, dnn_dropout=dnn_dropout, hidden_units=hidden_units, weight_share=False, use_coattention=False,
                                                num_fea_size=self.fea_num_list[i+2], emb_dim=embed_dim, co_mix=co_mix) for i in range(len(mod_split_list) - 2)]
        self.linear = DenseLayer([32, 16], activation, dnn_dropout)
        self.out_layer = Dense(1, activation=None)

    def call(self, mod_fea, training=None):
        mod_mapping = []
        mod_mapping.append(self.mod_emb_iteract(mod_fea[:,:self.fea_num_list[0]]))
        mod_mapping.append(self.mod_emb_profile(mod_fea[:, self.mod_split_list[1][0]-1:self.mod_split_list[1][1]]))
        for i in range(len(self.mod_split_list) -2):
            mod_mapping.append(self.mod_emb_oth[i](mod_fea[:, self.mod_split_list[i+2][0]-1:self.mod_split_list[i+2][1]]))

        l_input = tf.transpose(mod_mapping, [1, 0, 2])
        l_input = tf.reshape(l_input, shape=(-1, l_input.shape[1] * l_input.shape[2]))
        l_output = self.linear(l_input)
        l_output = self.out_layer(l_output)
        return tf.nn.sigmoid(l_output)




class FM_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, self.k)
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)

        output = linear_part + inter_part
        return tf.nn.sigmoid(output)