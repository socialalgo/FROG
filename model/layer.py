# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization

class Co_attention_layer(Layer):
    def __init__(self, emb_dim=8, num_fea_size=32, mix="concat"):
        super(Co_attention_layer, self).__init__()
        self.dense_emb_layers = [Dense(emb_dim, activation=None) for _ in range(num_fea_size)]
        self.dense_layer = Dense(16, activation="relu")
        self.mix = mix
        self.emb_dim = emb_dim
    def build(self, input_shape):
        super(Co_attention_layer, self).build(input_shape)
        self.M = self.add_weight(name='w_weights', shape=[self.emb_dim, self.emb_dim],initializer='glorot_uniform',trainable=True)


    def call(self, inputs, **kwargs):
        mid = inputs.shape[1]//2
        out = inputs
        inputs = [layer(tf.reshape(inputs[:, i], shape=(-1, 1))) for i, layer in enumerate(self.dense_emb_layers)]
        inputs = tf.transpose(inputs, [1, 0, 2])

        left = inputs[:, :mid, :]
        right = inputs[:, mid:, :]
        G = tf.matmul(tf.matmul(left, self.M), tf.transpose(right, [0, 2, 1]))
        G = tf.nn.tanh(G)
        lr = tf.nn.softmax(tf.reduce_sum(G, axis=1))
        rl = tf.nn.softmax(tf.reduce_sum(G, axis=2))
        a = tf.concat([lr, rl], axis=-1)
        out = out * a
        if self.mix=="concat":
            out = self.dense_layer(out)
        elif self.mix=="sum":
            left = out[:, :mid]
            right = out[:, mid:]
            out = left + right
        return out
class Attention_embedding_layer(Layer):
    def __init__(self, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16], weight_share=True, use_coattention=True, num_fea_size=256, emb_dim=8, co_mix="concat"):
        super(Attention_embedding_layer, self).__init__()
        self.mod_dense_emb_layers1 = [Dense(emb_dim, activation=None) for _ in range(num_fea_size//2)]
        self.mod_dense_emb_layers2 = [Dense(emb_dim, activation=None) for _ in range(num_fea_size//2)]
        self.mod_dense_emb_layers = [Dense(emb_dim, activation=None) for _ in range(num_fea_size)]

        self.mod_multi_head_att1 = MultiHeadAttention(n_heads=2, head_dim=4, dropout=0.1)
        self.mod_multi_head_att2 = MultiHeadAttention(n_heads=2, head_dim=4, dropout=0.1)

        self.mod_dnn1 = DenseLayer(hidden_units, activation, dnn_dropout)
        self.mod_dnn2 = DenseLayer(hidden_units, activation, dnn_dropout)
        self.use_coattention = use_coattention
        self.weight_share = weight_share
        self.Co_attention_layer = Co_attention_layer(emb_dim=emb_dim, num_fea_size=32, mix=co_mix)
    def call(self, inputs, **kwargs):
        if self.use_coattention == True:
            mid = inputs.shape[1] // 2
            left = inputs[:, :mid]
            right = inputs[:, mid:]
            if self.weight_share == True:
                left = [layer(tf.reshape(left[:, i], shape=(-1, 1))) for i, layer in enumerate(self.mod_dense_emb_layers1)]
                right = [layer(tf.reshape(right[:, i], shape=(-1, 1))) for i, layer in enumerate(self.mod_dense_emb_layers1)]
                left = tf.transpose(left, [1, 0, 2])
                left_att_out = left
                for _ in range(3):
                    # att_out_res = tf.matmul(emb, self.W_res)
                    left_att_out_res = left_att_out
                    left_att_out = self.mod_multi_head_att1([left_att_out, left_att_out, left_att_out])
                    left_att_out = left_att_out + left_att_out_res
                    left_att_out = tf.nn.relu(left_att_out)
                right = tf.transpose(right, [1, 0, 2])
                right_att_out = right
                for _ in range(3):
                    # att_out_res = tf.matmul(emb, self.W_res)
                    right_att_out_res = right_att_out
                    right_att_out = self.mod_multi_head_att1([right_att_out, right_att_out, right_att_out])
                    right_att_out = right_att_out + right_att_out_res
                    right_att_out = tf.nn.relu(right_att_out)
                left_att_out = tf.reshape(left_att_out, [-1, left_att_out.shape[1] * left_att_out.shape[2]])
                right_att_out = tf.reshape(right_att_out, [-1, right_att_out.shape[1] * right_att_out.shape[2]])
                left = self.mod_dnn1(left_att_out)
                right = self.mod_dnn1(right_att_out)
            else:
                left = [layer(tf.reshape(left[:, i], shape=(-1, 1))) for i, layer in enumerate(self.mod_dense_emb_layers1)]
                right = [layer(tf.reshape(right[:, i], shape=(-1, 1))) for i, layer in enumerate(self.mod_dense_emb_layers2)]
                left = tf.transpose(left, [1, 0, 2])
                left_att_out = left
                for _ in range(3):
                    # att_out_res = tf.matmul(emb, self.W_res)
                    left_att_out_res = left_att_out
                    left_att_out = self.mod_multi_head_att1([left_att_out, left_att_out, left_att_out])
                    left_att_out = left_att_out + left_att_out_res
                    left_att_out = tf.nn.relu(left_att_out)
                right = tf.transpose(right, [1, 0, 2])
                right_att_out = right
                for _ in range(3):
                    # att_out_res = tf.matmul(emb, self.W_res)
                    right_att_out_res = right_att_out
                    right_att_out = self.mod_multi_head_att2([right_att_out, right_att_out, right_att_out])
                    right_att_out = right_att_out + right_att_out_res
                    right_att_out = tf.nn.relu(right_att_out)
                left_att_out = tf.reshape(left_att_out, [-1, left_att_out.shape[1] * left_att_out.shape[2]])
                right_att_out = tf.reshape(right_att_out, [-1, right_att_out.shape[1] * right_att_out.shape[2]])
                left = self.mod_dnn1(left_att_out)
                right = self.mod_dnn2(right_att_out)
            out = tf.concat([left,right], axis=-1)
            out = self.Co_attention_layer(out)
        else:
            inputs = [layer(tf.reshape(inputs[:, i], shape=(-1, 1))) for i, layer in enumerate(self.mod_dense_emb_layers)]
            att_out = tf.transpose(inputs, [1, 0, 2])
            for i in range(3):
                # att_out_res = tf.matmul(emb, self.W_res)
                att_out_res = att_out
                att_out = self.mod_multi_head_att1([att_out, att_out, att_out])
                att_out = att_out + att_out_res
                att_out = tf.nn.relu(att_out)
            att_out = tf.reshape(att_out, [-1, att_out.shape[1] * att_out.shape[2]])  # [None, 39*k]
            out =self.mod_dnn1(att_out)

        return out


class MLP_embedding_layer(Layer):
    def __init__(self, activation='relu', dnn_dropout=0.0, hidden_units=[64, 32, 16], weight_share=True, use_coattention=True, num_fea_size=256, emb_dim=8, co_mix="concat"):
        super(MLP_embedding_layer, self).__init__()
        self.mod_dnn1 = DenseLayer(hidden_units, activation, dnn_dropout)
        self.mod_dnn2 = DenseLayer(hidden_units, activation, dnn_dropout)
        self.use_coattention = use_coattention
        self.weight_share = weight_share
        self.Co_attention_layer = Co_attention_layer(emb_dim=emb_dim, num_fea_size=32, mix=co_mix)
    def call(self, inputs, **kwargs):
        if self.use_coattention == True:
            mid = inputs.shape[1]//2
            left = inputs[:, :mid]
            right = inputs[:, mid:]
            if self.weight_share == True:
                left = self.mod_dnn1(left)
                right = self.mod_dnn1(right)
            else:
                left = self.mod_dnn1(left)
                right = self.mod_dnn2(right)
            out = tf.concat([left,right], axis=-1)
            out = self.Co_attention_layer(out)
        else:
            out =self.mod_dnn1(inputs)
        return out



class DenseLayer(Layer):
    """Dense Layer
    """
    def __init__(self, hidden_units, activation='relu', dropout=0.0):
        super(DenseLayer, self).__init__()
        self.dense_layer = [Dense(i, activation=activation) for i in hidden_units]
        # self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
        return x


class DotProductAttention(Layer):
    """Dot-Production Operation for Attention
    """
    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self._dropout = dropout
        self._masking_num = -2**32 + 1

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        score = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # [None, n, n]
        score = score/int(queries.shape[-1])**0.5   # 缩放
        score = K.softmax(score)                    # SoftMax
        score = K.dropout(score, self._dropout)     # dropout
        outputs = K.batch_dot(score, values)        # [None, n, k]
        return outputs


class MultiHeadAttention(Layer):
    """Multi-Head Attention Layer
    """
    def __init__(self, n_heads=2, head_dim=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout = dropout
        self._att_layer = DotProductAttention(dropout=self._dropout)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_values')

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        if self._n_heads*self._head_dim != queries.shape[-1]:
            raise ValueError("n_head * head_dim not equal embedding dim {}".format(queries.shape[-1]))

        queries_linear = K.dot(queries, self._weights_queries)  # [None, n, k]
        keys_linear = K.dot(keys, self._weights_keys)           # [None, n, k]
        values_linear = K.dot(values, self._weights_values)     # [None, n, k]

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2),
                                        axis=0)  # [None*n_head, n, k/n_head]
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2),
                                     axis=0)        # [None*n_head, n, k/n_head]
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2),
                                       axis=0)    # [None*n_head, n, k/n_head]

        att_out = self._att_layer([queries_multi_heads, keys_multi_heads, values_multi_heads])   # [None*n_head, n, k/n_head]
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)    # [None, n, k]
        return outputs


class DotProductAttention_mask(Layer):
    """Dot-Production Operation for Attention
    """
    def __init__(self, dropout=0.0):
        super(DotProductAttention_mask, self).__init__()
        self._dropout = dropout
        self._masking_num = -2**32 + 1

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        score = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # [None, n, n]
        score = score/int(queries.shape[-1])**0.5   # 缩放
        score = tf.nn.tanh(score)
        score = tf.nn.relu(score)
        score = K.dropout(score, self._dropout)     # dropout
        outputs = K.batch_dot(score, values)        # [None, n, k]
        return outputs

