#-*- coding: utf-8 -*-
"""
1. 단어 vocabulary만들기
2. LSTM 짜기
3. 학습
"""
import theano
import theano.tensor as T

import cPickle
import os
import ipdb
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from keras import initializations
from keras.utils.theano_utils import shared_zeros

class LSTM():
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.emb = initializations.uniforms((input_dim, hidden_dim))

        self.encode_W = initializations.uniforms((input_dim, hidden_dim*4))
        self.encode_U = initializations.uniforms((hidden_dim, hidden_dim*4))
        self.encode_b = initializations.zero((hidden_dim*4,))

        self.decode_W = initializations.uniforms((input_dim, hidden_dim*4))
        self.decode_U = initializations.uniforms((hidden_dim, hidden_dim*4))
        self.decode_V = initializations.uniforms((hidden_dim, hidden_dim*4))
        self.decode_b = initializations.zero((input_dim, hidden_dim*4))

        self.output_W = initializations.uniforms((hidden_dim, output_dim))
        self.output_b = initializations.zero((output_dim, ))

        self.params = [
            self.emb,
            self.encode_W, self.encode_U, self.encode_b,
            self.decode_W, self.decode_U, self.decode_b,
            self.output_W, self.output_b,
        ]

    def encode_lstm(self, x, mask):
        def _step(m_tm_1, x_t, h_tm_1, c_tm_1):
            lstm_preactive = T.dot(h_tm_1, self.encode_U) + \
                             T.dot(x_t, self.encode_W) + \
                             self.encode_b

            i = T.nnet.sigmoid(lstm_preactive[0:self.lstm_dim])
            f = T.nnet.sigmoid(lstm_preactive[self.lstm_dim:self.lstm_dim*2])
            o = T.nnet.sigmoid(lstm_preactive[self.lstm_dim*2:self.lstm_dim*3])
            c = T.tanh(lstm_preactive[self.lstm_dim*3:self.lstm_dim*4])

            c = f*c_tm_1 + i*c
            c = m_tm_1[:,None]*c + (1.-m_tm_1)[:,None]*c_tm_1

            h = o*T.tanh(c)
            h = m_tm_1[:,None]*h + (1.-m_tm_1)[:,None]*h_tm_1

            return [h,c]

        h0 = T.alloc(0., x.shape[1], self.lstm_dim)
        c0 = T.alloc(0., x.shape[1], self.lstm_dim)

        rval, updates = theano.scan(
                fn=_step,
                sequences=[mask,x],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def decode_lstm(self, y, mask, context):
        def _step(m_tm_1, y_tm_1, y_t, h_tm_1, c_tm_1):
            lstm_preactive = T.dot(h_tm_1, self.decode_U) + \
                             T.dot(y_tm_1, self.decode_W) + \
                             T.dot(context, self.decode_V) + \
                             self.decode_b

            i = T.nnet.sigmoid(lstm_preactive[0:self.lstm_dim])
            f = T.nnet.sigmoid(lstm_preactive[self.lstm_dim:self.lstm_dim*2])
            o = T.nnet.sigmoid(lstm_preactive[self.lstm_dim*2:self.lstm_dim*3])
            c = T.tanh(lstm_preactive[self.lstm_dim*3:self.lstm_dim*4])

            c = f*c_tm_1 + i*c
            c = m_tm_1[:,None]*c + (1.-m_tm_1)[:,None]*c_tm_1

            h = o*T.tanh(c)
            h = m_tm_1[:,None]*h + (1.-m_tm_1)[:,None]*h_tm_1

            return [h,c]

        h0 = T.alloc(0., y.shape[1], self.lstm_dim)
        c0 = T.alloc(0., y.shape[1], self.lstm_dim)

        y_in = T.zeros_like(y)
        y_in = T.set_subtensor(y_in[1:], y[:-1])
        y_out = y

        rval, updates = theano.scan(
                fn=_step,
                sequences=[mask,y_in, y_out],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def build_model(self):
        news_sequence = T.matrix('news_sequence')
        news_mask = T.matrix('news_mask')
        comment_sequence = T.matrix('comment_sequence')
        comment_mask = T.matrix('comment_mask')

        news_emb = self.emb[news_sequence]

        h_list = self.encode_lstm( news_emb, news_mask )
        context = h_list[-1]

        aaa = self.decode_lstm(comment_sequence, comment_mask, context)


# 일단 vocabulary부터
def get_vectorizer(data, n_max_words):
    vectorizer = CountVectorizer(max_features=n_max_words)

    news = data['news']
    comments = data['comments'].map(lambda x: ' '.join(x))

    all_sentences = ' '.join(news.values + comments.values)
    vectorizer.fit(all_sentences.split(' '))
    return vectorizer

def read_data(data_files):
    return pd.concat(map(lambda data_file: pd.read_pickle(data_file), data_files))

def split_dataset(data_path):
    data_files = map(lambda x: os.path.join(data_path, x), os.listdir(data_path))
    np.random.shuffle(data_files)

    n_files = len(data_files)
    trainset = read_data(data_files[:int(n_files*0.6)])
    validset = read_data(data_files[int(n_files*0.6):int(n_files*0.8)])
    testset = read_data(data_files[int(n_files*0.8):])

    trainset.to_pickle('train.pickle')
    validset.to_pickle('valid.pickle')
    testset.to_pickle('test.pickle')

def main():
    data_path = './data_processed/'
    n_max_words = 50000
    split_dataset(data_path)
    trainset = pd.read_pickle('train.pickle')

    vectorizer = get_vectorizer(trainset, n_max_words)

