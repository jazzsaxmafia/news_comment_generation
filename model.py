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
from keras.preprocessing import sequence, text
from keras.utils.theano_utils import shared_zeros

class Comment_Generator():
    def __init__(self, n_words, embedding_dim,  hidden_dim):
        self.n_words = n_words
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.emb = initializations.uniform((n_words, embedding_dim))

        self.encode_W = initializations.uniform((embedding_dim, hidden_dim*4)) # input -> hidden
        self.encode_U = initializations.uniform((hidden_dim, hidden_dim*4)) # last hidden -> hidden (recurrent)
        self.encode_b = initializations.zero((hidden_dim*4,))

        self.decode_W = initializations.uniform((embedding_dim, hidden_dim*4)) # last word -> hidden
        self.decode_U = initializations.uniform((hidden_dim, hidden_dim*4)) # last hidden -> hidden
        self.decode_V = initializations.uniform((hidden_dim, hidden_dim*4)) # context -> hidden
        self.decode_b = initializations.zero((hidden_dim*4))

        self.output_W = initializations.uniform((hidden_dim, embedding_dim))
        self.output_b = initializations.zero((embedding_dim, ))

        self.word_W = initializations.uniform((embedding_dim, n_words))
        self.word_b = initializations.zero((n_words))

        self.params = [
            self.emb,
            self.encode_W, self.encode_U, self.encode_b,
            self.decode_W, self.decode_U, self.decode_b,
            self.output_W, self.output_b,
            self.word_W, self.word_b
        ]

    def encode_lstm(self, x, mask):
        def _step(m_tm_1, x_t, h_tm_1, c_tm_1):
            lstm_preactive = T.dot(h_tm_1, self.encode_U) + \
                             T.dot(x_t, self.encode_W) + \
                             self.encode_b

            i = T.nnet.sigmoid(lstm_preactive[:,0:self.hidden_dim])
            f = T.nnet.sigmoid(lstm_preactive[:,self.hidden_dim:self.hidden_dim*2])
            o = T.nnet.sigmoid(lstm_preactive[:,self.hidden_dim*2:self.hidden_dim*3])
            c = T.tanh(lstm_preactive[:,self.hidden_dim*3:self.hidden_dim*4])

            c = f*c_tm_1 + i*c
            c = m_tm_1[:,None]*c + (1.-m_tm_1)[:,None]*c_tm_1

            h = o*T.tanh(c)
            h = m_tm_1[:,None]*h + (1.-m_tm_1)[:,None]*h_tm_1

            return [h,c]

        h0 = T.alloc(0., x.shape[1], self.hidden_dim)
        c0 = T.alloc(0., x.shape[1], self.hidden_dim)

        rval, updates = theano.scan(
                fn=_step,
                sequences=[mask,x],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def decode_lstm(self, y, mask, context):
        def _step(m_tm_1, y_tm_1, h_tm_1, c_tm_1):
            lstm_preactive = T.dot(h_tm_1, self.decode_U) + \
                             T.dot(y_tm_1, self.decode_W) + \
                             T.dot(context, self.decode_V) + \
                             self.decode_b

            i = T.nnet.sigmoid(lstm_preactive[:,0:self.hidden_dim])
            f = T.nnet.sigmoid(lstm_preactive[:,self.hidden_dim:self.hidden_dim*2])
            o = T.nnet.sigmoid(lstm_preactive[:,self.hidden_dim*2:self.hidden_dim*3])
            c = T.tanh(lstm_preactive[:,self.hidden_dim*3:self.hidden_dim*4])

            c = f*c_tm_1 + i*c
            c = m_tm_1[:,None]*c + (1.-m_tm_1)[:,None]*c_tm_1

            h = o*T.tanh(c)
            h = m_tm_1[:,None]*h + (1.-m_tm_1)[:,None]*h_tm_1

            return [h,c]

        h0 = T.alloc(0., y.shape[1], self.hidden_dim)
        c0 = T.alloc(0., y.shape[1], self.hidden_dim)

        y_in = T.zeros_like(y)
        y_in = T.set_subtensor(y_in[1:], y[:-1])
        #y_out = y

        rval, updates = theano.scan(
                fn=_step,
                sequences=[mask,y_in],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def build_model(self):
        news_sequence = T.imatrix('news_sequence')
        news_mask = T.matrix('news_mask')
        comment_sequence = T.imatrix('comment_sequence')
        comment_mask = T.matrix('comment_mask')

        n_news_sample, n_news_timestep = news_sequence.shape
        news_emb = self.emb[news_sequence.flatten()]
        news_emb = news_emb.reshape([n_news_sample, n_news_timestep, -1])

        news_emb_dimshuffle = news_emb.dimshuffle(1,0,2)
        news_mask_dimshuffle = news_mask.dimshuffle(1,0)

        h_list = self.encode_lstm( news_emb_dimshuffle, news_mask_dimshuffle )
        context = h_list[-1]

        n_comment_sample, n_comment_timestep = comment_sequence.shape
        comment_emb = self.emb[comment_sequence.flatten()]
        comment_emb = comment_emb.reshape([n_comment_sample, n_comment_timestep, -1])

        comment_emb_dimshuffle = comment_emb.dimshuffle(1,0,2)
        comment_mask_dimshuffle = comment_mask.dimshuffle(1,0)

        outputs = self.decode_lstm( comment_emb_dimshuffle, comment_mask_dimshuffle, context )
        outputs = outputs.dimshuffle(1,0,2)

        output_emb = T.dot(outputs, self.output_W) + self.output_b
        output_word = T.dot(output_emb, self.word_W) + self.word_b

        output_shape = output_word.shape

        probs = T.nnet.softmax(output_shape[0]*output_shape[1], output_shape[2])

        encode_function = theano.function(
                inputs=[news_sequence, news_mask],
                outputs=context,
                allow_input_downcast=True)

        decode_function = theano.function(
                inputs=[news_sequence, news_mask, comment_sequence, comment_mask],
                outputs=output_word,
                allow_input_downcast=True)

        return encode_function, decode_function


# 일단 vocabulary부터
def get_vectorizer(data, n_max_words):
    vectorizer = CountVectorizer(max_features=n_max_words, token_pattern=u'(?u)\\b\\w+\\b')

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
    batch_size = 10
    n_max_words = 20000
    embedding_dim = 1024
    hidden_dim = 512
    output_dim = n_max_words
    split_dataset(data_path)

    trainset = pd.read_pickle('train.pickle')
    trainset_news = trainset['news'].values
    trainset_comments = trainset['comments'].map(lambda x: x[0]).values # at this moment, I will treat the first comment only as the target

    vectorizer = get_vectorizer(trainset, n_max_words)
    dictionary = pd.Series(vectorizer.vocabulary_)

    comment_generator = Comment_Generator(n_max_words, embedding_dim, hidden_dim)
    encode_function, decode_function = comment_generator.build_model()

    for start, end in zip(
            range(0, len(trainset)+batch_size, batch_size),
            range(batch_size, len(trainset)+batch_size, batch_size)
        ):

        news_word_index = map(lambda one_news: dictionary[one_news.split(' ')], trainset_news[start:end])
        news_word_index = map(lambda x: x.values[~np.isnan(x.values)], news_word_index)

        news_maxlen = np.max(map(lambda x: len(x), news_word_index))
        news_word = np.zeros((batch_size, news_maxlen))
        news_mask = np.zeros((batch_size, news_maxlen))

        comment_word_index = map(lambda one_comment: dictionary[one_comment.split(' ')], trainset_comments[start:end])
        comment_word_index = map(lambda x: x.values[~np.isnan(x.values)], comment_word_index)

        comment_maxlen = np.max(map(lambda x: len(x), news_word_index))
        comment_word = np.zeros((batch_size, comment_maxlen))
        comment_mask = np.zeros((batch_size, comment_maxlen))

        for inds,arr in enumerate(news_word_index):
            news_word[inds, :len(arr)] = arr
            news_mask[inds, :len(arr)] = 1

        for inds,arr in enumerate(comment_word_index):
            comment_word[inds, :len(arr)] = arr
            comment_mask[inds, :len(arr)] = 1
        ipdb.set_trace()
        output = decode_function(news_word, news_mask, comment_word, comment_mask)


