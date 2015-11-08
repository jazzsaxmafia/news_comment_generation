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
import optim

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
            self.decode_W, self.decode_U, self.decode_V, self.decode_b,
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


        # context : (n_samples, dim_lstm)
        # mask: (n_timestep, n_samples)
        rval, updates = theano.scan(
                fn=_step,
                sequences=[mask,y_in],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def generate_lstm(self, context):
        x0 = T.alloc(0., context.shape[0], self.embedding_dim)
        h0 = T.alloc(0., context.shape[0], self.hidden_dim)
        c0 = T.alloc(0., context.shape[0], self.hidden_dim)

        def _step(x_tm_1, h_tm_1, c_tm_1):
            lstm_preactive = T.dot(h_tm_1, self.decode_U)+ \
                             T.dot(context, self.decode_V)+ \
                             T.dot(x_tm_1, self.decode_W) + \
                             self.decode_b

            i = T.nnet.sigmoid(lstm_preactive[:,0:self.hidden_dim])
            f = T.nnet.sigmoid(lstm_preactive[:,self.hidden_dim:self.hidden_dim*2])
            o = T.nnet.sigmoid(lstm_preactive[:,self.hidden_dim*2:self.hidden_dim*3])
            c = T.tanh(lstm_preactive[:,self.hidden_dim*3:self.hidden_dim*4])

            c = f*c_tm_1 + i*c
            h = o*T.tanh(c)

            x_emb = T.dot(h, self.output_W) + self.output_b # (n_samples, embedding_dim)
            x_word = T.dot(x_emb, self.word_W) + self.word_b # (n_samples, n_words)

            x_index = T.argmax(x_word, axis=1)
            x = self.emb[x_index]

            return [x,h,c]

        rval, updates = theano.scan(
                fn=_step,
                outputs_info=[x0,h0,c0],
                n_steps=20)

        generated_sequence = rval[0]
        return generated_sequence

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

        probs = T.nnet.softmax(output_word.reshape([output_shape[0]*output_shape[1], output_shape[2]]))

        comment_flat = comment_sequence.flatten()
        p_flat = probs.flatten()

        cost = -T.log(p_flat[T.arange(comment_flat.shape[0])*probs.shape[1]+comment_flat] + 1e-8)
        cost = cost.reshape([comment_sequence.shape[0], comment_sequence.shape[1]])
        masked_cost = cost * comment_mask

        cost = (masked_cost).sum() / comment_mask.sum()
        return [
                cost,
                news_sequence,
                news_mask,
                comment_sequence,
                comment_mask,
                cost
                ]

    def build_tester(self):
        news_sequence = T.imatrix('news_sequence')
        news_mask = T.matrix('news_mask')

        n_news_sample, n_news_timestep = news_sequence.shape
        news_emb = self.emb[news_sequence.flatten()]
        news_emb = news_emb.reshape([n_news_sample, n_news_timestep, -1])

        news_emb_dimshuffle = news_emb.dimshuffle(1,0,2)
        news_mask_dimshuffle = news_mask.dimshuffle(1,0)

        h_list = self.encode_lstm( news_emb_dimshuffle, news_mask_dimshuffle )
        context = h_list[-1]

        generated_sequence = self.generate_lstm(context)

        f_generate = theano.function(
                inputs=[news_sequence, news_mask],
                outputs=generated_sequence,
                allow_input_downcast=True)

        return f_generate

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
    model_path = './models/v1/'
    n_epochs = 100
    batch_size = 10
    n_max_words = 10000
    embedding_dim = 256
    hidden_dim = 256
    decay_c = 0.0001
    grad_clip=2.
    learning_rate = 0.005

    trainset_file = 'train.pickle'
    vectorizer_file = 'vectorizer.pickle'

    if not os.path.exists(trainset_file):
        split_dataset(data_path)

    trainset = pd.read_pickle('train.pickle')
    trainset_news = trainset['news'].values
    trainset_comments = trainset['comments'].map(lambda x: x[0]).values # at this moment, I will treat the first comment only as the target

    if not os.path.exists(vectorizer_file):
        vectorizer = get_vectorizer(trainset, n_max_words)
        dictionary = pd.Series(vectorizer.vocabulary_)
    else:
        with open(vectorizer_file) as f:
            vectorizer = cPickle.load(f)
        dictionary = pd.Series(vectorizer.vocabulary_)

    comment_generator = Comment_Generator(n_max_words, embedding_dim, hidden_dim)
    (
        cost,
        news_sequence,
        news_mask,
        comment_sequence,
        comment_mask,
        outputs
    ) = comment_generator.build_model()

    # l2 norm regularizer
    if decay_c > 0. :
        decay_c = theano.shared(np.float32(decay_c), name='decay_c')
        weight_decay = 0.

        for param in comment_generator.params:
            weight_decay += (param ** 2).sum()

        weight_decay *= decay_c
        cost += weight_decay

    # grad 너무 크면 clipping
    grads = T.grad(cost=cost, wrt=comment_generator.params)
    if grad_clip > 0.:
        g2 = 0
        for g in grads:
            g2 += (g**2).sum()
        new_grads=[]

        for g in grads:
            new_grads.append(T.switch(g2 > (grad_clip**2),
                                      g / T.sqrt(g2) * grad_clip,
                                      g ))
        grads = new_grads

    lr = T.scalar('lr')

    f_grad_shared, f_update = optim.sgd_2(
            lr=lr,
            params=comment_generator.params,
            grads=grads,
            inp=[news_sequence, news_mask, comment_sequence, comment_mask],
            cost=cost)

    updates = optim.sgd(cost=outputs,params=comment_generator.params, lr=0.001)
    f_encode = theano.function(
            inputs=[news_sequence, news_mask, comment_sequence, comment_mask],
            outputs=outputs,
            updates=updates,
            allow_input_downcast=True)

    for epoch in range(n_epochs):
        for start, end in zip(
                range(0, len(trainset)+batch_size, batch_size),
                range(batch_size, len(trainset)+batch_size, batch_size)
            ):

            news_word_index = map(lambda one_news: dictionary[one_news.split(' ')], trainset_news[start:end])
            news_word_index = map(lambda x: x.values[~np.isnan(x.values)], news_word_index)
            #news_word_index = map(lambda x: x[:10], news_word_index)

            news_maxlen = np.max(map(lambda x: len(x), news_word_index))
            news_word = np.zeros((batch_size, news_maxlen))
            news_m = np.zeros((batch_size, news_maxlen))

            comment_word_index = map(lambda one_comment: dictionary[one_comment.split(' ')], trainset_comments[start:end])
            comment_word_index = map(lambda x: x.values[~np.isnan(x.values)], comment_word_index)

            #comment_word_index = map(lambda x: x, comment_word_index)

            comment_maxlen = np.max(map(lambda x: len(x), comment_word_index))
            comment_word = np.zeros((batch_size, comment_maxlen))
            comment_m = np.zeros((batch_size, comment_maxlen))

            for inds,arr in enumerate(news_word_index):
                news_word[inds, :len(arr)] = arr
                news_m[inds, :len(arr)] = 1

            for inds,arr in enumerate(comment_word_index):
                comment_word[inds, :len(arr)] = arr
                comment_m[inds, :len(arr)] = 1


            #out = f_encode(news_word, news_m)
            out = f_encode(news_word, news_m, comment_word, comment_m)
            print out

            cost = f_grad_shared(news_word, news_m, comment_word, comment_m)
            f_update(learning_rate)

        learning_rate *= 0.99

        with open(os.path.join(model_path, 'model_'+str(epoch)+'.pickle'), 'wb') as f:
            cPickle.dump(comment_generator, f)


def generate_sequence():
    model_path = 'models/v1/model_15.pickle'
    with open(model_path) as f:
        model = cPickle.load(f)

    testset = pd.read_pickle('test.pickle')
    f_generate = model.build_tester()

    ipdb.set_trace()


