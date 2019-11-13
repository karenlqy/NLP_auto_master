import tensorflow as tf
from typing import Dict
import numpy as np
from tensorflow.keras.preprocessing import text, sequence
from gensim.models import Word2Vec
import os
import pandas as pd
from itertools import islice

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, vec_dim, matrix, gru_size):
        super(Encoder, self).__init__()
        weights = [matrix]
        self.embedding = tf.keras.layer.Embedding(vocab_size, vec_dim, weights=weights, trainable=False)

        self.gru = tf.keras.layer.GRU(gru_size, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        # xavier
        # 'Batch Normalization Accelerating Deep Network Training'
        # https://zhuanlan.zhihu.com/p/25110150

        self.gru_size = gru_size

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h = self.gru(embed, initial_state=states)
        # output:[batch_size, max_length, gru_size]
        # state_h: [batch_size, gru_size]
        return output, state_h

    def init_states(self, batch_size):
        return tf.zeros([batch_size, self.gru_size])


class BahdnauAttention(tf.keras.Model):
    def __init__(self, units):
        # units：gru_size
        super(BahdnauAttention, self).__init__()
        self.W1 = tf.keras.layer.Dense(units)
        self.W2 = tf.keras.layer.Dense(units)
        self.V = tf.keras.layer.Dense(1)

    def call(self, query, values):
        # query: state_h
        # values: output
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # hidden_with_time_axis: [batch_size, 1, gru_size]

        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # self.W2(hidden_with_time_axis): [batch_size, 1, gru_size]
        # self.W1(values): [batch_size, max_len, gru_size]
        # score: [batch_size, max_len, 1]

        attention_weight = tf.nn.softmax(score, axis=1)

        context_vector = attention_weight * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # context_vector: [batch_size, gru_size]

        return context_vector, attention_weight


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, vec_dim, matrix, gru_size):
        super(Decoder, self).__init__()
        self.gru_size = gru_size
        weights = [matrix]
        self.embedding = tf.keras.layer.Embedding(vocab_size, vec_dim, weights=weights, trainable=False)

        self.gru = tf.keras.layer.GRU(gru_size, return_sequences=True, return_state=True,
                                      recurrent_initializer='glorot_uniform')
        self.attention = BahdnauAttention(self.gru_size)
        self.wc = tf.keras.layer.Dense(self.gru_size, activation='tanh')
        self.ws = tf.keras.layer.Dense(vocab_size)

    def call(self, sequence, state, encoder_output):
        embed = self.embedding(sequence)
        gru_out, state_h = self.gru(embed)
        context_vector, attention_weight = self.attention(gru_out, encoder_output)
        gru_out = tf.concat([tf.squeeze(context_vector, 1), tf.squeeze(gru_out, 1)], 1)
        # gru_out: ht(values)  [batch, 1, gru_size]
        # tf.squeeze: 删掉大小是1的维度
        # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
        #         shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
        # gru_out: [batch_size, gru_size x 2]

        gru_out = self.wc(gru_out)
        # gru_out: [batch_size, gru_size]

        logits = self.ws(gru_out)
        # output: [batch_size, vocab]

        return logits, state_h, attention_weight

    def init_states(self,batch_size):
        return (tf.zeros([batch_size,self.gru_size]),
                tf.zeros([batch_size,self.gru_size]))

def load_w2v_model(filename):
    model = Word2Vec.load(filename)
    return model
def tokenize(lang,max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',lower=False,num_words = max_features)
    tokenizer.fit_on_texts(lang) #这样保证encoder和decoder使用的是同一个字典，按道理讲，decoder输出少，不应该维护那么长的字典
    word_index = tokenizer.word_index
    # print(word_index)
    #print(len(word_index))#一共十多万个词语，太多了
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post',maxlen=max_len)
    return tensor, tokenizer,word_index


def load_dataset(data):
    source = [str(m) for m in data['input'].values.tolist()]
    target = [str(m) for m in data['Report'].values.tolist()]
    # inp_lang = data['Report'].values.tolist()[:num_examples]
    input_tensor, tokenizer1,word_index1 = tokenize(source,max_length_inp)
    target_tensor, tokenizer2,word_index2 = tokenize(target,max_length_targ)
    return input_tensor, target_tensor,word_index1,word_index2,tokenizer1,tokenizer2

root_path = os.path.abspath('../')
model_path = os.path.join(root_path, "data/train_save.csv")
data = pd.read_csv(model_path,encoding='utf-8')
max_features = 300
maxlen = 300
embed_size = 100
max_length_inp,max_length_targ= 500, 50

def get_embedding():
    root_path = os.path.abspath('../')
    model_path = os.path.join(root_path, "data/w2v.model")
    data_path = os.path.join(root_path, "data/train_save.csv")
    data = pd.read_csv(data_path, encoding='utf-8')
    model = load_w2v_model(model_path)
    input_tensor, target_tensor,word_index1,word_index2,tokenizer1,tokenizer2= load_dataset(data)

    #encoder embedding
    nb_words1 = min(max_features,len(word_index1))
    embedding_matrix1 = np.zeros((nb_words1, embed_size))
    # print(len(embedding_matrix[2]))
    for word,i in word_index1.items():
        if int(i) >= nb_words1: continue
        if word not in model.wv.vocab:
            embedding_vector = np.random.uniform(-0.025, 0.025, (embed_size))
            embedding_matrix1[i] = embedding_vector
        else:
            embedding_vector = model.wv[word]
            embedding_matrix1[i] = embedding_vector


    # decoder embedding
    nb_words2 = min(max_features, len(word_index2))
    embedding_matrix2 = np.zeros((nb_words2, embed_size))
    # print(len(embedding_matrix[2]))
    for word, i in word_index2.items():
        if int(i) >= nb_words2: continue
        if word not in model.wv.vocab:
            embedding_vector = np.random.uniform(-0.025, 0.025, (embed_size))
            embedding_matrix2[i] = embedding_vector
        else:
            embedding_vector = model.wv[word]
            embedding_matrix2[i] = embedding_vector

    return embedding_matrix1,embedding_matrix2,input_tensor,target_tensor,tokenizer1,tokenizer2

from embedding import get_embedding

embedding_matrix1,embedding_matrix2,input_tensor,target_tensor,tokenizer1,tokenizer2 = get_embedding()

def data_loader(input_tensor,target_tensor):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(len(input_tensor))
    dataset = dataset.batch(len(target_tensor), drop_remainder=True)
    # example_input_batch, example_target_batch = next(iter(dataset))
    # example_input_batch.shape, example_target_batch.shape
    return dataset

embedding_matrix1

class Auto_model:
    def __init__(self,input_tensor,
                 target_tensor,
                 batch_size,
                 embedding_matrix1,
                 embedding_matrix2,
                 tokenizer1,
                 tokenizer2,
                 unit):
        self.BUFFER_SIZE = len(input_tensor)
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor
        self.encoder_embedding = embedding_matrix1
        self.decoder_embedding = embedding_matrix2
        self.batch_size = batch_size
        self.steps_per_epoch = len(input_tensor) // batch_size
        self.embedding_dim = embedding_matrix1.shape[1]
        self.unit = unit
        self.vocab_inp_size = embedding_matrix1.shape[0]
        self.vocab_tar_size = embedding_matrix2.shape[0]
        self.tokenizer_encoder = tokenizer1
        self.tokenizer_decoder = tokenizer2

        example_input_batch, example_target_batch = self.get_batch()
        self.build_network()



    def get_batch(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor, self.target_tensor)).shuffle(len(self.input_tensor))
        self.dataset = dataset.batch(len(self.target_tensor), drop_remainder=True)
        example_input_batch, example_target_batch = next(iter(self.dataset))
        return example_input_batch, example_target_batch

    def build_network(self):
        #encoder part
        example_input_batch, example_target_batch = self.get_batch()
        self.encoder = Encoder(self.vocab_inp_size,self.embedding_dim,self.encoder_embedding,self.unit)
        sample_hidden = self.encoder.init_states(self.batch_size)
        output,state_h,context_v = self.encoder(example_input_batch, sample_hidden)
        #attention part
        self.attention_layer = BahdanauAttention(2)
        attention_result, attention_weights = self.attention_layer(state_h, output)
        #decoder part
        self.decoder = Decoder(self.tokenizer_decoder,self.embedding_dim,self.decoder_embedding,self.unit)
        logits, state_h, state_c, aligment  = self.decoder(tf.random.uniform((self.batch_size, 1)), sample_hidden, output)


    @tf.function
    def train_loss_op(self,inp, targ, enc_hidden):
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        checkpoint_dir = './model_save'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden ,context= self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.tokenizer_decoder.word_index['<s>']] * self.batch_size, 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def run_op(self,epochs):
        for epoch in epochs:
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.train_loss_op(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


Auto_model(input_tensor = input_tensor,target_tensor = target_tensor,batch_size = 100, embedding_matrix1 = embedding_matrix1,embedding_matrix2 = embedding_matrix2,tokenizer1 = tokenizer1,tokenizer2 = tokenizer2, unit = ).run_op(10)
