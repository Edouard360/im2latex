import abc

import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, BasicDecoder, \
    ScheduledEmbeddingTrainingHelper, dynamic_decode, GreedyEmbeddingHelper, BeamSearchDecoder, tile_batch

from convolutional_network import init_cnn
from utils.receptive_field import ReceptiveFieldCalculator
from utils.visualize import plot_attention


class Model:
    def __init__(self, vocab_size, beam_width=1, alignment_history=False):
        self.beam_width = beam_width  # For generalization btw Training and Inference

        self.inp = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.num_words = tf.placeholder(tf.int32, shape=[1])
        self.true_labels = tf.placeholder(tf.int32, shape=[None, None])
        self.learning_rate = tf.placeholder(tf.float32)

        self.batch_size = tf.shape(self.inp)[0]

        enc_lstm_dim = 256
        dec_lstm_dim = 512
        self.vocab_size = vocab_size + 4
        embedding_size = 80

        cnn = init_cnn(self.inp)

        # function for map to apply the rnn to each row
        def fn(inp):
            with tf.variable_scope('encoder_rnn'):
                with tf.variable_scope('forward'):
                    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
                with tf.variable_scope('backward'):
                    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)

            output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inp, dtype=tf.float32)
            return tf.concat(output, 2)

        fun = tf.make_template('fun', fn)
        rows_first = tf.transpose(cnn, [1, 0, 2, 3])
        res = tf.map_fn(fun, rows_first, dtype=tf.float32)
        self.encoder_output = tf.transpose(res, [1, 0, 2, 3])

        attention_states_depth = 2 * enc_lstm_dim

        attention_states = tf.reshape(self.encoder_output, [self.batch_size, -1, attention_states_depth])
        attention_states_tiled = tile_batch(attention_states, self.beam_width)  # For generalization

        attention_weights_depth = attention_states_depth
        attention_layer_size = attention_states_depth
        attention_mechanism = BahdanauAttention(attention_weights_depth, attention_states_tiled)

        dec_lstm_cell = tf.nn.rnn_cell.LSTMCell(dec_lstm_dim)
        self.cell = AttentionWrapper(cell=dec_lstm_cell,
                                     attention_mechanism=attention_mechanism,
                                     attention_layer_size=attention_layer_size,
                                     alignment_history=alignment_history)

        self.embedding = tf.get_variable("embedding", [self.vocab_size, embedding_size])

        self.setup_decoder()

        self.final_outputs, self.final_state, _ = dynamic_decode(self.decoder, maximum_iterations=self.num_words[0] - 1)

        self.finalize_model()

    def eval(self, sess, feed_dict):
        return sess.run(self.accuracy, feed_dict=feed_dict)

    @abc.abstractmethod
    def setup_decoder(self):
        pass

    @abc.abstractmethod
    def finalize_model(self):
        pass


class TrainModel(Model):
    def __init__(self, vocab_size):
        super(TrainModel, self).__init__(vocab_size, beam_width=1)

    def setup_decoder(self):
        decoder_emb_inp = tf.nn.embedding_lookup([self.embedding], self.true_labels[:, :-1])
        decoder_lengths = tf.tile([self.num_words[0] - 1], [self.batch_size])
        helper = ScheduledEmbeddingTrainingHelper(decoder_emb_inp, decoder_lengths, self.embedding, 0.1)
        # self.helper = TrainingHelper(decoder_emb_inp, decoder_lengths)
        dec_init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        self.decoder = BasicDecoder(self.cell, helper, dec_init_state,
                                    output_layer=tf.layers.Dense(self.vocab_size))

    def finalize_model(self):
        final_outputs = self.final_outputs.rnn_output
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_outputs,
                                                           labels=self.true_labels[:, 1:]))
        self.train_step = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.to_int32(tf.argmax(final_outputs, 2)), self.true_labels[:, 1:])
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, sess, feed_dict):
        _, train_accuracy = sess.run([self.train_step, self.accuracy], feed_dict=feed_dict)
        return train_accuracy


class GreedyInferenceModel(Model):
    def __init__(self, vocab_size):
        super(GreedyInferenceModel, self).__init__(vocab_size, beam_width=1, alignment_history=True)
        self.rf_calc = ReceptiveFieldCalculator()

    def setup_decoder(self):
        helper = GreedyEmbeddingHelper(self.embedding, tf.tile([0], [self.batch_size]), -1)
        dec_init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        self.decoder = BasicDecoder(self.cell, helper, dec_init_state,
                                    output_layer=tf.layers.Dense(self.vocab_size))

    def finalize_model(self):
        final_outputs = self.final_outputs.rnn_output
        self.predicted_labels = tf.to_int32(tf.argmax(final_outputs, 2))
        correct_prediction = tf.equal(self.predicted_labels, self.true_labels[:, 1:])
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, sess, feed_dict):
        return sess.run(self.predicted_labels, feed_dict=feed_dict)

    def visualize(self, sess, image, feed_dict, time=0):
        rf_coords = self.rf_calc.get_receptive_field_coords(image)
        tf_alignment_history = tf.squeeze(self.final_state.alignment_history.stack())
        alignment_history, predicted_labels = sess.run([tf_alignment_history, self.predicted_labels],
                                                       feed_dict=feed_dict)
        plot_attention(image, predicted_labels, rf_coords, alignment_history, time)


class BeamSearchInferenceModel(Model):
    def __init__(self, vocab_size, beam_width=5):
        super(BeamSearchInferenceModel, self).__init__(vocab_size, beam_width=beam_width)

    def setup_decoder(self):
        self.dec_init_state = self.cell.zero_state(self.batch_size * self.beam_width, dtype=tf.float32)
        self.decoder = BeamSearchDecoder(cell=self.cell,
                                         embedding=self.embedding,
                                         start_tokens=tf.tile([0], [self.batch_size]),
                                         end_token=-1,
                                         initial_state=self.dec_init_state,
                                         beam_width=self.beam_width,
                                         output_layer=tf.layers.Dense(self.vocab_size))

    def finalize_model(self):
        self.predicted_labels = self.final_outputs.predicted_ids[:, :, 0]
        correct_prediction = tf.equal(self.predicted_labels, self.true_labels[:, 1:])
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, sess, feed_dict):
        return sess.run(self.predicted_labels, feed_dict=feed_dict)
