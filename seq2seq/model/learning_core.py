# randomized word embedding
# pretrained word embedding


import tensorflow as tf
import tensorflow.contrib.layers as layers

class Seq2seqCore:
    def __init__(self,
                 gpu_device=1,
                 encoder_vocab_size=10000,
                 decoder_vocab_size=5000,
                 embedding_size=512,
                 pretrained_embedding_size=200,
                 encoder_hidden_size=128):

        session_config = tf.ConfigProto(device_count={'GPU': 1})
        session_config.gpu_options.visible_device_list = str(gpu_device)
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_config)

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_size = embedding_size
        self.pretrained_embedding_size = pretrained_embedding_size

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = self.encoder_hidden_size * 2
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.encoder_pretrained_inputs = tf.placeholder(shape=(None, None, self.pretrained_embedding_size),
                                                        dtype=tf.float32,
                                                        name='encoder_pretrained_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        batch_size, output_max_length = tf.unstack(tf.shape(self.encoder_inputs))
        start_tokens = tf.ones([batch_size], dtype=tf.int32)
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        decoder_inputs = tf.concat([tf.expand_dims(start_tokens, 1), self.decoder_targets], 1)
        self.decoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')

        encoder_embeddings = tf.Variable(tf.random_uniform([self.encoder_vocab_size, self.embedding_size], -1.0, 1.0),
                                         dtype=tf.float32)
        decoder_embeddings = tf.Variable(tf.random_uniform([self.decoder_vocab_size, self.embedding_size], -1.0, 1.0),
                                         dtype=tf.float32)

        encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)

        encoder_inputs_embedded_concat = tf.concat((encoder_inputs_embedded, self.encoder_pretrained_inputs), axis=2)

        fw_encoder_cell = tf.nn.rnn_cell.LSTMCell(self.encoder_hidden_size)
        bw_encoder_cell = tf.nn.rnn_cell.LSTMCell(self.encoder_hidden_size)

        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_final_state,
          encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_encoder_cell,
                                                                      cell_bw=bw_encoder_cell,
                                                                      inputs=encoder_inputs_embedded_concat,
                                                                      sequence_length=self.encoder_inputs_length,
                                                                      dtype=tf.float32))
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded, self.decoder_inputs_length)
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                               start_tokens=start_tokens, end_token=0)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.encoder_hidden_size, memory=encoder_outputs,
                    memory_sequence_length=self.encoder_inputs_length)
                decoder_cell = tf.nn.rnn_cell.LSTMCell(self.decoder_hidden_size)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism, attention_layer_size=self.encoder_hidden_size)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, self.decoder_vocab_size, reuse=reuse
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=batch_size))
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=5
                )
                return outputs[0]

        train_outputs = decode(train_helper, 'decode')
        self.pred_outputs = decode(pred_helper, 'decode', reuse=True)

        weights = tf.to_float(tf.not_equal(decoder_inputs[:, 1:], 0))
        self.loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output, self.decoder_targets, weights=weights)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        # Saver
        self.saver = tf.train.Saver(max_to_keep=0)

        # Finalizer
        self.sess.graph.finalize()

    def save(self, ckpt_path):
        self.saver.save(self.sess, ckpt_path)

    def load(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def fit(self, input_feed_dict):
        feed_dict = dict()
        feed_dict[self.decoder_targets] = input_feed_dict["decoder_target"]
        feed_dict[self.encoder_inputs] = input_feed_dict["encoder_input"]
        feed_dict[self.encoder_pretrained_inputs] = input_feed_dict["encoder_pretrained"]
        feed_dict[self.encoder_inputs_length] = input_feed_dict["encoder_length"]
        feed_dict[self.decoder_inputs_length] = input_feed_dict["decoder_length"]
        _, l = self.sess.run([self.train_op, self.loss], feed_dict)
        print(l)

    def predict(self, input_feed_dict):
        feed_dict = dict()
        feed_dict[self.encoder_inputs_length] = input_feed_dict["encoder_length"]
        # feed_dict[self.decoder_inputs_length] = input_feed_dict["decoder_length"]
        feed_dict[self.encoder_pretrained_inputs] = input_feed_dict["encoder_pretrained"]
        feed_dict[self.encoder_inputs] = input_feed_dict["encoder_input"]
        prediction_output = self.sess.run(self.pred_outputs, feed_dict)
        return prediction_output

    def evaluate(self, input_feed_dict):
        prediction_output = self.predict(input_feed_dict).tolist()
        groundtruth = input_feed_dict["decoder_target"].T.tolist()
        assert len(prediction_output) == len(groundtruth)
        total_size = len(prediction_output)
        correct_flag = 0
        for idx in range(len(prediction_output)):
            if prediction_output[idx] == groundtruth[idx]:
                correct_flag += 1
        return correct_flag