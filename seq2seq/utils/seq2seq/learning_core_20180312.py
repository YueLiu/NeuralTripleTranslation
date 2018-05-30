import tensorflow as tf

class Seq2seqCore:
    def __init__(self,
                 gpu_device=1,
                 encoder_vocab_size=10000,
                 decoder_vocab_size=5000,
                 embedding_size=512,
                 encoder_hidden_size=128):

        session_config = tf.ConfigProto(device_count={'GPU': 1})
        session_config.gpu_options.visible_device_list = str(gpu_device)
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_config)

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_size = embedding_size

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = self.encoder_hidden_size * 2
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

        encoder_embeddings = tf.Variable(tf.random_uniform([self.encoder_vocab_size, self.embedding_size], -1.0, 1.0),
                                         dtype=tf.float32)
        decoder_embeddings = tf.Variable(tf.random_uniform([self.decoder_vocab_size, self.embedding_size], -1.0, 1.0),
                                         dtype=tf.float32)

        encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.encoder_inputs)

        encoder_cell = tf.nn.rnn_cell.LSTMCell(self.encoder_hidden_size)

        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_final_state,
          encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                      cell_bw=encoder_cell,
                                                                      inputs=encoder_inputs_embedded,
                                                                      sequence_length=self.encoder_inputs_length,
                                                                      dtype=tf.float32, time_major=True))
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h)

        decoder_cell = tf.nn.rnn_cell.LSTMCell(self.decoder_hidden_size)

        encoder_max_time, batch_size = tf.unstack(tf.shape(self.encoder_inputs))

        W = tf.Variable(tf.random_uniform([self.decoder_hidden_size, self.decoder_vocab_size], -1, 1), dtype=tf.float32)
        b = tf.Variable(tf.zeros([decoder_vocab_size]), dtype=tf.float32)

        # EOS = 1
        # PAD = 0

        eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
        pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

        eos_step_embedded = tf.nn.embedding_lookup(decoder_embeddings, eos_time_slice)
        pad_step_embedded = tf.nn.embedding_lookup(decoder_embeddings, pad_time_slice)

        def loop_fn_initial():
            initial_elements_finished = (0 >= self.decoder_inputs_length)  # all False at the initial step
            initial_input = eos_step_embedded
            initial_cell_state = encoder_final_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
            def get_next_input():
                output_logits = tf.add(tf.matmul(previous_output, W), b)
                prediction = tf.argmax(output_logits, axis=1)
                next_input = tf.nn.embedding_lookup(decoder_embeddings, prediction)
                return next_input

            elements_finished = (time >= self.decoder_inputs_length)  # this operation produces boolean tensor of [batch_size]
            # defining if corresponding sequence has ended

            finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            transition_input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
            transition_state = previous_state
            transition_output = previous_output
            transition_loop_state = None

            return (elements_finished,
                    transition_input,
                    transition_state,
                    transition_output,
                    transition_loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:  # time == 0
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()

        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
        self.decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.decoder_vocab_size))

        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.decoder_vocab_size, dtype=tf.float32),
            logits=self.decoder_logits,
        )

        self.loss = tf.reduce_mean(stepwise_cross_entropy)
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
        feed_dict[self.encoder_inputs_length] = input_feed_dict["encoder_length"]
        feed_dict[self.decoder_inputs_length] = input_feed_dict["decoder_length"]
        _, l = self.sess.run([self.train_op, self.loss], feed_dict)
        print(l)

    def predict(self, input_feed_dict):
        feed_dict = dict()
        feed_dict[self.encoder_inputs_length] = input_feed_dict["encoder_length"]
        feed_dict[self.decoder_inputs_length] = input_feed_dict["decoder_length"]
        feed_dict[self.encoder_inputs] = input_feed_dict["encoder_input"]
        prediction_output = self.sess.run(self.decoder_prediction, feed_dict)
        prediction_output = prediction_output[0:3,:].T
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
