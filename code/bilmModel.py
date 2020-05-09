import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


class Model(object):

    def __init__(self, options, is_training):
        self.dropout = 0.5
        self.is_training = False
        self.options = options
        self._build()

    def generate(self, seq):
        X = []
        y = []
        for i in range(len(seq) - self.options["TIME_STEPS"]):
            X.append([seq[i:i + self.options["TIME_STEPS"]]])
            y.append([seq[i + self.options["TIME_STEPS"]]])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _build(self):
        TIME_STEPS = 10
        BATCH_SIZE = 128
        HIDDEN_UNITS = 128
        LEARNING_RATE = 0.001
        EPOCH = 50
        # place hoder
        self.X_p = tf.placeholder(dtype=tf.float32, shape=(None, TIME_STEPS, 1), name="input_placeholder")
        self.y_p = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="pred_placeholder")

        # lstm instance
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_UNITS)
        lstm_cell1 = rnn.BasicLSTMCell(num_units=1)
        multi_lstm = rnn.MultiRNNCell(cells=[lstm_cell, lstm_cell1])

        # initialize to zero
        # init_state=lstm_cell.zero_state(batch_size=BATCH_SIZE,dtype=tf.float32)
        init_state = multi_lstm.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)

        # dynamic rnn
        outputs, states = tf.nn.dynamic_rnn(cell=multi_lstm, inputs=self.X_p, initial_state=init_state,
                                            dtype=tf.float32)
        # print(outputs.shape)
        h = outputs[:, -1, :]
        # print(h.shape)
        # --------------------------------------------------------------------------------------------#

        # ---------------------------------define loss and optimizer----------------------------------#
        mse = tf.losses.mean_squared_error(labels=self.y_p, predictions=h)
        # print(loss.shape)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=mse)

        init = tf.global_variables_initializer()
        self.init = init
        self.optimizer = optimizer
        self.mse = mse
        self.result = h
        self.saver = tf.train.Saver(tf.global_variables())


class BiLstm_Model(object):

    def __init__(self, config,input,label):
        self.TimeStep = config.TimeStep
        self.input = config.input
        self.batchsize = config.batchsize
        self.hidden_unit = config.hidden_unit
        self.learning_rate = config.learning_rate
        self.epoch = config.epoch
        self.projection_dim = config.projection_dim
        self.token_size = config.token_size
        self.n_negative_samples_batch = config.n_negative_samples_batch
        self.weight = {
            'in': tf.Variable(tf.random_normal([self.token_size, self.projection_dim]),name="W"),
        }
        self.bias = {
            'in': tf.Variable(tf.random_normal([self.token_size]),name='bias'),
        }
        self.is_Training = config.is_Training
        self.initial_state = []
        self.final_state = []

        self._build(input,label)

    def _build(self,input,label):

        """
        双向LSTM模型来对图像进行分类
        :return:
        """
        '''
        LSTM在进行对序列数据进行处理的时候，需要先将其转化为满足网络的格式[batch，timestep,features]
        '''

        fw_Cell = []
        bw_Cell = []
        # 进行的多层双向神经网络
        for i in range(2):
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit,cell_clip=3,num_proj=self.projection_dim,proj_clip=3)
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit,cell_clip=3,num_proj=self.projection_dim,proj_clip=3)
            if i != 0:
                fw_lstm_cell = tf.nn.rnn_cell.ResidualWrapper(fw_lstm_cell)
                bw_lstm_cell = tf.nn.rnn_cell.ResidualWrapper(bw_lstm_cell)
            if self.is_Training:
                fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell,
                                                          input_keep_prob=0.9)
                bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell,
                                                          input_keep_prob=0.9)
            fw_Cell.append(fw_lstm_cell)
            bw_Cell.append(bw_lstm_cell)



        stack_lstm_fw = tf.nn.rnn_cell.MultiRNNCell(cells=fw_Cell)
        stack_lstm_bw = tf.nn.rnn_cell.MultiRNNCell(cells=bw_Cell)

        self.output = []
        with tf.control_dependencies([input[0]]):
            with tf.variable_scope('RNN_0'):
                self.initial_state.append(stack_lstm_fw.zero_state(self.batchsize, dtype=tf.float32))
                _lstm_output_unpacked_fw, final_state_fw = tf.nn.static_rnn(stack_lstm_fw,inputs=tf.unstack(input[0],axis=1),initial_state=self.initial_state[0],dtype=tf.float32)
                # if self.is_Training is False:
                self.output.append(_lstm_output_unpacked_fw)

        with tf.control_dependencies([input[1]]):
            with tf.variable_scope('RNN_1'):
                self.initial_state.append(stack_lstm_bw.zero_state(self.batchsize, dtype=tf.float32))
                _lstm_output_unpacked_bw, final_state_bw = tf.nn.static_rnn(stack_lstm_bw, inputs=tf.unstack(input[1],axis=1),initial_state=self.initial_state[-1],dtype=tf.float32)
                # if self.is_Training is False:
                self.output.append(_lstm_output_unpacked_bw)

        self.final_state.append(final_state_fw)
        self.final_state.append(final_state_bw)

        lstm_output_flat_fw=tf.reshape(tf.stack(_lstm_output_unpacked_fw, axis=1),[-1, self.projection_dim])
        lstm_output_flat_bw = tf.reshape(tf.stack(_lstm_output_unpacked_bw, axis=1), [-1, self.projection_dim])

        lstm_output = []

        if self.is_Training:
            lstm_output.append(tf.nn.dropout(lstm_output_flat_fw,0.9))
            lstm_output.append(tf.nn.dropout(lstm_output_flat_bw, 0.9))
        else:
            lstm_output.append(lstm_output_flat_fw)
            lstm_output.append(lstm_output_flat_bw)

        self._build_loss(lstm_output,label)
        # return tf.add(tf.matmul(output, self.weight['out']), self.bias['out'])  # 全连接层进行输出

    def _build_loss(self, lstm_output,label):
        self.all_losses = []
        next_token_id_flat_fw = tf.reshape(label[0], [-1, 1])
        next_token_id_flat_bw = tf.reshape(label[1], [-1, 1])
        with tf.control_dependencies([lstm_output[0]]):
            if self.is_Training:
                losses = tf.nn.sampled_softmax_loss(
                    self.weight["in"], self.bias["in"],
                    next_token_id_flat_fw, lstm_output[0],
                    self.n_negative_samples_batch,
                    self.token_size,
                    num_true=1)
                self.all_losses.append(tf.reduce_mean(losses))
            else:
                self.all_losses.append(0.0)
        with tf.control_dependencies([lstm_output[1]]):
            if self.is_Training:
                losses = tf.nn.sampled_softmax_loss(
                    self.weight["in"], self.bias["in"],
                    next_token_id_flat_bw, lstm_output[1],
                    self.n_negative_samples_batch,
                    self.token_size,
                    num_true=1)
                self.all_losses.append(tf.reduce_mean(losses))
            else:
                self.all_losses.append(0.0)

        self.total_loss = 0.5 * (self.all_losses[0]
                                    + self.all_losses[1])

    def _fine_tuning_loss_top_k(self,k,input,lstm_output,label):

        pass



class BiLstm_Model_Structual(object):

    def __init__(self, config,input,label):
        self.TimeStep = config.TimeStep
        self.input = config.input
        self.batchsize = config.batchsize
        self.hidden_unit = config.hidden_unit
        self.learning_rate = config.learning_rate
        self.epoch = config.epoch
        self.projection_dim = config.projection_dim
        self.token_size = config.token_size
        self.n_negative_samples_batch = config.n_negative_samples_batch
        self.weight = {
            'in': tf.Variable(tf.random_normal([self.token_size, self.projection_dim]),name="W"),
        }
        self.bias = {
            'in': tf.Variable(tf.random_normal([self.token_size]),name='bias'),
        }
        self.embedding_hash = tf.Variable(tf.random_normal([self.token_size, self.projection_dim]),name="embhash",trainable=False)
        self.is_Training = config.is_Training
        self.initial_state = []
        self.final_state = []

        self._build(input,label)

    def _build(self,input,label):

        """
        双向LSTM模型来对图像进行分类
        :return:
        """
        '''
        LSTM在进行对序列数据进行处理的时候，需要先将其转化为满足网络的格式[batch，timestep,features]
        '''
        e1 = tf.nn.embedding_lookup(params=self.embedding_hash,ids=input[0])
        e2 = tf.nn.embedding_lookup(params=self.embedding_hash,ids=input[-1])
        fw_Cell = []
        bw_Cell = []
        # 进行的多层双向神经网络
        for i in range(2):
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit,cell_clip=3,num_proj=self.projection_dim,proj_clip=3)
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit,cell_clip=3,num_proj=self.projection_dim,proj_clip=3)
            if i != 0:
                fw_lstm_cell = tf.nn.rnn_cell.ResidualWrapper(fw_lstm_cell)
                bw_lstm_cell = tf.nn.rnn_cell.ResidualWrapper(bw_lstm_cell)
            if self.is_Training:
                fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell,
                                                          input_keep_prob=0.9)
                bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell,
                                                          input_keep_prob=0.9)
            fw_Cell.append(fw_lstm_cell)
            bw_Cell.append(bw_lstm_cell)



        stack_lstm_fw = tf.nn.rnn_cell.MultiRNNCell(cells=fw_Cell)
        stack_lstm_bw = tf.nn.rnn_cell.MultiRNNCell(cells=bw_Cell)

        self.output = []
        with tf.control_dependencies([input[0]]):
            with tf.variable_scope('RNN_0'):
                self.initial_state.append(stack_lstm_fw.zero_state(self.batchsize, dtype=tf.float32))
                _lstm_output_unpacked_fw, final_state_fw = tf.nn.static_rnn(stack_lstm_fw,inputs=tf.unstack(e1,axis=1),initial_state=self.initial_state[0],dtype=tf.float32)
                # if self.is_Training is False:
                self.output.append(_lstm_output_unpacked_fw)

        with tf.control_dependencies([input[1]]):
            with tf.variable_scope('RNN_1'):
                self.initial_state.append(stack_lstm_bw.zero_state(self.batchsize, dtype=tf.float32))
                _lstm_output_unpacked_bw, final_state_bw = tf.nn.static_rnn(stack_lstm_bw, inputs=tf.unstack(e2,axis=1),initial_state=self.initial_state[-1],dtype=tf.float32)
                # if self.is_Training is False:
                self.output.append(_lstm_output_unpacked_bw)

        self.final_state.append(final_state_fw)
        self.final_state.append(final_state_bw)

        lstm_output_flat_fw=tf.reshape(tf.stack(_lstm_output_unpacked_fw, axis=1),[-1, self.projection_dim])
        lstm_output_flat_bw = tf.reshape(tf.stack(_lstm_output_unpacked_bw, axis=1), [-1, self.projection_dim])

        lstm_output = []

        if self.is_Training:
            lstm_output.append(tf.nn.dropout(lstm_output_flat_fw,0.9))
            lstm_output.append(tf.nn.dropout(lstm_output_flat_bw, 0.9))
        else:
            lstm_output.append(lstm_output_flat_fw)
            lstm_output.append(lstm_output_flat_bw)

        self._build_loss(lstm_output,label)
        # return tf.add(tf.matmul(output, self.weight['out']), self.bias['out'])  # 全连接层进行输出

    def _build_loss(self, lstm_output,label):
        self.all_losses = []
        next_token_id_flat_fw = tf.reshape(label[0], [-1, 1])
        next_token_id_flat_bw = tf.reshape(label[1], [-1, 1])
        with tf.control_dependencies([lstm_output[0]]):
            if self.is_Training:
                losses = tf.nn.sampled_softmax_loss(
                    self.weight["in"], self.bias["in"],
                    next_token_id_flat_fw, lstm_output[0],
                    self.n_negative_samples_batch,
                    self.token_size,
                    num_true=1)
                self.all_losses.append(tf.reduce_mean(losses))
            else:
                self.all_losses.append(0.0)
        with tf.control_dependencies([lstm_output[1]]):
            if self.is_Training:
                losses = tf.nn.sampled_softmax_loss(
                    self.weight["in"], self.bias["in"],
                    next_token_id_flat_bw, lstm_output[1],
                    self.n_negative_samples_batch,
                    self.token_size,
                    num_true=1)
                self.all_losses.append(tf.reduce_mean(losses))
            else:
                self.all_losses.append(0.0)

        self.total_loss = 0.5 * (self.all_losses[0]
                                    + self.all_losses[1])

    def _fine_tuning_loss_top_k(self,k,input,lstm_output,label):

        pass




class Fine_Tuning_BiLstm_Model(object):

    def __init__(self, config,input,label):
        self.TimeStep = config.TimeStep
        self.input = config.input
        self.batchsize = config.batchsize
        self.hidden_unit = config.hidden_unit
        self.learning_rate = config.learning_rate
        self.epoch = config.epoch
        self.projection_dim = config.projection_dim
        self.token_size = config.token_size
        self.n_negative_samples_batch = config.n_negative_samples_batch
        self.weight = {
            'in': tf.Variable(tf.random_normal([1, self.projection_dim]),name="W"),
        }
        self.bias = {
            'in': tf.Variable(tf.random_normal([1]),name='bias'),
        }
        self.is_Training = config.is_Training
        self.initial_state = []
        self.final_state = []

        self._build(input,label)

    def _build(self,input,label):

        """
        双向LSTM模型来对图像进行分类
        :return:
        """
        '''
        LSTM在进行对序列数据进行处理的时候，需要先将其转化为满足网络的格式[batch，timestep,features]
        '''

        fw_Cell = []
        bw_Cell = []
        # 进行的多层双向神经网络
        for i in range(2):
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit,cell_clip=3,num_proj=self.projection_dim,proj_clip=3)
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit,cell_clip=3,num_proj=self.projection_dim,proj_clip=3)
            if i != 0:
                fw_lstm_cell = tf.nn.rnn_cell.ResidualWrapper(fw_lstm_cell)
                bw_lstm_cell = tf.nn.rnn_cell.ResidualWrapper(bw_lstm_cell)
            if self.is_Training:
                fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell,
                                                          input_keep_prob=0.9)
                bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell,
                                                          input_keep_prob=0.9)
            fw_Cell.append(fw_lstm_cell)
            bw_Cell.append(bw_lstm_cell)



        stack_lstm_fw = tf.nn.rnn_cell.MultiRNNCell(cells=fw_Cell)
        stack_lstm_bw = tf.nn.rnn_cell.MultiRNNCell(cells=bw_Cell)

        self.output = []
        with tf.control_dependencies([input[0]]):
            with tf.variable_scope('RNN_0'):
                self.initial_state.append(stack_lstm_fw.zero_state(self.batchsize, dtype=tf.float32))
                _lstm_output_unpacked_fw, final_state_fw = tf.nn.static_rnn(stack_lstm_fw,inputs=tf.unstack(input[0],axis=1),initial_state=self.initial_state[0],dtype=tf.float32)
                # if self.is_Training is False:
                self.output.append(_lstm_output_unpacked_fw)

        with tf.control_dependencies([input[1]]):
            with tf.variable_scope('RNN_1'):
                self.initial_state.append(stack_lstm_bw.zero_state(self.batchsize, dtype=tf.float32))
                _lstm_output_unpacked_bw, final_state_bw = tf.nn.static_rnn(stack_lstm_bw, inputs=tf.unstack(input[1],axis=1),initial_state=self.initial_state[-1],dtype=tf.float32)
                # if self.is_Training is False:
                self.output.append(_lstm_output_unpacked_bw)

        self.final_state.append(final_state_fw)
        self.final_state.append(final_state_bw)

        # lstm_output_flat_fw=tf.reshape(tf.stack(_lstm_output_unpacked_fw[-1], axis=1),[-1, self.projection_dim])
        # lstm_output_flat_bw = tf.reshape(tf.stack(_lstm_output_unpacked_bw[-1], axis=1), [-1, self.projection_dim])

        lstm_output_flat_fw = _lstm_output_unpacked_fw[-1]
        lstm_output_flat_bw = _lstm_output_unpacked_bw[-1]

        lstm_output = []

        if self.is_Training:
            lstm_output.append(tf.nn.dropout(lstm_output_flat_fw,0.9))
            lstm_output.append(tf.nn.dropout(lstm_output_flat_bw, 0.9))
        else:
            lstm_output.append(lstm_output_flat_fw)
            lstm_output.append(lstm_output_flat_bw)


        fine_tuning_out_fw = tf.layers.dense(inputs=lstm_output[0], activation='relu', units=4096)
        fine_tuning_out_bw = tf.layers.dense(inputs=lstm_output[-1], activation='relu', units=4096)

        if self.is_Training:
            fine_tuning_out_fw = tf.nn.dropout(fine_tuning_out_fw,0.9)
            fine_tuning_out_bw = tf.nn.dropout(fine_tuning_out_bw,0.9)

        concat_input = tf.concat([fine_tuning_out_fw,fine_tuning_out_bw],axis=1)
        self.loss_input = tf.layers.dense(concat_input,activation=None,units=1)
        self._fine_tuning_loss_top_k(int(lstm_output_flat_bw.shape[0]),self.loss_input,label)
        #self._build_loss(lstm_output,label)
        # return tf.add(tf.matmul(output, self.weight['out']), self.bias['out'])  # 全连接层进行输出

    def _build_loss(self, lstm_output,label):
        self.all_losses = []
        next_token_id_flat_fw = tf.reshape(label[0], [-1, 1])
        next_token_id_flat_bw = tf.reshape(label[1], [-1, 1])
        with tf.control_dependencies([lstm_output[0]]):
            if self.is_Training:
                losses = tf.nn.sampled_softmax_loss(
                    self.weight["in"], self.bias["in"],
                    next_token_id_flat_fw, lstm_output[0],
                    self.n_negative_samples_batch,
                    self.token_size,
                    num_true=1)
                self.all_losses.append(tf.reduce_mean(losses))
            else:
                self.all_losses.append(0.0)
        with tf.control_dependencies([lstm_output[1]]):
            if self.is_Training:
                losses = tf.nn.sampled_softmax_loss(
                    self.weight["in"], self.bias["in"],
                    next_token_id_flat_bw, lstm_output[1],
                    self.n_negative_samples_batch,
                    self.token_size,
                    num_true=1)
                self.all_losses.append(tf.reduce_mean(losses))
            else:
                self.all_losses.append(0.0)

        self.total_loss = 0.5 * (self.all_losses[0]
                                    + self.all_losses[1])

    def _fine_tuning_loss_top_k(self,topk,input,label):
        if self.is_Training:
            labels = tf.reshape(label[0], [-1, 1])
            loss = tf.losses.mean_squared_error(labels=labels, predictions=input)
            # prop = self.output[0][1]
            # prop = tf.nn.l2_normalize(prop,axis=1)
            #
            # self.average_distance = self._dist_loss(topk=topk,prop=prop)
            # self.average_distance = self._cos_loss(topk=topk,prop=prop)
            # alpha = 1
            # self.total_loss = loss + alpha * self.average_distance
            self.total_loss = loss #+ alpha * tf.math.divide(1,tf.math.log(1+self.average_distance))
        else:
            self.total_loss = 0

    def _cos_loss(self,topk,prop):
        diff = []
        for i in range(topk):  # calculate div
            for j in range(topk):
                x1 = tf.reduce_sum(tf.multiply(prop[i],prop[j]))
                x2 = tf.sqrt(tf.reduce_sum(tf.square(prop[i])))
                x3 = tf.sqrt(tf.reduce_sum(tf.square(prop[j])))
                res = x1/(x2*x3)
                diff.append(res)
        average_distance = tf.reduce_mean(diff)
        return average_distance

    def _dist_loss(self,topk,prop):
        diff = []
        for i in range(topk):  # calculate div
            for j in range(topk):
                res = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(prop[i], prop[j])))
                diff.append(res)
        average_distance = tf.reduce_mean(diff)
        return average_distance






class Fine_Tuning_BiLstm_Model_Test(object):

    def __init__(self, config,input,label):
        self.TimeStep = config.TimeStep
        self.input = config.input
        self.batchsize = config.batchsize
        self.hidden_unit = config.hidden_unit
        self.learning_rate = config.learning_rate
        self.epoch = config.epoch
        self.projection_dim  = config.projection_dim
        self.token_size = config.token_size
        self.n_negative_samples_batch = config.n_negative_samples_batch
        # self.weight = {
        #     'in': tf.Variable(tf.random_normal([1, self.projection_dim]),name="W"),
        # }
        # self.bias = {
        #     'in': tf.Variable(tf.random_normal([1]),name='bias'),
        # }
        self.is_Training = config.is_Training
        self.is_Diversity = config.is_Diversity
        self.initial_state = []
        self.final_state = []

        self._build(input,label)

    def _build(self,input,label):

        """
        双向LSTM模型来对图像进行分类
        :return:
        """
        '''
        LSTM在进行对序列数据进行处理的时候，需要先将其转化为满足网络的格式[batch，timestep,features]
        '''

        fw_Cell = []
        bw_Cell = []
        # 进行的多层双向神经网络
        for i in range(2):
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit,cell_clip=3,num_proj=self.projection_dim,proj_clip=3)
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit,cell_clip=3,num_proj=self.projection_dim,proj_clip=3)
            if i != 0:
                fw_lstm_cell = tf.nn.rnn_cell.ResidualWrapper(fw_lstm_cell)
                bw_lstm_cell = tf.nn.rnn_cell.ResidualWrapper(bw_lstm_cell)
            if self.is_Training:
                fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell,
                                                          input_keep_prob=0.9)
                bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell,
                                                          input_keep_prob=0.9)
            fw_Cell.append(fw_lstm_cell)
            bw_Cell.append(bw_lstm_cell)



        stack_lstm_fw = tf.nn.rnn_cell.MultiRNNCell(cells=fw_Cell)
        stack_lstm_bw = tf.nn.rnn_cell.MultiRNNCell(cells=bw_Cell)

        self.output = []
        with tf.control_dependencies([input[0]]):
            with tf.variable_scope('RNN_0'):
                self.initial_state.append(stack_lstm_fw.zero_state(self.batchsize, dtype=tf.float32))
                _lstm_output_unpacked_fw, final_state_fw = tf.nn.static_rnn(stack_lstm_fw,inputs=tf.unstack(input[0],axis=1),initial_state=self.initial_state[0],dtype=tf.float32)
                # if self.is_Training is False:
                self.output.append(_lstm_output_unpacked_fw)

        with tf.control_dependencies([input[1]]):
            with tf.variable_scope('RNN_1'):
                self.initial_state.append(stack_lstm_bw.zero_state(self.batchsize, dtype=tf.float32))
                _lstm_output_unpacked_bw, final_state_bw = tf.nn.static_rnn(stack_lstm_bw, inputs=tf.unstack(input[1],axis=1),initial_state=self.initial_state[-1],dtype=tf.float32)
                # if self.is_Training is False:
                self.output.append(_lstm_output_unpacked_bw)

        self.final_state.append(final_state_fw)
        self.final_state.append(final_state_bw)

        # lstm_output_flat_fw=tf.reshape(tf.stack(_lstm_output_unpacked_fw[-1], axis=1),[-1, self.projection_dim])
        # lstm_output_flat_bw = tf.reshape(tf.stack(_lstm_output_unpacked_bw[-1], axis=1), [-1, self.projection_dim])

        lstm_output_flat_fw = tf.concat((_lstm_output_unpacked_fw[1],_lstm_output_unpacked_fw[-1]),axis=1)
        lstm_output_flat_bw = tf.concat((_lstm_output_unpacked_bw[1],_lstm_output_unpacked_bw[-1]),axis=1)

        lstm_output = []

        if self.is_Training:
            lstm_output.append(tf.nn.dropout(lstm_output_flat_fw,0.9))
            lstm_output.append(tf.nn.dropout(lstm_output_flat_bw, 0.9))
        else:
            lstm_output.append(lstm_output_flat_fw)
            lstm_output.append(lstm_output_flat_bw)

        # input0 = tf.layers.batch_normalization(lstm_output[0],training=self.is_Training)
        # input1 = tf.layers.batch_normalization(lstm_output[-1], training=self.is_Training)
        dense_out = []

        with tf.variable_scope('ranking'):
            dense_input = tf.concat([lstm_output[0], lstm_output[-1]], axis=1)
            dense_input = tf.layers.batch_normalization(dense_input, training=self.is_Training)
            ranking_out = tf.layers.dense(inputs=dense_input,activation='relu',units=4096)
            if self.is_Training:
                ranking_out = tf.nn.dropout(ranking_out,0.9)
            ranking_out = tf.layers.dense(inputs=ranking_out,activation='relu',units=4096)
            dense_out.append(ranking_out)
            self.loss_input = tf.layers.dense(dense_out[0], activation=None, units=1)

        with tf.variable_scope('diversity'):
            dense_input = self.output[0][1]
            i=4
            while i < len(self.output[0]):
                dense_input += self.output[0][i]
                i += 3
            dense_input = tf.layers.batch_normalization(dense_input, training=self.is_Training)
            diversity_out = tf.layers.dense(inputs=dense_input,activation='relu',units=4096)
            if self.is_Training:
                diversity_out = tf.nn.dropout(diversity_out,0.9)
            diversity_out = tf.layers.dense(inputs=diversity_out,activation='relu',units=4096)
            dense_out.append(diversity_out)
            self.loss_input_diversity = tf.layers.dense(dense_out[-1], activation=None, units=1)
            # fine_tuning_out_fw = tf.layers.dense(inputs=input0, activation='relu', units=4096)
            # fine_tuning_out_bw = tf.layers.dense(inputs=input1, activation='relu', units=4096)
            # if self.is_Training:
            #     fine_tuning_out_fw = tf.nn.dropout(fine_tuning_out_fw, 0.9)
            #     fine_tuning_out_bw = tf.nn.dropout(fine_tuning_out_bw, 0.9)
            # fine_tuning_out_fw = tf.layers.dense(inputs=fine_tuning_out_fw, activation='relu', units=4096)
            # fine_tuning_out_bw = tf.layers.dense(inputs=fine_tuning_out_bw, activation='relu', units=4096)

        # fine_tuning_out_fw_diversity = tf.layers.dense(inputs=input0, activation='relu', units=4096)
        # fine_tuning_out_bw_diversity = tf.layers.dense(inputs=input1, activation='relu', units=4096)
        #
        # if self.is_Training:
        #     fine_tuning_out_fw = tf.nn.dropout(fine_tuning_out_fw,0.9)
        #     fine_tuning_out_bw = tf.nn.dropout(fine_tuning_out_bw,0.9)
        #     fine_tuning_out_fw_diversity = tf.nn.dropout(fine_tuning_out_fw_diversity,0.9)
        #     fine_tuning_out_bw_diversity = tf.nn.dropout(fine_tuning_out_bw_diversity,0.9)
        #
        # fine_tuning_out_fw = tf.layers.dense(inputs=fine_tuning_out_fw, activation='relu', units=4096)
        # fine_tuning_out_bw = tf.layers.dense(inputs=fine_tuning_out_bw, activation='relu', units=4096)
        #
        # fine_tuning_out_fw_diversity = tf.layers.dense(inputs=fine_tuning_out_fw_diversity, activation='relu', units=4096)
        # fine_tuning_out_bw_diversity = tf.layers.dense(inputs=fine_tuning_out_bw_diversity, activation='relu', units=4096)
        #
        # concat_input = tf.concat([fine_tuning_out_fw,fine_tuning_out_bw],axis=1)
        # concat_input_diversity = tf.concat([fine_tuning_out_fw_diversity, fine_tuning_out_bw_diversity], axis=1)


        self._fine_tuning_loss_top_k(int(lstm_output_flat_bw.shape[0]),label)
        #self._build_loss(lstm_output,label)
        # return tf.add(tf.matmul(output, self.weight['out']), self.bias['out'])  # 全连接层进行输出

    def _fine_tuning_loss_top_k(self,topk,label):
        if self.is_Training:
            labels = tf.reshape(label[0], [-1, 1])
            alpha = 1
            if self.is_Diversity:
                diver = []
                for i in range(self.batchsize):
                    prop = []
                    j = 1
                    while j < self.batchsize * 3:
                        prop.append(self.output[0][j][i])
                        j+=3
                    prop = tf.nn.l2_normalize(prop, axis=1)

                    # self.average_distance = self._dist_loss(topk=topk,prop=prop)
                    average_distance = self._cos_loss(topk=self.batchsize, prop=prop)
                    diver.append(average_distance)

                loss_div = tf.losses.mean_squared_error(labels=tf.reshape(diver,shape=[-1,1]),predictions=self.loss_input_diversity)
                self.total_loss = loss_div
            else:
                loss = tf.losses.mean_squared_error(labels=labels, predictions=self.loss_input)
                self.total_loss = loss
            # self.total_loss = loss #+ alpha * tf.math.divide(1,tf.math.log(1+self.average_distance))
        else:
            self.total_loss = 0

    def _cos_loss(self,topk,prop):
        diff = []
        for i in range(topk):  # calculate div
            for j in range(topk):
                x1 = tf.reduce_sum(tf.multiply(prop[i],prop[j]))
                x2 = tf.sqrt(tf.reduce_sum(tf.square(prop[i])))
                x3 = tf.sqrt(tf.reduce_sum(tf.square(prop[j])))
                res = x1/(x2*x3)
                diff.append(res)
        average_distance = tf.reduce_mean(diff)
        return average_distance

    def _dist_loss(self,topk,prop):
        diff = []
        for i in range(topk):  # calculate div
            for j in range(topk):
                res = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(prop[i], prop[j])))
                diff.append(res)
        average_distance = tf.reduce_mean(diff)
        return average_distance




class Fine_Tuning_KGE_Model(object):


    def __init__(self, config,input,label):
        self.TimeStep = config.TimeStep
        self.input = config.input
        self.batchsize = config.batchsize
        self.hidden_unit = config.hidden_unit
        self.learning_rate = config.learning_rate
        self.epoch = config.epoch
        self.projection_dim  = config.projection_dim
        self.token_size = config.token_size
        self.n_negative_samples_batch = config.n_negative_samples_batch
        # self.weight = {
        #     'in': tf.Variable(tf.random_normal([1, self.projection_dim]),name="W"),
        # }
        # self.bias = {
        #     'in': tf.Variable(tf.random_normal([1]),name='bias'),
        # }
        self.is_Training = config.is_Training
        self.is_Diversity = config.is_Diversity
        self.initial_state = []
        self.final_state = []

        self._build(input,label)

    def _build(self, input, label):

        #batchsize * vecsize or (batchsize * len(tri)) * vecsize

        dense_out = []
        input = input[0]
        self.output = input
        with tf.variable_scope('ranking'):
            dense_input = tf.reshape(input, shape=[-1,900])
            dense_input = tf.layers.batch_normalization(dense_input, training=self.is_Training)
            ranking_out = tf.layers.dense(inputs=dense_input, activation='relu', units=4096)
            if self.is_Training:
                ranking_out = tf.nn.dropout(ranking_out, 0.9)
            ranking_out = tf.layers.dense(inputs=ranking_out, activation='relu', units=4096)
            dense_out.append(ranking_out)
            self.loss_input = tf.layers.dense(dense_out[0], activation=None, units=1)

        with tf.variable_scope('diversity'):
            i = 1
            if self.is_Diversity:
                dense_input = tf.reduce_sum(input[0], 0, keep_dims=True)
                i = 1
                while i < self.batchsize:
                    suminp = tf.reduce_sum(input[i],0,keep_dims=True)
                    dense_input = tf.concat((dense_input,suminp),axis=0)
                    i+=1
            else:
                dense_input = input
            dense_input = tf.layers.batch_normalization(dense_input, training=self.is_Training)
            diversity_out = tf.layers.dense(inputs=dense_input, activation='relu', units=4096)
            if self.is_Training:
                diversity_out = tf.nn.dropout(diversity_out, 0.9)
            diversity_out = tf.layers.dense(inputs=diversity_out, activation='relu', units=4096)
            dense_out.append(diversity_out)
            self.loss_input_diversity = tf.layers.dense(dense_out[-1], activation=None, units=1)

        self._fine_tuning_loss_top_k(self.batchsize, label)

    def _fine_tuning_loss_top_k(self, topk, label):
        if self.is_Training:
            labels = tf.reshape(label[0], [-1, 1])
            alpha = 1
            if self.is_Diversity:
                diver = []
                for i in range(self.batchsize):
                    prop = self.output[i]

                    # self.average_distance = self._dist_loss(topk=topk,prop=prop)
                    average_distance = self._cos_loss(topk=self.batchsize, prop=prop)
                    diver.append(average_distance)

                loss_div = tf.losses.mean_squared_error(labels=tf.reshape(diver, shape=[-1, 1]),
                                                        predictions=self.loss_input_diversity)
                self.total_loss = loss_div
            else:
                loss = tf.losses.mean_squared_error(labels=labels, predictions=self.loss_input)
                self.total_loss = loss
            # self.total_loss = loss #+ alpha * tf.math.divide(1,tf.math.log(1+self.average_distance))
        else:
            self.total_loss = 0

    def _cos_loss(self, topk, prop):
        diff = []
        for i in range(topk):  # calculate div
            for j in range(topk):
                x1 = tf.reduce_sum(tf.multiply(prop[i], prop[j]))
                x2 = tf.sqrt(tf.reduce_sum(tf.square(prop[i])))
                x3 = tf.sqrt(tf.reduce_sum(tf.square(prop[j])))
                res = x1 / (x2 * x3)
                diff.append(res)
        average_distance = tf.reduce_mean(diff)
        return average_distance

    def _dist_loss(self, topk, prop):
        diff = []
        for i in range(topk):  # calculate div
            for j in range(topk):
                res = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(prop[i], prop[j])))
                diff.append(res)
        average_distance = tf.reduce_mean(diff)
        return average_distance














