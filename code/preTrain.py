import tensorflow as tf
from code.bilmModel import BiLstm_Model
from code.BatchGenerator import BatchGenerator
import time

'''
'''



class Config():
    """
    配置文件的类
    """

    def __init__(self, input, timestep, batchsize, hidden_unit, learning_rate, epoch, projection_dim,n_negative_samples_batch,token_size,is_Training,is_Diveristy=False):
        self.TimeStep = timestep
        self.input = input
        self.batchsize = batchsize
        self.hidden_unit = hidden_unit
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.projection_dim = projection_dim
        self.is_Training = is_Training
        self.token_size = token_size
        self.n_negative_samples_batch = n_negative_samples_batch
        self.is_Diversity = is_Diveristy

class BatchConfig():
    def __init__(self,batchsize,timestep,randompathcount):
        self.batchsize = batchsize
        self.timestep = timestep
        self.randompathcount = randompathcount

def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    # wrapper around tf.clip_by_global_norm that also does summary ops of norms

    # compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tf.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    # clip
    clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tf.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

    return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, options, do_summaries, global_step):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val
        if do_summaries:
            clipped_tensors, g_norm, so = clip_by_global_norm_summary(
                grad_tensors, scaled_val, name, vv)
        else:
            so = []
            clipped_tensors, g_norm = tf.clip_by_global_norm(
                grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret, so

    all_clip_norm_val = 10.0
    ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret, summary_ops


def average_gradients(tower_grads, batch_size, options):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.

        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))

    return average_grads

def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
      values, new_index_positions,
      tf.shape(unique_indices)[0])
    return (summed_values, unique_indices)



def trainModel(BG,config):
    X_fw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_fw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep])
    X_bw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_bw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep])

    with tf.device('/cpu:0'):

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        lr = config.learning_rate
        opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                        initial_accumulator_value=1.0)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        with tf.device('/gpu:0'):
            with tf.variable_scope('lm'):
                lstm_model = BiLstm_Model(config, input=[X_fw, X_bw], label=[Y_fw, Y_bw])
                loss = lstm_model.total_loss
                # grads = opt.minimize(lstm_model.total_loss)
                grads = opt.compute_gradients(
                        loss * config.TimeStep,
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
        grads = average_gradients([grads],config.batchsize,config)
        grads, norm_summary_ops = clip_grads(grads, config, True, global_step)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        init = tf.initialize_all_variables()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,gpu_options=gpu_options)) as sess:
        sess.run(init)
        init_state_tensor = []
        final_state_tensor = []
        init_state_tensor.extend(lstm_model.initial_state)
        final_state_tensor.extend(lstm_model.final_state)
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("./ckpt")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find saved Model")
        fw_data,bw_data = BG.generateData(batchsize=config.batchsize, timestep=config.TimeStep, vecsize=config.input)
        initial_state_value = sess.run(init_state_tensor,feed_dict={X_fw:fw_data[0],X_bw:bw_data[0],Y_fw:fw_data[-1],Y_bw:bw_data[-1]})

        start = time.time()
        for batchcount in range(0,(BG.all_train_token//config.batchsize)*config.epoch):
            fw_data, bw_data = BG.generateData(batchsize=config.batchsize, timestep=config.TimeStep,
                                               vecsize=config.input)
            feed_dict = {X_fw: fw_data[0], X_bw: bw_data[0], Y_fw: fw_data[-1], Y_bw: bw_data[-1],
                             init_state_tensor[0]: initial_state_value[0], init_state_tensor[1]: initial_state_value[1]}
            ret = sess.run([train_op,lstm_model.total_loss] + lstm_model.output + final_state_tensor,feed_dict=feed_dict)
            print(str(ret[1]))
            initial_state_value = ret[4:]
            if batchcount % 1000 == 0:
                saver.save(sess, "./ckpt/bilm", global_step=batchcount)
                print("  time : " + str(time.time() - start))
                start = time.time()


            # feed_dict = {X_fw: fw_data[0], X_bw: bw_data[0], Y_fw: fw_data[-1], Y_bw: bw_data[-1]}
            # ret = sess.run(lstm_model.output,feed_dict=feed_dict)

def useModel(BG,config):
    X_fw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_fw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep])
    X_bw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_bw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep])

    with tf.device('/cpu:0'):
        lr = config.learning_rate
        with tf.device('/gpu:0'):
            with tf.variable_scope('lm'):
                lstm_model = BiLstm_Model(config, input=[X_fw, X_bw], label=[Y_fw, Y_bw])
        init = tf.initialize_all_variables()

    all_training_variable = [v for v in tf.trainable_variables()]
    saver = tf.train.Saver(var_list=all_training_variable)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        sess.run(init)
        checkpoint = tf.train.get_checkpoint_state("./ckpt")
        saver.restore(sess, checkpoint.model_checkpoint_path)

        fw_data,bw_data = BG.generateData(batchsize=config.batchsize, timestep=config.TimeStep, vecsize=config.input)
        feed_dict = {X_fw: fw_data[0], X_bw: bw_data[0], Y_fw: fw_data[-1], Y_bw: bw_data[-1]}
        ret = sess.run(lstm_model.output,feed_dict=feed_dict)
    return ret,sess




if __name__ == "__main__":

    # 定义一个配置类的对象
    batchconfig = BatchConfig(batchsize=512,timestep=7,randompathcount=1)
    BG = BatchGenerator(batchconfig)

    config = Config(learning_rate=0.2, batchsize=batchconfig.batchsize, input=300, timestep=8, projection_dim=300,
                    epoch=60, hidden_unit=4096,n_negative_samples_batch=8192,token_size=BG.token_size,is_Training = True)

    trainModel(BG,config)
    # ret = useModel(BG,config)
