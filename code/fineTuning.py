from code.bilmModel import Fine_Tuning_BiLstm_Model_Test
import numpy as np
import tensorflow as tf
import os
from code.BatchGenerator import FineTuningBatchGenerator
from code.preTrain import Config
import time



def fine_tuning_model_ranking(config,FTBG,saveID):
    config.TimeStep = 3
    X_fw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_fw = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    X_bw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_bw = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    with tf.device('/cpu:0'):
        lr = config.learning_rate
        with tf.device('/gpu:0'):
            with tf.variable_scope('lm'):
                lstm_model = Fine_Tuning_BiLstm_Model_Test(config,input=[X_fw,X_bw],label=[Y_fw,Y_bw])
                loss = lstm_model.total_loss
                opt = tf.train.AdamOptimizer()
                train_op = opt.minimize(loss)
        init = tf.initialize_all_variables()

    all_training_variable = [v for v in tf.trainable_variables()]
    not_restore_variable = []
    for li in all_training_variable:
        if "RNN" not in li.name:
            not_restore_variable.append(li)
    for li in not_restore_variable:
        all_training_variable.remove(li)
    saver = tf.train.Saver(var_list=all_training_variable)
    tfConfig = tf.ConfigProto(allow_soft_placement=True)
    tfConfig.gpu_options.allow_growth = True
    save_path = "./ckpt/fed/" + str(saveID)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    with tf.Session(config=tfConfig) as sess:
        sess.run(init)
        checkpoint = tf.train.get_checkpoint_state("./ckpt/bilm")
        saver.restore(sess, checkpoint.model_checkpoint_path)
        saver = tf.train.Saver(max_to_keep=1)
        start = time.time()
        for batchcount in range(0, len(FTBG.training_ids) * config.epoch):
            instance,label = FTBG.generate_fine_tuning_data(batchSize=config.batchsize, topkSize=config.batchsize, sampleSize=5)
            feed_dict = {X_fw: instance[0], X_bw: instance[-1], Y_fw: label[0], Y_bw: label[-1]}
            ret = sess.run([train_op,lstm_model.total_loss,lstm_model.output],feed_dict=feed_dict)
            print(str(ret[1]))

            if (batchcount+1) % 10000 == 0:
                saver.save(sess, save_path + "/bilm", global_step=batchcount)
                # saver.save(sess, "../AllCkpt/imdb_sample_ckpt_top5/bilm", global_step=batchcount)
                print("  time : " + str(time.time() - start))
                start = time.time()

def fine_tuning_model_diversity(config,FTBG,saveID):
    config.TimeStep  = config.batchsize * config.TimeStep

    X_fw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_fw = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    X_bw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_bw = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    with tf.device('/cpu:0'):
        lr = config.learning_rate
        with tf.device('/gpu:0'):
            with tf.variable_scope('lm'):
                lstm_model = Fine_Tuning_BiLstm_Model_Test(config,input=[X_fw,X_bw],label=[Y_fw,Y_bw])
                atv = [v for v in tf.trainable_variables()]
                selected_trained_variable = []
                for val in atv:
                    if "diversity" in val.name:
                        selected_trained_variable.append(val)
                loss = lstm_model.total_loss
                opt = tf.train.AdamOptimizer()
                train_op = opt.minimize(loss,var_list=selected_trained_variable)
                # train_op = opt.minimize(loss)
        init = tf.initialize_all_variables()



    all_training_variable = [v for v in tf.trainable_variables()]
    not_restore_variable = []
    for li in all_training_variable:
        if "diversity" in li.name:
            not_restore_variable.append(li)
    for li in not_restore_variable:
        all_training_variable.remove(li)
    saver = tf.train.Saver(var_list=all_training_variable, max_to_keep=1)
    tfConfig = tf.ConfigProto(allow_soft_placement=True)
    tfConfig.gpu_options.allow_growth = True
    save_path = "/ckpt/fed/" + str(saveID)
    with tf.Session(config=tfConfig) as sess:
        sess.run(init)
        checkpoint = tf.train.get_checkpoint_state(save_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
        saver = tf.train.Saver(max_to_keep=1)
        start = time.time()
        instance = []
        label = []
        for batchcount in range(0, len(FTBG.training_ids) * config.epoch):
            ins,lab = FTBG.generate_fine_tuning_data(batchSize=config.batchsize, topkSize=config.batchsize, sampleSize=5)
            if len(instance) > 0:
                instance[0] = np.concatenate((instance[0],np.reshape(ins[0],[1,-1,300])))
                instance[-1] = np.concatenate((instance[-1],np.reshape(ins[-1],[1,-1,300])))
            else:
                instance.append(np.reshape(ins[0],[1,-1,300]))
                instance.append(np.reshape(ins[-1],[1,-1,300]))
                label = lab
            if (batchcount+1) % config.batchsize == 0:
                feed_dict = {X_fw: instance[0], X_bw: instance[-1], Y_fw: label[0], Y_bw: label[-1]}
                ret = sess.run([train_op,lstm_model.total_loss,lstm_model.output,lstm_model.loss_input_diversity],feed_dict=feed_dict)
                print(str(ret[1]))
                instance.clear()
                label.clear()

            if (batchcount+1) % 10000 == 0:

                saver.save(sess, save_path + "/bilm", global_step=batchcount)
                print("  time : " + str(time.time() - start))
                start = time.time()

if __name__ == "__main__":
    FTBG = FineTuningBatchGenerator()
    config = Config(learning_rate=0.2, batchsize=5, input=300, timestep=3, projection_dim=300,
                    epoch=20, hidden_unit=4096,n_negative_samples_batch=8192,token_size=0,is_Training = True)
    config.is_Diversity = False
    fine_tuning_model_ranking(config,FTBG,0)
    tf.reset_default_graph()
    config.is_Diversity = True
    fine_tuning_model_diversity(config,FTBG,0)
    tf.reset_default_graph()
