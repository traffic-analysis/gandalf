import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
import os.path as opth
import os
from sklearn.utils import shuffle
import argparse
HOME = os.path.expanduser('~')
parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "7";
layers = tf.keras.layers


def define_generator():

    def conv1d_block(filters, upsample=True, activation=tf.nn.relu, index=0):
        if upsample:
            model.add(layers.UpSampling1D(name="UpSampling" + str(index), size=2))
        model.add(layers.Conv1D(filters=filters, kernel_size=5, padding='same', name="Conv1D" + str(index),
            activation=activation))
        model.add(layers.BatchNormalization())

    model = tf.keras.models.Sequential(name="Generator")
    model.add(layers.Dense(int(316), activation=tf.nn.relu, name="NoiseToSpatial")) #50
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((int(316),1)))

    conv1d_block(filters=512, upsample=True, index=0)
    conv1d_block(filters=512, upsample=True, index=1)
    conv1d_block(filters=256, upsample=True, index=2)
    conv1d_block(filters=256, upsample=True, index=3)
    conv1d_block(filters=128, upsample=False, index=4)
    conv1d_block(filters=128, upsample=False, index=5)
    conv1d_block(filters=64, upsample=False, index=6)
    conv1d_block(filters=64, upsample=False, index=7)
    conv1d_block(filters=1, upsample=False, activation=tf.nn.tanh, index=8)
    return model


class Discriminator:

    def __init__(self):
        self.tail = self._define_tail()
        self.head = self._define_head()

    def _define_tail(self, name="Discriminator"):
        feature_model = tf.keras.models.Sequential(name=name)

        def conv1d_dropout(filters, strides, index=0):
            suffix = str(index)
            feature_model.add(layers.Conv1D(filters=filters, strides=strides, name="Conv{}".format(suffix), padding='same',
                kernel_size=5, activation=tf.nn.leaky_relu))
            feature_model.add(layers.Dropout(name="Dropout{}".format(suffix), rate=0.3))

        conv1d_dropout(filters=32, strides=2, index=5)
        conv1d_dropout(filters=32, strides=2, index=6)
        conv1d_dropout(filters=64, strides=2, index=0)
        conv1d_dropout(filters=64, strides=2, index=1)
        conv1d_dropout(filters=128, strides=2, index=2)
        conv1d_dropout(filters=128, strides=2, index=3)
        conv1d_dropout(filters=256, strides=1, index=4) #64
        conv1d_dropout(filters=256, strides=1, index=7)

        feature_model.add(layers.Flatten(name="Flatten")) # This is feature layer for FM loss !!
        return feature_model

    def _define_head(self):
        head_model = tf.keras.models.Sequential(name="DiscriminatorHead")

        head_model.add(layers.Dense(units=2048, activation='relu'))
        head_model.add(layers.Dropout(rate=0.5))
        head_model.add(layers.Dense(units=2048, activation='relu'))
        head_model.add(layers.Dropout(rate=0.5))
        head_model.add(layers.Dense(units=1024, activation='relu'))
        head_model.add(layers.Dropout(rate=0.5))
        head_model.add(layers.Dense(units=512, activation='relu'))
        head_model.add(layers.Dropout(rate=0.5))

        head_model.add(layers.Dense(units=args.num_classes, activation=None, name="Logits"))
        return head_model

    @property
    def trainable_variables(self):
        return self.tail.trainable_variables + self.head.trainable_variables

    def __call__(self, x, *args, **kwargs):
        features = self.tail(x, *args, **kwargs)
        print(features.shape)
        return self.head(features, *args, **kwargs), features




def return_logits(logits, labels):
    return tf.nn.softmax(logits=logits, dim=-1)

def return_labels(logits, labels):
    return labels

def main(args):
    global best_tpr
    global best_pre
    global pre_pair
    global tpr_pair
    with tf.Graph().as_default():
        print("Input data preprocessing...")
        with tf.name_scope("DataPreprocess"):
            r_train = 500.0 / 6.0
            r_test = 100.0 / 6.0
            nClass = 100  # 100
            mon_instance = 2498.0  # 300.0
            unmon_instance = 50000
            dim = 5000#4096  # 3969#784
            unmon_end = 399989
            b_label = False
            with tf.device('/cpu:0'):
                (train_x, train_y, test_x_data, test_y_data) = split_awf(r_train, r_test, nClass,
                                                                    mon_instance, unmon_instance, dim, b_label, unmon_end)

            print('train_x',train_x.shape)

            def reshape_and_scale(x, img_shape=(-1, dim, 1)):
                return x.reshape(img_shape).astype(np.float32)

            train_x = reshape_and_scale(train_x)

            X, y = shuffle(train_x, train_y)
            print(X.shape)
            print(y.shape)

        print("Setup the input pipeline...")
        with tf.name_scope("InputPipeline"):
            train_x_labeled, train_y_labeled = [], []
            for i in range(args.num_classes):
                train_x_labeled.append(X[y == i][:args.num_labeled_examples])
                if i == 0:  # unmonitored traces
                    train_x_labeled.append(X[y == i][:args.num_labeled_examples * (args.num_classes - 1)])
            train_x_labeled_data = np.concatenate(train_x_labeled)
            labeled_X = tf.placeholder(tf.float32, shape=[None, dim, 1])
            labeled_y = tf.placeholder(tf.int64, shape=[None])

            test_X = tf.placeholder(tf.float32, shape=[None, dim, 1])
            test_y = tf.placeholder(tf.int64, shape=[None])

            train_labeled_dataset = tf.data.Dataset.from_tensor_slices((labeled_X, labeled_y))\
                .shuffle(buffer_size=len(train_x_labeled_data))\
                .repeat()
            train_labeled_dataset = train_labeled_dataset.batch(args.batch_size)
            iterator_labeled = train_labeled_dataset.make_initializable_iterator()
            traces_lab, labels_lab = iterator_labeled.get_next()

            test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))\
                .repeat()
            test_dataset = test_dataset.batch(args.batch_size)
            iterator_test = test_dataset.make_initializable_iterator()
            traces_test, labels_test = iterator_test.get_next()

        with tf.name_scope("BatchSize"):
            batch_size_tensor = tf.shape(traces_lab)[0]

        print("Get the noise vectors")
        z, z_perturbed = define_noise(batch_size_tensor)

        print("Generate traces from noise vector")
        with tf.name_scope("Generator"):
            g_model = define_generator()
            traces_fake_perturbed = g_model(z_perturbed)

        with tf.name_scope("Discriminator") as discriminator_scope:
            d_model = Discriminator()
            logits_fake_perturbed, _ = d_model(traces_fake_perturbed, training=True)
            logits_train, _ = d_model(traces_lab, training=False)


        with tf.name_scope(discriminator_scope):
            with tf.name_scope("Test"):
                logits_test, _ = d_model(traces_test, training=False)
                test_logit_op = return_logits(logits_test, labels_test)
                test_label_op = return_labels(logits_test, labels_test)


        print("Run testing...")
        best_tpr = 0.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore (sess, save_path=args.model_root + args.model_path)
            #saver.restore(sess,save_path='/data/seoh/ssl_saved_model/NEWWF20open_pre_1d_epoch102_tpr0.82_pre0.48.ckpt') # WF optimized model

            #test = np.load('/data/seoh/ow_wf_test.npz',allow_pickle=True)
            test = np.load (args.data_root + args.test_path, allow_pickle=True)

            #test_x_data = reshape_and_scale(test['x'])
            test_x_data = reshape_and_scale(test['X'])
            test_y_data = test['y']

            test_mon_X = test_x_data[test_y_data != 0]
            test_unmon_X = test_x_data[test_y_data == 0]
            test_mon_y = test_y_data[test_y_data != 0]
            test_unmon_y = test_y_data[test_y_data == 0]
            test_x_data = np.concatenate((test_mon_X, test_unmon_X[:int(args.back_size)]), axis=0)
            test_y_data = np.concatenate((test_mon_y, test_unmon_y[:int(args.back_size)]), axis=0)

            steps_per_test = test_x_data.shape[0] // args.batch_size

            sess.run(iterator_test.initializer, feed_dict={test_X: test_x_data, test_y: test_y_data})
            pred_list, true_list= [], []
            for _ in range(steps_per_test):
                try:
                    pred, true = sess.run(
                        [test_logit_op, test_label_op])
                    pred_list.append(pred)
                    true_list.append(true)
                except tf.errors.OutOfRangeError:
                    print('End of sequence.')
                    pass

            logits=[]
            labels=[]
            for i in range(len(pred_list)):
                for j in range(args.batch_size):
                    logits.append(pred_list[i][j])
                    labels.append(true_list[i][j])
            logits = np.array(logits)
            labels = np.array(labels)
            print('logits', logits.shape)
            print('labels',labels.shape)
            pred_mon = logits[labels != 0]
            pred_unmon = logits[labels == 0]
            print('pred_mon',pred_mon.shape)
            print('pred_unmon',pred_unmon.shape)

            unmon_label = 0
            # Test with Monitored testing instances
            for threshold in [0.2, 0.24210526315789474, 0.28421052631578947, 0.3263157894736842, 0.368421052631579, 0.4105263157894737,
                              0.45263157894736844, 0.49473684210526314, 0.5368421052631579, 0.5789473684210527, 0.6210526315789473,
                              0.6631578947368422,
                              0.9778947368421054,
                              0.9888888888, 0.995,
                              0.998, 0.999995, 0.999999995, 0.999999999, 1.0]:
                TP, FP, TN, FN, total = 0, 0, 0, 0, 0
                for s in range(len(pred_mon)):

                    #############
                    predict_prob = pred_mon[s]
                    #############
                    best_n = np.argmax(predict_prob)
                    if int(best_n) != unmon_label:  # predicted as "Monitored"
                        if predict_prob[best_n] >= threshold:
                            TP = TP + 1
                        else:
                            FN = FN + 1
                    else:
                        FN = FN + 1
                # Test with Unmonitored testing instances

                for s in range(len(pred_unmon)):
                    #############
                    predict_prob = pred_unmon[s]
                    #############
                    best_n = np.argmax(predict_prob)
                    if int(best_n) != unmon_label:  # predicted as "Monitored"
                        if predict_prob[best_n] >= threshold:
                            FP = FP + 1
                        else:
                            TN = TN + 1
                    else:
                        TN = TN + 1
                print('threshold', threshold)
                print('Pre: ', TP / (TP + FP))
                print('Recall: ',TP/len(pred_mon))

def split_awf(r_train, r_test, nClass, mon_instance, unmon_instance, dim, b_label, unmon_end):

    if b_label:
        print("It's binary classification!!!")

    mon_data = np.load(HOME+'/datasets/awf1.npz', allow_pickle=True)
    mon_x = mon_data['feature']

    unmon_data = np.load(HOME+'/datasets/tor_open_400000w.npz', allow_pickle=True)
    unmon_x = unmon_data['data']

    ## We need to uniformly random selection over each monitored class
    #print('mon_instance',mon_instance)
    #print('unmon_instance',unmon_instance)
    num_mtrain_instance = mon_instance * (r_train / (r_train + r_test))  ## number of monitored training instances for each class
    num_umtrain_instance = unmon_instance * (r_train / (r_train + r_test))  ## number of monitored training instances for each class

    mon_random = np.array(range(int(mon_instance)))
    np.random.shuffle(mon_random)

    mon_train_ins = mon_random[:int(num_mtrain_instance)] #1666
    mon_test_ins = mon_random[int(num_mtrain_instance):]
    #print('mon_test_ins', len(mon_test_ins))

    unmon_random = np.array(range(int(unmon_end)))
    np.random.shuffle(unmon_random)
    #print('unmon_random',len(unmon_random))

    unmon_train_ins = unmon_random[:int(num_umtrain_instance)]
    unmon_test_ins = unmon_random[int(num_umtrain_instance):]
    #print('unmon_test_ins', len(unmon_test_ins))

    # Due to the memory error, initialize np arrays here first
    train_feature = np.zeros((nClass*len(mon_train_ins)+len(unmon_train_ins), dim), dtype=int)
    test_feature = np.zeros((nClass*len(mon_test_ins)+len(unmon_test_ins),dim), dtype=int)
    #print('test_feature', len(test_feature))
    train_label = np.zeros((nClass*len(mon_train_ins)+len(unmon_train_ins),), dtype=int)
    test_label = np.zeros((nClass*len(mon_test_ins)+len(unmon_test_ins),), dtype=int)

    print(len(mon_train_ins)+len(unmon_train_ins))
    print(len(mon_test_ins)+len(unmon_test_ins))

    i = 0
    mon_instance = int(mon_instance)
    #print('Monitored training set partitioning...')
    #print(nClass)
    #print(len(mon_train_ins))
    for c in range(nClass):
        c=int(c)
        print(c)
        for instance in mon_train_ins:
            if not b_label:
                train_label[i] = c+1 # positive label is c+1

            else:
                train_label[i] = 0
            train_feature[i] = mon_x[(c*mon_instance)+instance][:dim]
            i += 1
    #print(i)
    #print('Monitored testing set partitioning...')
    j = 0
    for c in range(nClass):
        c = int(c)
        for instance in mon_test_ins:
            if not b_label:
                test_label[j]=c+1 # positive label is c+1

            else:
                test_label[j] = 0

            test_feature[j]=mon_x[(c*mon_instance)+instance][:dim]
            j += 1
    #print(j)

    #print('Unmonitored training set partitioning...')
    for instance in unmon_train_ins:
        train_feature[i]=unmon_x[instance][:dim]
        train_label[i]=0 # negative label is 0

        i += 1
    #print(i)
    #print('Unmonitored testing set partitioning...')

    for instance in unmon_test_ins:
        test_feature[j]=unmon_x[instance][:dim]
        test_label[j]=0 # negative label is 0

        j += 1

    #print(j)



    return train_feature, train_label, test_feature, test_label

def define_noise(batch_size_tensor):
    with tf.name_scope("LatentNoiseVector"):
        z = tfd.Normal(loc=0.0, scale=args.stddev).sample(
            sample_shape=(batch_size_tensor, args.z_dim_size))
        z_perturbed = z + tfd.Normal(loc=0.0, scale=args.stddev).sample(
            sample_shape=(batch_size_tensor, args.z_dim_size)) * 1e-5
    return z, z_perturbed



def _next_logdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    subdirs = [d for d in os.listdir(path) if opth.isdir(opth.join(path, d))]
    logdir = opth.join(path, "run" + str(len(subdirs)).zfill(4))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir


if __name__ == "__main__":
    parser.add_argument ('--batch_size', required=False, default=32)
    parser.add_argument ('--train_epochs', required=False, default=1000)
    parser.add_argument ('--lr', required=False, default=2e-4)
    parser.add_argument ('--stddev', required=False, default=1e-2)
    parser.add_argument ('--num_classes', required=False, default=101)
    parser.add_argument ('--z_dim_size', required=False, default=100)
    parser.add_argument ('--num_labeled_examples', required=False, default=20)
    parser.add_argument ('--back_size', required=False, default=360000)
    parser.add_argument ('--data_root', required=False, default=HOME)
    parser.add_argument ('--model_root', required=False, default=HOME)
    parser.add_argument ('--model_path', required=False, default='/ssl_saved_model/wfi_ow_paper/NEWWF20open_pre_1d_epoch102_tpr0.82_pre0.48.ckpt')
    parser.add_argument ('--test_path', required=False, default='/datasets/ow_wf_test_compressed.npz')
    parser.add_argument ('--man_reg', required=False, default=False)
    args = parser.parse_args ()
    main(args)
