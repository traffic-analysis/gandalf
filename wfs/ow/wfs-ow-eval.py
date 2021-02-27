import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
import os.path as opth
from keras.layers.advanced_activations import ELU
import os
from sklearn.utils import shuffle
import argparse
HOME = os.path.expanduser('~')
parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "4";
layers = tf.keras.layers


def define_generator():
    def conv1d_block(filters, upsample=True, activation=tf.nn.relu, index=0):
        if upsample:
            model.add(layers.UpSampling1D(name="UpSampling" + str(index), size=2))
        if activation == 'elu':
            model.add(layers.Conv1D(filters=filters, kernel_size=20, padding='same', name="Conv1D" + str(index),
                                    activation=ELU(alpha=0.1)))
        else:
            model.add(layers.Conv1D(filters=filters, kernel_size=20, padding='same', name="Conv1D" + str(index),
                                    activation=activation))
        model.add(layers.BatchNormalization())
        if (index <= 7) and (index >= 1):
            model.add(layers.Dropout(rate=0.3))

    model = tf.keras.models.Sequential(name="Generator")
    model.add(layers.Dense(int(316), activation=tf.nn.relu, name="NoiseToSpatial"))  # 50
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(int(316), activation=tf.nn.relu, name="NoiseToSpatial"))  # 50
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(int(316), activation=tf.nn.relu, name="NoiseToSpatial"))  # 50
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Reshape((int(316), 1)))

    conv1d_block(filters=256, upsample=True, index=0, activation='relu')
    conv1d_block(filters=256, upsample=True, index=1, activation='relu')
    conv1d_block(filters=128, upsample=True, index=2, activation='relu')
    conv1d_block(filters=128, upsample=True, index=3, activation='relu')
    conv1d_block(filters=64, upsample=False, index=4, activation='relu')
    conv1d_block(filters=64, upsample=False, index=5, activation='relu')
    conv1d_block(filters=32, upsample=False, index=6, activation='relu')
    conv1d_block(filters=32, upsample=False, index=7, activation='relu')
    conv1d_block(filters=1, upsample=False, activation=tf.nn.tanh, index=8)
    return model


class Discriminator:

    def __init__(self):
        self.tail = self._define_tail()
        self.head = self._define_head()

    def _define_tail(self, name="Discriminator"):
        feature_model = tf.keras.models.Sequential(name=name)

        def conv1d_dropout(filters, strides, index=0, act='relu'):
            suffix = str(index)
            if act == 'lrelu':
                feature_model.add(
                    layers.Conv1D(filters=filters, strides=strides, name="Conv{}".format(suffix), padding='same',
                                  kernel_size=20, activation=tf.nn.leaky_relu))
            else:
                feature_model.add(
                    layers.Conv1D(filters=filters, strides=strides, name="Conv{}".format(suffix), padding='same',
                                  kernel_size=20, activation=ELU(alpha=0.1)))
            feature_model.add(layers.Dropout(name="Dropout{}".format(suffix), rate=0.3))

        conv1d_dropout(filters=32, strides=4, index=0, act='lrelu')
        conv1d_dropout(filters=32, strides=4, index=1, act='lrelu')
        conv1d_dropout(filters=64, strides=4, index=2, act='lrelu')
        conv1d_dropout(filters=64, strides=4, index=3, act='lrelu')
        conv1d_dropout(filters=128, strides=2, index=4, act='lrelu')
        conv1d_dropout(filters=128, strides=2, index=5, act='lrelu')
        conv1d_dropout(filters=256, strides=1, index=6, act='lrelu')  # 64
        conv1d_dropout(filters=256, strides=1, index=7, act='lrelu')
        feature_model.add(layers.Flatten(name="Flatten"))  # This is feature layer for FM loss !!
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

        head_model.add(layers.Dense(units=args.num_classes, activation=None, name="Logits"))  # since 10 classes
        return head_model

    @property
    def trainable_variables(self):
        return self.tail.trainable_variables + self.head.trainable_variables

    def __call__(self, x, *args, **kwargs):
        features = self.tail(x, *args, **kwargs)
        print(features.shape)
        return self.head(features, *args, **kwargs), features



def count_pos(logits, labels):
    return tf.cast(tf.count_nonzero(labels), tf.float32)


def count_neg(logits, labels):
    return tf.cast(tf.count_nonzero(tf.math.less(tf.cast((labels - 1), tf.int64), 0)), tf.float32)


def return_logits(logits, labels):
    return tf.nn.softmax(logits=logits, dim=-1)

def return_labels(logits, labels):
    return labels


def main(args):
    with tf.Graph().as_default():
        print("Input data preprocessing...")
        with tf.name_scope("DataPreprocess"):
            dim = 5000
            subf_data = np.load(HOME+'/datasets/gdlf25_ipd.npz', allow_pickle=True)

            all_x = subf_data['X']
            all_y = subf_data['y']

            new_all_x = []
            # padding
            for x in all_x:
                new_all_x.append(np.pad(x[:dim], (0, dim - len(x[:dim])), 'constant'))
            all_x = np.array(new_all_x)
            # shuffling
            mon_x = all_x  # [all_y < 11]
            mon_y = all_y  # [all_y < 11]


            def reshape_and_scale(x, img_shape=(-1, dim, 1)):
                return x.reshape(img_shape).astype(np.float32)

            mon_x = reshape_and_scale(mon_x)

            X, y = shuffle(mon_x, mon_y)

        print("Setup the input pipeline...")
        with tf.name_scope("InputPipeline"):
            train_x_labeled, train_y_labeled = [], []
            for i in range(args.num_classes):
                train_x_labeled.append(X[y == i][:args.num_labeled_examples])

            train_x_labeled_data = np.concatenate(train_x_labeled)

            labeled_X = tf.placeholder(tf.float32, shape=[None, dim, 1])
            labeled_y = tf.placeholder(tf.int64, shape=[None])

            test_X = tf.placeholder(tf.float32, shape=[None, dim, 1])
            test_y = tf.placeholder(tf.int64, shape=[None])

            train_labeled_dataset = tf.data.Dataset.from_tensor_slices((labeled_X, labeled_y)) \
                .shuffle(buffer_size=len(train_x_labeled_data)) \
                .repeat()
            train_labeled_dataset = train_labeled_dataset.batch(args.batch_size)
            iterator_labeled = train_labeled_dataset.make_initializable_iterator()
            traces_lab, labels_lab = iterator_labeled.get_next()

            test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)) \
                .repeat()
            test_dataset = test_dataset.batch(args.batch_size)
            iterator_test = test_dataset.make_initializable_iterator()
            traces_test, labels_test = iterator_test.get_next()

        with tf.name_scope("BatchSize"):
            batch_size_tensor = tf.shape(traces_lab)[0]

        z, z_perturbed = define_noise(batch_size_tensor, args)

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

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore (sess, save_path=args.model_root + args.model_path)  # wfs-ow90.ckpt')
            print("...saved model restored successfully!")
            test = np.load (args.data_root + args.test_path, allow_pickle=True)
            def reshape_and_scale(x, img_shape=(-1, dim, 1)):
                return x.reshape(img_shape).astype(np.float32)
            test_x_data = reshape_and_scale(test['X'])
            test_y_data = test['y']
            steps_per_test = test_x_data.shape[0] // args.batch_size

            test_mon_X = test_x_data[test_y_data != 0]
            test_unmon_X = test_x_data[test_y_data == 0]
            test_mon_y = test_y_data[test_y_data != 0]
            test_unmon_y = test_y_data[test_y_data == 0]
            test_x_data = np.concatenate((test_mon_X, test_unmon_X[:int(args.back_size)]), axis=0)
            test_y_data = np.concatenate((test_mon_y, test_unmon_y[:int(args.back_size)]), axis=0)


            sess.run(iterator_test.initializer, feed_dict={test_X: test_x_data, test_y: test_y_data})
            pred_list, true_list = [], []
            for _ in range(steps_per_test):
                try:
                    accuracy_val, fp_val = sess.run([test_logit_op, test_label_op])
                    pred_list.append(accuracy_val)
                    true_list.append(fp_val)
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
                              0.7052631578947368, 0.7473684210526317,
                              0.7894736842105263,
                              0.831578947368421,
                              0.8736842105263158, 0.9157894736842105, 0.9578947368421054, #1.0]:
                              0.9778947368421054,
                              0.9888888888, 0.995, 0.998]:
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

def define_noise(batch_size_tensor, args):
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
    parser.add_argument ('--batch_size', required=False, default=16)
    parser.add_argument ('--train_epochs', required=False, default=1000)
    parser.add_argument ('--lr', required=False, default=2e-4)
    parser.add_argument ('--stddev', required=False, default=1e-2)
    parser.add_argument ('--num_classes', required=False, default=26)
    parser.add_argument ('--z_dim_size', required=False, default=100)
    parser.add_argument ('--num_labeled_examples', required=False, default=90)
    parser.add_argument ('--back_size', required=False, default=70000)
    parser.add_argument ('--data_root', required=False, default=HOME)
    parser.add_argument ('--model_root', required=False, default=HOME)
    parser.add_argument ('--model_path', required=False, default='/ssl_saved_model/wfs_ow_paper/NEW26_subfN90open_Nbest_pre_third.ckpt')
    parser.add_argument ('--test_path', required=False, default='/datasets/wfs_ow_test_paper.npz')
    parser.add_argument ('--man_reg', required=False, default=False)
    args = parser.parse_args ()
    main(args)
