import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
import os.path as opth
import tqdm
import os
from sklearn.utils import shuffle
from keras.layers.advanced_activations import ELU
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "6";
layers = tf.keras.layers
parser = argparse.ArgumentParser()
HOME = os.path.expanduser('~')



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

        head_model.add(layers.Dense(units=args.num_classes, activation=None, name="Logits"))
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

def fp(logits, labels):
    preds = tf.argmax(logits, axis=1)
    return tf.cast(tf.count_nonzero(tf.math.less(tf.cast(preds * (labels - 1), tf.int64), 0)), tf.float32)

def tp(logits, labels):
    preds = tf.argmax(logits, axis=1)
    return tf.cast(tf.count_nonzero(preds * labels), tf.float32)



def main(args):
    global best_pre
    best_pre = 0
    with tf.Graph().as_default():
        print("Input data preprocessing...")
        with tf.name_scope("DataPreprocess"):
            dim = 5000
            subf_data = np.load(args.data_root+'/datasets/gdlf25_ipd.npz', allow_pickle=True)
            all_x = subf_data['X']
            all_y = subf_data['y']


            new_all_x = []
            # padding
            for x in all_x:
                new_all_x.append(np.pad(x[:dim], (0, dim - len(x[:dim])), 'constant'))
            all_x = np.array(new_all_x)
            mon_x = all_x
            mon_y = all_y
            subf_data_unmonitored = np.load(args.data_root+'/datasets/gdlf_ow_ipd.npz', allow_pickle=True)
            unmon_x = subf_data_unmonitored['X']
            unmon_y = np.array([0]*len(unmon_x))




            def reshape_and_scale(x, img_shape=(-1, dim, 1)):
                return x.reshape(img_shape).astype(
                    np.float32)
            awf_data = np.load(args.data_root+'/datasets/awf1_ipd.npz', allow_pickle=True)
            train_x_unlabeled = awf_data['X']
            train_y_unlabeled = awf_data['y']

            awf_data = np.load(args.data_root+'/datasets/gdlf25_ow_old_ipd.npz', allow_pickle=True)
            awf_x_unlabeled = awf_data['X']
            awf_y_unlabeled = np.array([0] * len(awf_x_unlabeled))

            train_x_unlabeled = np.concatenate(
                (train_x_unlabeled, awf_x_unlabeled), axis=0)
            train_y_unlabeled = np.concatenate(
                (train_y_unlabeled, awf_y_unlabeled), axis=0)
            train_x_unlabeled = reshape_and_scale(train_x_unlabeled)

            train_x_unlabeled = reshape_and_scale(train_x_unlabeled)

            mon_x = reshape_and_scale(mon_x)
            unmon_x = reshape_and_scale(unmon_x)
            X, y = shuffle(mon_x, mon_y)

        print("Setup the input pipeline...")
        with tf.name_scope("InputPipeline"):
            train_x_labeled, train_y_labeled = [], []
            test_x2, test_y2 = [], []
            for i in range(args.num_classes-1):
                train_x_labeled.append(X[y == i][:args.num_labeled_examples])
                train_y_labeled.append(np.array([i+1]*args.num_labeled_examples))
                test_x2.append(X[y == i][
                               args.num_labeled_examples:args.num_labeled_examples + (96*5)])  # use 10% for testing. (90x12x0.1)
                test_y2.append(np.array([i + 1] * (96*5)))

            train_x_labeled_data = np.concatenate(train_x_labeled)
            train_y_labeled_data = np.concatenate(train_y_labeled)

            test_x2 = np.concatenate(test_x2)
            test_y2 = np.concatenate(test_y2)
            X, y = shuffle(unmon_x, unmon_y)
            train_x_labeled_data = np.concatenate((train_x_labeled_data,X[:args.num_labeled_examples*(args.num_classes-1)]), axis=0)
            train_y_labeled_data = np.concatenate((train_y_labeled_data,y[:args.num_labeled_examples*(args.num_classes-1)]), axis=0)
            test_x_data=np.concatenate((test_x2,X[args.num_labeled_examples*(args.num_classes-1):args.num_labeled_examples * (args.num_classes-1)+args.test_unmon]), axis=0)
            test_y_data=np.concatenate((test_y2,y[args.num_labeled_examples*(args.num_classes-1):args.num_labeled_examples * (args.num_classes-1)+args.test_unmon]), axis=0)

            print('train_x_labeled_data',train_x_labeled_data.shape)
            print('train_y_labeled_data', train_y_labeled_data.shape)

            print('test_x_data', test_x_data.shape)
            print('test_y_data', test_y_data.shape)

            train_x_unlabeled_data = train_x_unlabeled#np.concatenate(train_x_unlabeled)
            train_y_unlabeled_data = train_y_unlabeled#np.concatenate(train_y_unlabeled)

            # save testing set for the testing phase
            np.savez_compressed(args.data_root+'/datasets/wfs-ow-awf-gdow.npz', X=test_x_data, y=test_y_data)
            print('test_x_data',test_x_data.shape)
            print('train_x_labeled_data', train_x_labeled_data.shape)

            train_x_unlabeled2, train_y_unlabeled2 = shuffle(train_x_unlabeled, train_y_unlabeled)
            train_x_unlabeled2_data = train_x_unlabeled2#np.concatenate(train_x_unlabeled2)
            train_y_unlabeled2_data = train_y_unlabeled2#np.concatenate(train_y_unlabeled2)

            labeled_X = tf.placeholder(tf.float32, shape=[None, dim, 1])
            labeled_y = tf.placeholder(tf.int64, shape=[None])

            unlabeled_X = tf.placeholder(tf.float32, shape=[None, dim, 1])
            unlabeled_y = tf.placeholder(tf.int64, shape=[None])

            unlabeled_X2 = tf.placeholder(tf.float32, shape=[None, dim, 1])
            unlabeled_y2 = tf.placeholder(tf.int64, shape=[None])

            test_X = tf.placeholder(tf.float32, shape=[None, dim, 1])
            test_y = tf.placeholder(tf.int64, shape=[None])

            train_labeled_dataset = tf.data.Dataset.from_tensor_slices((labeled_X, labeled_y)) \
                .shuffle(buffer_size=len(train_x_labeled_data)) \
                .repeat()
            train_labeled_dataset = train_labeled_dataset.batch(args.batch_size)
            iterator_labeled = train_labeled_dataset.make_initializable_iterator()
            traces_lab, labels_lab = iterator_labeled.get_next()

            train_unlabeled_dataset = tf.data.Dataset.from_tensor_slices(
                (unlabeled_X, unlabeled_y, unlabeled_X2, unlabeled_y2)) \
                .shuffle(buffer_size=len(train_x_labeled_data)) \
                .repeat()
            train_unlabeled_dataset = train_unlabeled_dataset.batch(args.batch_size)
            iterator_unlabeled = train_unlabeled_dataset.make_initializable_iterator()
            traces_unl, labels_unl, traces_unl2, labels_unl2 = iterator_unlabeled.get_next()

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
            traces_fake = g_model(z)
            traces_fake_perturbed = g_model(z_perturbed)

        with tf.name_scope("Discriminator") as discriminator_scope:
            d_model = Discriminator()
            logits_fake, features_fake = d_model(traces_fake, training=True)
            logits_fake_perturbed, _ = d_model(traces_fake_perturbed, training=True)
            logits_real_unl, features_real_unl = d_model(traces_unl, training=True)
            logits_real_lab, features_real_lab = d_model(traces_lab, training=True)  # 1) For supervised loss
            logits_train, _ = d_model(traces_lab, training=False)

        with tf.name_scope("DiscriminatorLoss"):
            loss_supervised = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_lab, logits=logits_real_lab))

            logits_sum_real = tf.reduce_logsumexp(logits_real_unl, axis=1)
            logits_sum_fake = tf.reduce_logsumexp(logits_fake, axis=1)
            loss_unsupervised = 0.5 * (
                    tf.negative(tf.reduce_mean(logits_sum_real)) +
                    tf.reduce_mean(tf.nn.softplus(logits_sum_real)) +
                    tf.reduce_mean(tf.nn.softplus(logits_sum_fake)))
            loss_d = loss_supervised + loss_unsupervised
            if args.man_reg:
                loss_d += 1e-3 * tf.nn.l2_loss(logits_fake - logits_fake_perturbed) \
                          / tf.to_float(batch_size_tensor)

        with tf.name_scope("Train") as train_scope:
            optimizer = tf.train.AdamOptimizer(args.lr * 0.25)
            optimize_d = optimizer.minimize(loss_d, var_list=d_model.trainable_variables)
            train_tp_op = tp(logits_train, labels_lab)
            train_fp_op = fp(logits_train, labels_lab)
            positive_op = count_pos(logits_train, labels_lab)
            negative_op = count_neg(logits_train, labels_lab)

        with tf.name_scope(discriminator_scope):
            with tf.control_dependencies([optimize_d]):
                logits_fake, features_fake = d_model(traces_fake, training=True)
                logits_real_unl, features_real_unl = d_model(traces_unl2, training=True)

        with tf.name_scope("GeneratorLoss"):
            feature_mean_real = tf.reduce_mean(features_real_unl, axis=0)
            feature_mean_fake = tf.reduce_mean(features_fake, axis=0)
            # L2 distance of features is the loss for the generator
            loss_g = tf.reduce_mean(tf.squared_difference(feature_mean_real, feature_mean_fake))

        with tf.name_scope(train_scope):
            optimizer = tf.train.AdamOptimizer(args.lr, beta1=0.5)
            train_op = optimizer.minimize(loss_g, var_list=g_model.trainable_variables)

        with tf.name_scope(discriminator_scope):
            with tf.name_scope("Test"):
                logits_test, _ = d_model(traces_test, training=False)
                test_tp_op = tp(logits_test, labels_test)
                test_fp_op = fp(logits_test, labels_test)
                test_pos_op = count_pos(logits_test, labels_test)
                test_neg_op = count_neg(logits_test, labels_test)

        # Setup summaries
        with tf.name_scope("Summaries"):
            summary_op = tf.summary.merge([
                tf.summary.scalar("LossDiscriminator", loss_d),
                tf.summary.scalar("LossGenerator", loss_g)])

        print("Run training...")

        steps_per_epoch = (len(train_x_labeled_data)+len(train_x_unlabeled_data)) // args.batch_size
        steps_per_test = test_x_data.shape[0] // args.batch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            saver = tf.train.Saver()
            for epoch in range(args.train_epochs):
                losses_d, losses_g, accuracies, train_tps, train_fps, positives, negatives = [], [], [], [], [], [], []
                print("Epoch {}".format(epoch))
                pbar = tqdm.trange(steps_per_epoch)

                sess.run(iterator_labeled.initializer,
                         feed_dict={labeled_X: train_x_labeled_data, labeled_y: train_y_labeled_data})

                sess.run(iterator_unlabeled.initializer,
                         feed_dict={unlabeled_X: train_x_unlabeled_data, unlabeled_y: train_y_unlabeled_data,
                                    unlabeled_X2: train_x_unlabeled2_data, unlabeled_y2: train_y_unlabeled2_data})

                sess.run(iterator_test.initializer, feed_dict={test_X: test_x_data, test_y: test_y_data})

                for _ in pbar:
                    if step % 1000 == 0:
                        _, loss_g_batch, loss_d_batch, summ, tp_batch, fp_batch, positive, negative = \
                            sess.run([train_op, loss_g, loss_d, summary_op, train_tp_op, train_fp_op,
                                      positive_op, negative_op])
                    else:
                        _, loss_g_batch, loss_d_batch, tp_batch, fp_batch, positive, negative = sess.run(
                            [train_op, loss_g, loss_d, train_tp_op, train_fp_op, positive_op, negative_op])
                    pbar.set_description("Discriminator loss {0:.3f}, Generator loss {1:.3f}"
                                         .format(loss_d_batch, loss_g_batch))
                    losses_d.append(loss_d_batch)
                    losses_g.append(loss_g_batch)
                    train_tps.append(tp_batch)
                    train_fps.append(fp_batch)
                    positives.append(positive)
                    negatives.append(negative)
                    tpr_train = np.sum(train_tps) / np.sum(positives)
                    fpr_train = np.sum(train_fps) / np.sum(negatives)
                    precision_train = np.sum(train_tps) / (np.sum(train_tps) + np.sum(train_fps))
                    step += 1

                print("Discriminator loss: {0:.4f}, Generator loss: {1:.4f}, Train TPR: {2:.4f}, Train FPR: {3:.4f}, Train Precision: {4:.4f}"
                      .format(np.mean(losses_d), np.mean(losses_g), np.mean(tpr_train), np.mean(fpr_train), np.mean(precision_train)))

                tps_test, fps_test, positives_test, negatives_test = [], [], [], []
                for _ in range(steps_per_test):
                    TP, FP, test_pos, test_neg = sess.run([test_tp_op, test_fp_op, test_pos_op, test_neg_op])
                    tps_test.append(TP)
                    fps_test.append(FP)
                    positives_test.append(test_pos)
                    negatives_test.append(test_neg)

                tpr = np.sum(tps_test) / np.sum(positives_test)
                fpr = np.sum(fps_test) / np.sum(negatives_test)
                precision = np.sum(tps_test) / (np.sum(tps_test) + np.sum(fps_test))
                print("Epoch {}".format(epoch), "Test TPR: ", tpr, " Precision: ", precision)
                #if (precision > best_pre) and (epoch != 0):
                if (epoch != 0) and (precision > 0.79) and (precision > best_pre):
                    best_pre = precision
                    print ('saving...')
                    saver.save (sess, args.model_root+"/ssl_saved_model/wfs-ow" + str (precision) + ".ckpt")
                    print ('saved')


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
    parser.add_argument ('--train_epochs', required=False, default=100)
    parser.add_argument ('--lr', required=False, default=2e-4)
    parser.add_argument ('--stddev', required=False, default=1e-2)
    parser.add_argument ('--num_classes', required=False, default=26)
    parser.add_argument ('--z_dim_size', required=False, default=100)
    parser.add_argument ('--num_labeled_examples', required=False, default=90)
    parser.add_argument ('--test_unmon', required=False, default=70000)
    parser.add_argument ('--data_root', required=False, default=HOME)
    parser.add_argument ('--model_root', required=False, default=HOME)
    parser.add_argument ('--man_reg', required=False, default=True)
    args = parser.parse_args ()
    for run in range(2):
        main(args)
