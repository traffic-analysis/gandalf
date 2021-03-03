import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
import os.path as opth
import tqdm
import os
from sklearn.utils import shuffle
import argparse
HOME = os.path.expanduser('~')
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
layers = tf.keras.layers
parser = argparse.ArgumentParser()


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


def accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.to_float(tf.equal(preds, labels)))


def main(args):
    global best_acc
    best_acc = 0
    with tf.Graph().as_default():
        print("Input data preprocessing...")
        with tf.name_scope("DataPreprocess"):
            r_train = 500.0 / 6.0
            r_test = 100.0 / 6.0
            nClass = 100  # 95  # 100
            mon_instance = 2498.0  # 1000.0  # 300.0
            unClass = 0  # 40000  # 30000
            unmon_instance = unClass
            dim = 5000  # 3969#784

            with tf.device('/cpu:0'):
                (train_x, train_y, test_x_data, test_y_data) = split_awf_closed(r_train, r_test, nClass, mon_instance,
                                                                           unmon_instance, dim)


            def reshape_and_scale(x, img_shape=(-1, dim, 1)):
                return x.reshape(img_shape).astype(
                    np.float32)

            train_x = reshape_and_scale(train_x)
            test_x_data = reshape_and_scale(test_x_data)

            # Use DF for unlabled set
            awf_data2 = np.load(args.data_root+'/datasets/df_data.npz', allow_pickle=True) # data, labels

            train_x_unlabeled = awf_data2['data']
            train_y_unlabeled = awf_data2['labels']



            train_x_unlabeled = reshape_and_scale(train_x_unlabeled)

            X, y = shuffle(train_x, train_y)
            print(X.shape)
            print(y.shape)
        print("Setup the input pipeline...")
        with tf.name_scope("InputPipeline"):
            train_x_labeled, train_y_labeled = [], []
            for i in range(args.num_classes):
                print(i)
                train_x_labeled.append(X[y == i][:args.num_labeled_examples])
                train_y_labeled.append(y[y == i][:args.num_labeled_examples])
            train_x_labeled_data = np.concatenate(train_x_labeled)
            train_y_labeled_data = np.concatenate(train_y_labeled)
            train_x_unlabeled_data = train_x_unlabeled#np.concatenate(train_x_unlabeled)
            train_y_unlabeled_data = train_y_unlabeled#np.concatenate(train_y_unlabeled)


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

        z, z_perturbed = define_noise(batch_size_tensor,args)

        with tf.name_scope("Generator"):
            g_model = define_generator()
            traces_fake = g_model(z)
            traces_fake_perturbed = g_model(z_perturbed)

        with tf.name_scope("Discriminator") as discriminator_scope:
            d_model = Discriminator()
            logits_fake, features_fake          = d_model(traces_fake, training=True)
            logits_fake_perturbed, _            = d_model(traces_fake_perturbed, training=True)
            logits_real_unl, features_real_unl  = d_model(traces_unl, training=True)
            logits_real_lab, features_real_lab  = d_model(traces_lab, training=True) # 1) For supervised loss
            logits_train, _                     = d_model(traces_lab, training=False)

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
            train_accuracy_op = accuracy(logits_train, labels_lab)

        with tf.name_scope(discriminator_scope):
            with tf.control_dependencies([optimize_d]):
                logits_fake, features_fake = d_model(traces_fake, training=True)
                logits_real_unl, features_real_unl = d_model(traces_unl2, training=True)


        with tf.name_scope("GeneratorLoss"):
            feature_mean_real = tf.reduce_mean(features_real_unl, axis=0)
            feature_mean_fake = tf.reduce_mean(features_fake, axis=0)
            # L1 distance of features is the loss for the generator
            loss_g = tf.reduce_mean(tf.abs(feature_mean_real - feature_mean_fake))

        with tf.name_scope(train_scope):
            optimizer = tf.train.AdamOptimizer(args.lr, beta1=0.5)
            train_op = optimizer.minimize(loss_g, var_list=g_model.trainable_variables)

        with tf.name_scope(discriminator_scope):
            with tf.name_scope("Test"):
                logits_test, _ = d_model(traces_test, training=False)
                test_accuracy_op = accuracy(logits_test, labels_test)


        with tf.name_scope("Summaries"):
            summary_op = tf.summary.merge([
                tf.summary.scalar("LossDiscriminator", loss_d),
                tf.summary.scalar("LossGenerator", loss_g),
                tf.summary.scalar("ClassificationAccuracyTrain", train_accuracy_op),
                tf.summary.scalar("ClassificationAccuracyTest", test_accuracy_op)])
        writer = tf.summary.FileWriter(_next_logdir("tensorboard/wfi20_cw_df"))

        print("Run training...")
        steps_per_epoch = (len(train_x_labeled_data) + len(
            train_x_unlabeled_data)) // args.batch_size
        steps_per_test = test_x_data.shape[0] // args.batch_size #41700
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for epoch in range(args.train_epochs):
                losses_d, losses_g, accuracies = [], [], []
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
                        _, loss_g_batch, loss_d_batch, summ, accuracy_batch = sess.run(
                            [train_op, loss_g, loss_d, summary_op, train_accuracy_op])
                        writer.add_summary(summ, global_step=step)
                    else:
                        _, loss_g_batch, loss_d_batch, accuracy_batch = sess.run(
                            [train_op, loss_g, loss_d, train_accuracy_op])
                    pbar.set_description("Discriminator loss {0:.3f}, Generator loss {1:.3f}"
                                         .format(loss_d_batch, loss_g_batch))
                    losses_d.append(loss_d_batch)
                    losses_g.append(loss_g_batch)
                    accuracies.append(accuracy_batch)
                    step += 1

                print("Discriminator loss: {0:.4f}, Generator loss: {1:.4f}, "
                      "Train accuracy: {2:.4f}"
                      .format(np.mean(losses_d), np.mean(losses_g), np.mean(accuracies)))


                accuracies = [sess.run(test_accuracy_op) for _ in range(steps_per_test)]
                if np.mean (accuracies) > best_acc:
                    best_acc = np.mean (accuracies)

                if epoch == (int(args.train_epochs)-1):
                    print ("Test accuracy: {0:.4f}".format (np.mean (accuracies)))
                    print ("Best accuracy: {0:.4f}".format (best_acc))


def split_awf_closed(r_train, r_test, nClass, mon_instance, unmon_instance, dim):

    mon_data = np.load(args.data_root+'/datasets/awf1.npz', allow_pickle=True)
    mon_x = mon_data['feature']



    ## We need to uniformly random selection over each monitored class
    print('mon_instance',mon_instance)
    print('unmon_instance',unmon_instance)
    num_mtrain_instance = mon_instance * (r_train / (r_train + r_test))  ## number of monitored training instances for each class
    mon_random = np.array(range(int(mon_instance)))
    np.random.shuffle(mon_random)

    mon_train_ins = mon_random[:int(num_mtrain_instance)] #1666
    mon_test_ins = mon_random[int(num_mtrain_instance):]
    print('mon_test_ins', len(mon_test_ins))


    # Due to the memory error, initialize np arrays here first
    train_feature = np.zeros((nClass*len(mon_train_ins), dim), dtype=int)
    test_feature = np.zeros((nClass*len(mon_test_ins),dim), dtype=int)
    print('test_feature', len(test_feature))
    train_label = np.zeros((nClass*len(mon_train_ins),), dtype=int)
    test_label = np.zeros((nClass*len(mon_test_ins),), dtype=int)

    print(len(mon_train_ins))
    print(len(mon_test_ins))
    i = 0
    mon_instance = int(mon_instance)
    print('Monitored training set partitioning...')
    print(nClass)
    print(len(mon_train_ins))
    for c in range(nClass):
        c=int(c)
        print(c)
        for instance in mon_train_ins:
            train_label[i] = c
            train_feature[i] = mon_x[(c*mon_instance)+instance][:dim]
            i += 1
    print(i)
    print('Monitored testing set partitioning...')
    j = 0
    for c in range(nClass):
        c = int(c)
        for instance in mon_test_ins:
            test_label[j]=c
            test_feature[j]=mon_x[(c*mon_instance)+instance][:dim]
            j += 1
    print(j)

    print(j)
    print('train_feature: ', len(train_feature))
    print('train_label: ', len(train_label))
    print('test_feature: ', len(test_feature))
    print('test_label: ', len(test_label))
    print('train_dim: ', len(train_feature[0]))
    print('test_dim: ', len(test_feature[0]))


    return train_feature, train_label, test_feature, test_label

def define_noise(batch_size_tensor, args):
    # Setup noise vector
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
    parser.add_argument ('--train_epochs', required=False, default=27)
    parser.add_argument ('--lr', required=False, default=2e-4)
    parser.add_argument ('--stddev', required=False, default=1e-2)
    parser.add_argument ('--num_classes', required=False, default=100)
    parser.add_argument ('--z_dim_size', required=False, default=100)
    parser.add_argument ('--num_labeled_examples', required=False, default=20)
    parser.add_argument ('--data_root', required=False, default=HOME)
    parser.add_argument ('--man_reg', required=False, default=True)
    args = parser.parse_args ()
    for i in range(10):
        main(args)
