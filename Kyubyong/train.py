# -*- coding: utf-8 -*-

'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from tqdm import tqdm
from Kyubyong.data_load import get_batch, load_vocab
from Kyubyong.modules import *
from Kyubyong.networks import encoder, decoder, postnet
from Kyubyong.utils import *
from Kyubyong.hyperparams import Hyperparams as hp
import tensorflow as tf


class TacotronGraph:
    def __init__(self, mode="train"):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set phase
        is_training = True if mode == "train" else False

        # Data Feeding
        # x: Text. (N, Tx)
        # y: Reduced mel-spectrum. (N, Ty//r, n_mels*r)
        # z: Magnitude. (N, Ty, n_fft//2+1)
        if mode == "train":
            self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
        elif mode == "eval":
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
            self.z = tf.placeholder(tf.float32, shape=(None, None, 1+hp.n_fft//2))
            self.fnames = tf.placeholder(tf.string, shape=(None,))
        else:
            # Synthesize
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels * hp.r))

        # Get encoder/decoder inputs
        self.encoder_inputs = embed(self.x, len(hp.vocab), hp.embed_size)  # (N, T_x, E)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1)  # (N, Ty/r, n_mels*r)
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:]  # feed last frames only (N, Ty/r, n_mels)

        # Networks
        with tf.variable_scope("net"):
            # Encoder
            self.memory = encoder(self.encoder_inputs, is_training=is_training)  # (N, T_x, E)

            # Decoder
            self.y_hat, self.alignments = decoder(self.decoder_inputs,
                                                  self.memory,
                                                  is_training=is_training)  # (N, T_y//r, n_mels*r)
            # Postnet
            self.z_hat = postnet(self.y_hat, is_training=is_training)  # (N, T_y//r, (1+n_fft//2)*r)

        # monitor
        self.audio = tf.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32)

        if mode in ("train", "eval"):
            # Loss
            self.loss1 = tf.reduce_mean(tf.abs(self.y_hat - self.y))
            self.loss2 = tf.reduce_mean(tf.abs(self.z_hat - self.z))
            self.loss = self.loss1 + self.loss2

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            # gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))

            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            tf.summary.scalar('{}/loss1'.format(mode), self.loss1)
            tf.summary.scalar('{}/loss'.format(mode), self.loss)
            tf.summary.scalar('{}/lr'.format(mode), self.lr)

            tf.summary.image("{}/mel_gt".format(mode), tf.expand_dims(self.y, -1), max_outputs=1)
            tf.summary.image("{}/mel_hat".format(mode), tf.expand_dims(self.y_hat, -1), max_outputs=1)
            tf.summary.image("{}/mag_gt".format(mode), tf.expand_dims(self.z, -1), max_outputs=1)
            tf.summary.image("{}/mag_hat".format(mode), tf.expand_dims(self.z_hat, -1), max_outputs=1)

            tf.summary.audio("{}/sample".format(mode), tf.expand_dims(self.audio, 0), hp.sr)
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    tacotron = TacotronGraph()
    print("Training Graph loaded")

    # with g.graph.as_default():
    sv = tf.train.Supervisor(logdir=hp.logdir, save_summaries_secs=60, save_model_secs=0)
    with sv.managed_session() as sess:
        for _ in range(hp.epochs):
            for _ in tqdm(range(tacotron.num_batch), total=tacotron.num_batch, ncols=70, leave=False, unit='b'):
                _, gs = sess.run([tacotron.train_op, tacotron.global_step])

                # Write checkpoint files
                if gs % 1000 == 0:
                    sv.saver.save(sess, hp.logdir + '/model_gs_{}k'.format(gs//1000))

                    # plot the first alignment for logging
                    al = sess.run(tacotron.alignments)
                    plot_alignment(al[0], gs)

    print("Training Done")
