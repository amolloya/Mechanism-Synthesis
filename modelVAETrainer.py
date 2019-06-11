import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
#import tensorflow_probability as tfp

# VAE model trainer
class VAETrainer(tf.keras.Model):
    def __init__(self, name, latent_dim, vae, batch_sz=64, lr=0.01):
        super(VAETrainer, self).__init__()
        self.latent_dim = latent_dim
        self.batch_sz = batch_sz
        self.vae = vae
        self.kl_tolerance = 0.2
        self.kl_weight_start = 0.01
        self.kl_decay_rate = 0.9995
        self.kl_weight_max = 1.0
        self.optimizer = tf.keras.optimizers.Adam(lr, decay=1e-6)
        self.train_step = tf.Variable(0.0, name='train_step', trainable=False)
        if not os.path.exists('./' + self.name + '-' + name):
            os.makedirs('./' + self.name + '-' + name)
        self.checkpoint_dir ='./' + self.name + '-' + name + '/training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        self.status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def load_dataset(self, data):
        ''' data is shuffled before splitting, so always trian model in a single run, if the same model is trained in two different load_dataset calls, it will be trained on test data as well
        '''
        data_train, data_test = train_test_split(data, test_size=0.2)
        self.dataset_train = tf.data.Dataset.from_tensor_slices(data_train).shuffle(len(data_train)).batch(self.batch_sz, drop_remainder=True)
        self.dataset_test = tf.data.Dataset.from_tensor_slices(data_test).shuffle(len(data_test)).batch(self.batch_sz, drop_remainder=True)

    def __call__(self, sequence, teacher_forcing=False):
        ''' sequence has shape [batch_sz, time_steps, 2]
        '''
        noisy_sequence = sequence# + tf.random.normal(shape=sequence.shape, mean=0.0, stddev=0.05)
        latent_mu, latent_sigma_log = self.vae.inference_net(noisy_sequence)

        eps = tf.random.normal(shape=latent_mu.shape)
        z = latent_mu + eps * tf.math.exp(latent_sigma_log)
        if teacher_forcing:
            lagged_sequence = tf.concat((tf.zeros((sequence[:,0:1,:].shape)), sequence[:,:-1,:]), axis=1) + tf.random.normal(shape=sequence.shape, mean=0.0, stddev=0.05)
            dec_output = self.vae.generative_net(z, lagged_sequence)
        else:
            dec_output = self.vae.generative_net(z)

        return latent_mu, latent_sigma_log, dec_output

    def train(self, epochs=10):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for i, batch in enumerate(self.dataset_train):
                batch = tf.cast(batch, tf.float32)
                with tf.GradientTape() as tape:
                    self.train_step.assign(self.train_step + 1.0)
                    mu, logvar, dec_output = self.__call__(batch)
                    gradients, loss = self.compute_loss_and_gradients(tape, batch, mu, logvar, dec_output)
                    if not (loss == tf.math.is_nan):
                        self.apply_gradients(gradients)
                    else:
                        raise ValueError('Loss is nan!')
                    if i % 10 == 0:
                        print('Training %d batch'%i, 'KL Loss : ', np.mean(self.loss_kl.numpy()), 'Rec Loss : ', np.mean(self.loss_rec.numpy()), 'Total Loss : ', loss)
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            end_time = time.time()
            self.evaluate()

    def compute_loss_and_gradients(self, tape, label_sequence, latent_mu, latent_sigma_log, dec_output):
        self.loss_rec = self.vae.calculate_rec_loss(label_sequence, dec_output)
        self.loss_kl = -0.5 * tf.reduce_sum(1 + 2*latent_sigma_log - tf.square(latent_mu) - tf.math.exp(2*latent_sigma_log), axis=1)
        self.loss_kl = tf.math.maximum(self.loss_kl - self.kl_tolerance, 0)

        curr_kl_weight = self.kl_weight_max - (self.kl_weight_max - self.kl_weight_start) * (self.kl_decay_rate)**self.train_step

        self.loss_total = tf.reduce_mean(self.loss_rec + curr_kl_weight*self.loss_kl)

        grad = tape.gradient(self.loss_total, self.vae.trainable_variables)
        return grad, self.loss_total

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))

    def evaluate(self):
        for i, batch in enumerate(self.dataset_test):
            with tf.GradientTape() as tape:
                batch = tf.cast(batch, tf.float32)
                mu, logvar, dec_output = self.__call__(batch)
                _, loss = self.compute_loss_and_gradients(tape, batch, mu, logvar, dec_output)
                if i % 10 == 0:
                    print('Testing %d batch'%i, 'KL Loss : ', np.mean(self.loss_kl.numpy()), 'Rec Loss : ', np.mean(self.loss_rec.numpy()), 'Total Loss : ', loss)

    def predict(self, sequence=None, num_samples=3):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for batch in self.dataset_test:
            batch = tf.cast(batch, tf.float32)
            for i in range(len(batch)):
                ax.clear()
                sequence = batch[i:i+1,:,:]
                x_test, y_test = tf.split(sequence, 2, axis=batch.shape[-1])
                x_pred, y_pred = tf.split(self.vae(sequence), 2, axis=batch.shape[-1]+1)
                for i in range(x_test.shape[0]):
                    ax.plot(x_test[i,:,0], y_test[i,:,0], '.k', ms=5)
                    for j in range(x_pred.shape[1]):
                        ax.plot(x_pred[i,j,:,0], y_pred[i,j,:,0], '-.', ms=2)
            plt.pause(1)
        plt.show()

    def predict_images(self, images=None, num_samples=3):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for batch in self.dataset_test:
            batch = tf.cast(batch, tf.float32)
            for i in range(len(batch)):
                ax.clear()
                sequence = batch[i:i+1,:,:,:]
                ax.imshow(sequence[0,:,:,0])
                prediction = tf.nn.sigmoid(self.vae.call2d(sequence))
                for j in range(prediction.shape[1]):
                    ax2.clear()
                    ax2.imshow(prediction[0,j,:,:,0])
                    plt.pause(1)
