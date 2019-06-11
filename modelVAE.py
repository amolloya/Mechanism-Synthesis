import tensorflow as tf
import numpy as np

class VAE(tf.keras.Model):
    # Initialize parameters
    def __init__(self, latent_dim, inference_net, generative_net):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = inference_net
        self.generative_net = generative_net
    
    # Sample epsilon from normal distribution
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(10, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
    
    # Encode X into mean and log_var
    def encode(self, x):
        mean, logvar = self.inference_net(x)
        return mean, logvar
    
    # Calculate z
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar) + mean
    
    # Decode z into X_hat from generative model
    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits
    
    # Calculate reconstruction loss
    def calculate_rec_loss(self, label_sequence, dec_output):
        return self.generative_net.calculate_rec_loss(label_sequence, dec_output)
    
    # Call functions for generative model
    def call(self, sequence, samples=5):
        mean, logvar = self.encode(sequence)
        output = []
        for _ in range(samples-1):
            z = self.reparameterize(mean, logvar)
            dec_output = self.decode(z)
            output.append(self.generative_net.get_output_samples(dec_output).numpy())
        output = np.array(output)
        print(output.shape)
        return np.transpose(output, [1, 0, 2, 3])

    def call2d(self, sequence, samples=5):
        mean, logvar = self.encode(sequence)
        output = []
        for _ in range(samples-1):
            z = self.reparameterize(mean, logvar)
            dec_output = self.decode(z)
            output.append(self.generative_net.get_output_samples(dec_output).numpy())
        output = np.array(output)
        return np.transpose(output, [1, 0, 2, 3, 4])

class FCNN_Decoder(tf.keras.Model):
    def __init__(self):
        super(FCNN_Decoder, self).__init__()
        self.d1 = tf.keras.layers.Dense(50, activation=tf.nn.leaky_relu)
        self.d2 = tf.keras.layers.Dense(200)

    def call(self, z):
        op = self.d1(z)
        op = self.d2(op)
        op = tf.reshape(op, [-1, 100, 2])
        return op

    def get_output_samples(self, dec_output):
        return dec_output

    def calculate_rec_loss(self, label_sequence, dec_output):
        loss_rec = tf.reduce_sum(tf.square(label_sequence - dec_output), axis=1)
        loss_rec = tf.reduce_sum(loss_rec, axis=1)
        return loss_rec

class FCNN_Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(FCNN_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(30, activation='relu')
        self.d2 = tf.keras.layers.Dense(latent_dim + latent_dim, activation='relu')
        self.d3 = tf.keras.layers.Dense(latent_dim)
        self.d4 = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x1, x2 = tf.split(x, 2, axis=1)
        mu = self.d3(x1)
        log_sigma = self.d4(x2)
        return mu, log_sigma

class CNN_Decoder(tf.keras.Model):
    def __init__(self):
        super(CNN_Decoder, self).__init__()
        self.d1 = tf.keras.layers.Dense(200)
        self.upsamp1 = tf.keras.layers.UpSampling1D(2)
        self.conv1d = tf.keras.layers.Conv1D(filters=16, kernel_size=8, padding='same', activation='relu')
        self.upsamp2 = tf.keras.layers.UpSampling1D(2)
        self.conv2d = tf.keras.layers.Conv1D(filters=8, kernel_size=8, padding='same', activation='relu')
        self.conv3d = tf.keras.layers.Conv1D(filters=2, kernel_size=8, strides=1, padding='same')

    def call(self, x):
        x = self.d1(x)
        x = tf.reshape(x, [-1, 25, 8])
        x = self.upsamp1(x)
        x = self.conv1d(x)
        x = self.upsamp2(x)
        x = self.conv2d(x)
        x = self.conv3d(x)
        return x

    def get_output_samples(self, dec_output):
        return dec_output

    def calculate_rec_loss(self, label_sequence, dec_output):
        loss_rec = tf.reduce_sum(tf.square(label_sequence - dec_output), axis=1)
        loss_rec = tf.reduce_sum(loss_rec, axis=1)
        return loss_rec

class CNN_Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CNN_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.ip = tf.keras.layers.InputLayer(input_shape=(100, 2))
        self.conv1d = tf.keras.layers.Conv1D(filters=8, kernel_size=8, strides=2, activation='relu')
        self.conv2d = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=2, activation='relu')
        self.d1 = tf.keras.layers.Dense(latent_dim + latent_dim)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x = self.ip(x)
        x = self.conv1d(x)
        x = self.conv2d(x)
        x = self.flatten(x)
        mu, log_sigma = tf.split(self.d1(x), 2, axis=1)
        return mu, log_sigma

class CNN_Decoder2D(tf.keras.Model):
    def __init__(self):
        super(CNN_Decoder2D, self).__init__()
        self.d1 = tf.keras.layers.Dense(16*16*32)
        self.reshape = tf.keras.layers.Reshape(target_shape=(16, 16, 32))
        self.conv1_trans = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')
        self.conv2_trans = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu')
        self.conv3_trans = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same')

    def call(self, x):
        x = self.d1(x)
        x = self.reshape(x)
        x = self.conv1_trans(x)
        x = self.conv2_trans(x)
        x = self.conv3_trans(x)
        return x

    def get_output_samples(self, dec_output):
        return dec_output

    def calculate_rec_loss(self, label_sequence, dec_output):
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_output, labels=label_sequence)
        logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        return logpx_z

class CNN_Encoder2D(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CNN_Encoder2D, self).__init__()
        self.latent_dim = latent_dim
        self.ip = tf.keras.layers.InputLayer(input_shape=(64, 64, 1))
        self.conv1d = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.conv2d = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu')
        self.d1 = tf.keras.layers.Dense(latent_dim + latent_dim)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x = self.ip(x)
        x = self.conv1d(x)
        x = self.conv2d(x)
        x = self.flatten(x)
        mu, log_sigma = tf.split(self.d1(x), 2, axis=1)
        return mu, log_sigma
