import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Length = 1000
LENGTH = 1000
in_size = LENGTH
out_size = LENGTH

def sample_data(size, length = 100):
	data =[]
	for _ in range(size):
		data.append(sorted(np.random.normal(4, 1.5, length)))
	return np.array(data)


def random_data(size, length = 100):
	data = []
	for _ in range(size):
		data.append(np.random.random(length))
	return np.array(data)


def preprocess_data(x):
	return [[np.mean(data), np.std(data)] for data in x]


x = tf.placeholder(tf.float32, [None, 2], name = 'feature')
y = tf.placeholder(tf.float32, [None, 1], name = 'label')
z = tf.placeholder(tf.float32, [None, Length], name = 'noise')

# Generator

W_g1 = tf.Variable(tf.random_normal([Length, 32]))
b_g1 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_g2 = tf.Variable(tf.random_normal([32, 32]))
b_g2 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_g3 = tf.Variable(tf.random_normal([32, Length]))
b_g3 = tf.Variable(tf.zeros([1, Length]) + 0.1)


def G_GAN(z, W_g1, b_g1, W_g2, b_g2, W_g3, b_g3):
	g_gan1 = tf.nn.relu(tf.matmul(z, W_g1) + b_g1)
	g_gan2 = tf.nn.sigmoid(tf.matmul(g_gan1, W_g2) + b_g2)
	g_gan3 = tf.matmul(g_gan2, W_g3) + b_g3
	return g_gan3

output_g = G_GAN(z, W_g1, b_g1, W_g2, b_g2, W_g3, b_g3)

G_PARAMS = [W_g1, b_g1, W_g2, b_g2, W_g3, b_g3]

# Discriminator

W_d1 = tf.Variable(tf.random_normal([2, 32]))
b_d1 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_d2 = tf.Variable(tf.random_normal([32, 32]))
b_d2 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_d3 = tf.Variable(tf.random_normal([32, 1]))
b_d3 = tf.Variable(tf.zeros([1, 1]) + 0.1)


def D_GAN(x, W_d1, b_d1, W_d2, b_d2, W_d3, b_d3):
	d_gan1 = tf.nn.relu(tf.matmul(x, W_d1) + b_d1)
	d_gan2 = tf.nn.sigmoid(tf.matmul(d_gan1, W_d2) + b_d2)
	d_gan3 = tf.matmul(d_gan2, W_d3) + b_d3
	return tf.sigmoid(d_gan3), d_gan3

output_d, d_gan_3 = D_GAN(x, W_d1, b_d1, W_d2, b_d2, W_d3, b_d3)
D_PARAMS = [W_d1, b_d1, W_d2, b_d2, W_d3, b_d3]


# Calculate mean & std for each output_g line

MEAN = tf.reduce_mean(output_g, 1)
MEAN_T = tf.transpose(tf.expand_dims(MEAN, 0))
STD = tf.sqrt(tf.reduce_mean(tf.square(output_g - MEAN_T), 1))
DATA = tf.concat([MEAN_T, tf.transpose(tf.expand_dims(STD, 0))], 1)


# GAN discriminator

W_gan1 = tf.Variable(tf.random_normal([2, 32]))
b_gan1 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_gan2 = tf.Variable(tf.random_normal([32, 32]))
b_gan2 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_gan3 = tf.Variable(tf.random_normal([32, 1]))
b_gan3 = tf.Variable(tf.zeros([1, 1]) + 0.1)


def GAN(DATA, W_gan1, b_gan1, W_gan2, b_gan2, W_gan3, b_gan3):
	gan1 = tf.nn.relu(tf.matmul(DATA, W_gan1) + b_gan1)
	gan2 = tf.nn.sigmoid(tf.matmul(gan1, W_gan2) + b_gan2)
	gan3 = tf.matmul(gan2, W_gan3) + b_gan3
	return tf.sigmoid(gan3), gan3

GAN_result, gan_3 = GAN(DATA, W_gan1, b_gan1, W_gan2, b_gan2, W_gan3, b_gan3)
D_GAN_PARAMS = [W_gan1, b_gan1, W_gan2, b_gan2, W_gan3, b_gan3]


d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_gan_3, labels = y))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = gan_3, labels = y))

d_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(d_loss, global_step = tf.Variable(0), var_list = D_PARAMS)
g_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(g_loss, global_step = tf.Variable(0), var_list = G_PARAMS)


init = tf.global_variables_initializer()

epoch = 200
d_loss_log = []
g_loss_log = []
with tf.Session() as sess:
	with tf.device('/gpu:0'):
		sess.run(init)
		print("MSG : Start Training...")
		for step in range(epoch):
			for _ in range(100):
				real = sample_data(100, length = Length)
				noise = random_data(100, length = Length)
				generate = sess.run(output_g, feed_dict = {z: noise})
				X = list(real) + list(generate)
				X = preprocess_data(X)
				Y = [[1] for _ in range(len(real))] + [[0] for _ in range(len(generate))]
				d_loss_value, _ = sess.run([d_loss, d_optimizer], feed_dict = {x: X, y: Y})
				d_loss_log.append(d_loss_value)
				
			dp_value = sess.run(D_PARAMS)
			for i, v in enumerate(D_GAN_PARAMS):
				sess.run(v.assign(dp_value[i]))

			for _ in range(100):
				noise = random_data(100, length = Length)
				g_loss_value, _ = sess.run([g_loss, g_optimizer], feed_dict = {z: noise, y: [[1] for _ in range(len(noise))]})
				g_loss_log.append(g_loss_value)


			if step % 20 == 0 or step + 1 == epoch:
				noise = random_data(1, length = Length)
				generate = sess.run(output_g, feed_dict = {z: noise})
				print("MSG : Epoch {}, GAN-D-LOSS = {:.12f}, GAN-G-LOSS = {:.12f}, generate-mean = {:.4f}, generate-std = {:.4f}".format((step // 20) + 1, d_loss_value, g_loss_value, generate.mean(), generate.std()))

				real = sample_data(1, length = Length)
				data, bins = np.histogram(real[0])
				plt.plot(bins[:-1], data, color = 'g')

				data, bins = np.histogram(noise[0])
				plt.plot(bins[:-1], data, color = 'b')

				data, bins = np.histogram(generate[0])
				plt.plot(bins[:-1], data, color = 'r')
				plt.show()
				