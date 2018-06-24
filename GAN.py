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


x = tf.placeholder(tf.float32, [None, 2], name = 'feature')		# features = ['mean', 'std']
y = tf.placeholder(tf.float32, [None, 1], name = 'label')
z = tf.placeholder(tf.float32, [None, Length], name = 'noise')

# Real Model

W_1 = tf.Variable(tf.random_normal([2, 32]))
b_1 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_2 = tf.Variable(tf.random_normal([32, 32]))
b_2 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_3 = tf.Variable(tf.random_normal([32, 1]))
b_3 = tf.Variable(tf.zeros([1, 1]) + 0.1)


def Real_Model(x, W_1, b_1, W_2, b_2, W_3, b_3):
	h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
	h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)
	h_3 = tf.matmul(h_2, W_3) + b_3
	return tf.sigmoid(h_3), h_3

output_sig, output = Real_Model(x, W_1, b_1, W_2, b_2, W_3, b_3)
PARAMS = [W_1, b_1, W_2, b_2, W_3, b_3]

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

# Calculate mean & std for each output_g

MEAN = tf.reduce_mean(output_g, 1)
MEAN_T = tf.transpose(tf.expand_dims(MEAN, 0))
STD = tf.sqrt(tf.reduce_mean(tf.square(output_g - MEAN_T), 1))
DATA = tf.concat([MEAN_T, tf.transpose(tf.expand_dims(STD, 0))], 1)

# Discriminator

W_d1 = tf.Variable(tf.random_normal([2, 32]))
b_d1 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_d2 = tf.Variable(tf.random_normal([32, 32]))
b_d2 = tf.Variable(tf.zeros([1, 32]) + 0.1)

W_d3 = tf.Variable(tf.random_normal([32, 1]))
b_d3 = tf.Variable(tf.zeros([1, 1]) + 0.1)


def D_GAN(DATA, W_d1, b_d1, W_d2, b_d2, W_d3, b_d3):
	d_gan1 = tf.nn.relu(tf.matmul(DATA, W_d1) + b_d1)
	d_gan2 = tf.nn.sigmoid(tf.matmul(d_gan1, W_d2) + b_d2)
	d_gan3 = tf.matmul(d_gan2, W_d3) + b_d3
	return tf.sigmoid(d_gan3), d_gan3

output_d_sig, output_d = D_GAN(DATA, W_d1, b_d1, W_d2, b_d2, W_d3, b_d3)
D_PARAMS = [W_d1, b_d1, W_d2, b_d2, W_d3, b_d3]


loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = output, labels = y))
gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = output_d, labels = y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss, global_step = tf.Variable(0), var_list = PARAMS)
gan_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(gan_loss, global_step = tf.Variable(0), var_list = G_PARAMS)


init = tf.global_variables_initializer()

epoch = 200
loss_log = []
gan_loss_log = []
with tf.Session() as sess:
	with tf.device('/cpu:0'):
		sess.run(init)
		print("MSG : Start Training...")
		for step in range(epoch):

			# train the real model
			for _ in range(100):
				real = sample_data(100, length = Length)
				noise = random_data(100, length = Length)
				generate = sess.run(output_g, feed_dict = {z: noise})
				X = list(real) + list(generate)
				X = preprocess_data(X)
				Y = [[1] for _ in range(len(real))] + [[0] for _ in range(len(generate))]
				loss_value, _ = sess.run([loss, optimizer], feed_dict = {x: X, y: Y})
				loss_log.append(loss_value)
				
			params_value = sess.run(PARAMS)
			for i, v in enumerate(D_PARAMS):
				sess.run(v.assign(params_value[i]))

			# adversarial learning
			for _ in range(100):
				noise = random_data(100, length = Length)
				gan_loss_value, _ = sess.run([gan_loss, gan_optimizer], feed_dict = {z: noise, y: [[1] for _ in range(len(noise))]})
				gan_loss_log.append(gan_loss_value)


			if step % 20 == 0 or step + 1 == epoch:
				noise = random_data(1, length = Length)
				generate = sess.run(output_g, feed_dict = {z: noise})
				print("MSG : Epoch {}, GAN-D-LOSS = {:.12f}, GAN-G-LOSS = {:.12f}, generate-mean = {:.4f}, generate-std = {:.4f}".format((step // 20) + 1, loss_value, gan_loss_value, generate.mean(), generate.std()))

				real = sample_data(1, length = Length)
				data, bins = np.histogram(real[0])
				plt.plot(bins[:-1], data, color = 'g')

				data, bins = np.histogram(noise[0])
				plt.plot(bins[:-1], data, color = 'b')

				data, bins = np.histogram(generate[0])
				plt.plot(bins[:-1], data, color = 'r')
				plt.savefig('result' + str(step) + '.png')
				plt.clf()
				