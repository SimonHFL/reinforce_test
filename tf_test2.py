import tensorflow as tf
from tensorflow.python.ops import rnn_cell

#TODO: 
# get gradients for probs in first run
# then get rewards, and feed back rewards and gradients for update.
# do stateful forward function.

input_arr = [
				[[1,0]],
				[[0,1]],
				[[1,0]],
				[[1,1]],
				[[1,0]],
				[[1,1]],
				[[1,0]],
			]

input = tf.placeholder(tf.float32, shape=(None,1,2), name='input')

def forward(input):
	with tf.control_dependencies(None):
		weights = tf.Variable(tf.random_normal([2, 2], stddev=0.35))
	out = tf.matmul(input, weights)
	probs = tf.nn.sigmoid(out)[0]
	prediction = tf.argmax(probs, axis=0)
	prob = tf.gather(probs, prediction)
	return prediction, prob


i = tf.constant(1)
total_prob = tf.constant(1, dtype=tf.float32)
while_condition = lambda i, total_prob, output, input: tf.less(i, 6)
output = tf.constant("")
def body(i, total_prob,output,input):
   
    prediction, prob = forward(input[i,:])
    #return tf.multiply(total_prob,2)

    output = tf.string_join( (output, tf.as_string(prediction)))
    #output = tf.add(output,tf.to_int32(prediction))
    return tf.add(i,1), tf.multiply(total_prob,prob ), output, input

# do the loop:
_, prob, output, _ = tf.while_loop(while_condition, body, [i, total_prob, output, input])

#prediction, prob = forward(input)


reward = tf.placeholder(tf.float32, shape=(), name='reward')

loss = - reward * prob

optimizer = tf.train.AdamOptimizer()

train_op = optimizer.minimize(loss)

def get_reward(output):
	sum = 0 
	for x in output.decode('UTF-8'):
		sum += float(x)
	return sum

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#print(sess.run(output, {input: input_arr}))
	for _ in range(10000):
		out, prob_out = sess.run([output, prob], {input: input_arr})

		_ = sess.run(train_op, {input: input_arr, reward:get_reward(out)})
		print(out)
		print(str(prob_out))
