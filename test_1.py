import tensorflow as tf

a = tf.get_variable("a",[10,10],tf.float32,tf.truncated_normal_initializer(stddev=0.01))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
res = sess.run(a)
print res