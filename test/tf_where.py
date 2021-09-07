import tensorflow as tf

mask = tf.constant([True, False, True])
a = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0])
b = tf.constant([2, 0, 0, 3, 0, 0, 4, 0, 0])
print(tf.where(mask, a, b))