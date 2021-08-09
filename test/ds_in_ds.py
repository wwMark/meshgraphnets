import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

ds = tf.data.Dataset.from_tensor_slices([[[1, 2, 3], [7, 11, 13]], [[1, 2, 3], [7, 11, 13]]])
print("original ds iter")
for item in ds:
    print(item)

ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
print("----------")
for item in ds:
    print(item)