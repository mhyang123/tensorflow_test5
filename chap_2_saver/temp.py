import tensorflow as tf
    def __int__(self):
        pass
    def exec(self):

var = tf.Variable([5.0], dtype=tf.float32)
con = tf.constant([10.0], dtype=tf.float32)
session = tf.Session()
init = tf.global_variables_initializer()

session.run(init)
print(session.run(var * con))

print('----------')

session.run(var.assign([10.0]))
print(session.run(var))

p1 = tf.placeholder(dtype=tf.float32)
p2 = tf.placeholder(dtype=tf.float32)

t1 = p1 * 3
t2 = p1 * p2
# t1 을 이용해서 12.0 를 출력해주세요
print(session.run(t1, {p1: [4.0]}))
print(session.run(t2, feed_dict={p1: 4.0, p2: [2.0,5.0]}))