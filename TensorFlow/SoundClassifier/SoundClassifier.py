#import tensorflow as tf
import csv
import random


#Preparing Data
########################

#Data sets
class_names = ["car", "glass", "shots", "thunder"]
dataset_dir = "../../dataset/"
dataset_format = ".csv"

files = []
for n in class_names:
    files.append(dataset_dir + n + dataset_format)
    
content = []
label = []
for i, f in enumerate(files):
    file = open(f)
    reader = csv.reader(file)
    for line in reader:
        line.pop()
        content.append(line)
        label.append(i)
print("Data processed!")

import tensorflow as tf
print("Tensorflow imported!")

#Data and labels
data = tf.placeholder(tf.float32, [None, 12])
labels = tf.placeholder(tf.int32, [None])

#Weights
W = tf.Variable(tf.zeros([12, 5]))

#Bias
b = tf.Variable(tf.zeros([5]))

#Output
output = tf.matmul(data, W) + b
output_softmax = tf.nn.softmax(output)
y = tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels)

train_step = tf.train.AdamOptimizer(0.005).minimize(tf.reduce_mean(y))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


#TRAINING
batch_no = 2000
batch_size = 3000
for i in range(batch_no):
    batch_ids = random.sample(range(0, len(content)), batch_size)
    batch_data = [content[j] for j in batch_ids]
    batch_labels = [label[j] for j in batch_ids]

    sess.run(train_step, feed_dict={data: batch_data, labels: batch_labels})
print("Network trained!")


#TESTING
correct = tf.equal(labels, tf.cast(tf.arg_max(output, 1), tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
batch_no = 1000;
batch_size = 1000;
batch_labels_nonsparse = []
conf_mat = [[0 for j in class_names] for i in class_names]
for i in range(batch_no):
    #testing
    batch_ids = random.sample(range(0, len(content)), batch_size)
    batch_data = [content[j] for j in batch_ids]
    batch_labels = [label[j] for j in batch_ids]

    print(sess.run(accuracy, feed_dict={data: batch_data, labels: batch_labels}))
    output_list = list(sess.run(tf.argmax(output, 1), feed_dict={data: batch_data, labels: batch_labels}))

    #calculating confusion matrix
    for (j, corr_lab) in enumerate(batch_labels):
        conf_mat[corr_lab][output_list[j]] += 1

print(conf_mat)

    




