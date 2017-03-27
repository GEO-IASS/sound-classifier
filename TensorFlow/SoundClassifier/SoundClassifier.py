#import tensorflow as tf
import csv
import random
import math
import sys
import string

def xavier(x):
    return math.sqrt(2.0/x)

######
#FLAGS
# 1 - Random Batches
# 2 - Cross Validation
VALIDATION_METHOD = 2
#####################

###############
#Preparing Data
#Data sets
class_names = ["car", "glass", "shots", "thunder"]
dataset_dir = "../../dataset/"
dataset_format = ".csv"

files = []
for n in class_names:
    files.append(dataset_dir + n + dataset_format)
    
content = []
label = []
class_count = [0 for i in class_names]
#Event is saving name and positions in the content and labels lists
class Event:
    def __init__(self, name, begin, end):
        self.name = name
        self.begin = begin
        self.end = end
    def print(self):
        print(self.name)
        print(self.begin, ' - ', self.end)
events = []
#File is saving name and positions in the events list
class File:
    files = dict()

    def __init__(self, name, begin, end):
        self.name = name
        self.begin = begin
        self.end = end
        File.files[self.name] = self

    def print(self):
        print(self.name)
        print(self.begin, ' - ', self.end)

for i, f in enumerate(files):
    file = open(f)
    reader = csv.reader(file)
    for line in reader:
        #line contains filename
        if len(line) == 1:
            fname = line[0]
            if len(events) > 0:
                events[-1].end = len(content) - 1
            events.append(Event(fname, len(content), len(content) + 1))
            #trimming to file name
            fname = fname.partition('#')[0]
            fname.pop()
            if fname in File.files:
                File.files[fname].end = len(events) - 1
            else:
                File(fname, len(events) - 1, len(events) - 1)
        #line contains data
        else:
            line.pop()
            content.append(line)
            label.append(i)
            class_count[i] += 1
print("Data processed!")
########################

import tensorflow as tf
print("Tensorflow imported!")

#######################################
#NEURAL NETWORK
INPUT_NO = 12
OUTPUT_NO = len(class_names)
HIDDEN_NO = 8

#Data and labels
data = tf.placeholder(tf.float32, [None, INPUT_NO])
labels = tf.placeholder(tf.int32, [None])

#Weights
W1 = tf.Variable(tf.random_uniform([INPUT_NO, HIDDEN_NO], -xavier(INPUT_NO), xavier(INPUT_NO)))
W2 = tf.Variable(tf.random_uniform([HIDDEN_NO, OUTPUT_NO], -xavier(HIDDEN_NO), xavier(HIDDEN_NO)))

#Biases
b1 = tf.Variable(tf.zeros([HIDDEN_NO]))
b2 = tf.Variable(tf.zeros([OUTPUT_NO]))

#Output
hidden = tf.matmul(data, W1) + b1
output = tf.matmul(hidden, W2) + b2
y = tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels)

#Training
train_step = tf.train.AdamOptimizer(0.01).minimize(tf.reduce_mean(y))

#Testing
correct = tf.equal(labels, tf.cast(tf.arg_max(output, 1), tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#Initializing
init = tf.global_variables_initializer()
#########################################

###########
#VALIDATION

def trainRange(session, begin, end, flag = 0):
    ######
    #flags
    #0 - single examples
    #1 - events
    #2 - files
    if flag == 0:
        train_ids = range(begin, end)
        train(session, train_ids)
    elif flag == 1:
        for e in events[begin:end]:
            trainRange(session, e.begin, e.end, 0)
    elif flag == 2:
        for f in files[begin:end]:
            trainRange(session, File.files[f].begin, File.files[f].end, 1)

def train(session, train_ids):
    random.shuffle(train_ids)
    train_data = [content[i] for i in train_ids]
    train_labels = [label[i] for i in train_ids]
    session.run(train_step, feed_dict={data: train_data, labels: train_labels})



if(VALIDATION_METHOD == 1):
    
    ###############
    #Random Batches
    sess = tf.Session()
    sess.run(init)

    #TRAINING
    batch_no = 1000
    batch_size = 1000
    for i in range(batch_no):
        batch_ids = random.sample(range(0, len(content)), batch_size)
        batch_data = [content[j] for j in batch_ids]
        batch_labels = [label[j] for j in batch_ids]

        sess.run(train_step, feed_dict={data: batch_data, labels: batch_labels})
    print("Network trained!")
    print(W1.eval(sess))
    print(W2.eval(sess))

    #TESTING
    batch_no = 100;
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

elif(VALIDATION_METHOD == 2):

    #################
    #Cross Validation
    FOLD_NO = 10
    TRAINING_REPEATS = 20

    sess = tf.Session()

    for i in range(FOLD_NO):

        print ("Fold {}: ".format(i))

        sess.run(init)

        temp_class_count = [0 for i in class_names]

        index_min = [(1.0/FOLD_NO) * i * count for count in class_count]
        index_max = [(1.0/FOLD_NO) * (i+1) * count for count in class_count]

        testing_ids = []
        training_ids = []

        for j, l in enumerate(label):

            if temp_class_count[l] >= index_min[l] and temp_class_count[l] < index_max[l]:
                testing_ids.append(j)
            else:
                training_ids.append(j)

            temp_class_count[l] += 1

                        
        #Training
        print("Training...")
        for j in range(TRAINING_REPEATS):
            random.shuffle(training_ids)
            training_data = [content[j] for j in training_ids]
            training_labels = [label[j] for j in training_ids]

            sess.run(train_step, feed_dict={data: training_data, labels: training_labels})
        print("Training finished!")
        #####

        #Testing
        print("Testing...")
        testing_data = [content[j] for j in testing_ids]
        testing_labels = [label[j] for j in testing_ids]
        conf_mat = [[0 for j in class_names] for i in class_names]

        print("Accuracy: ")
        print(sess.run(accuracy, feed_dict={data: testing_data, labels: testing_labels}))

        #calculating confusion matrix
        output_list = list(sess.run(tf.argmax(output, 1), feed_dict={data: testing_data, labels: testing_labels}))
        for (j, corr_lab) in enumerate(testing_labels):
            conf_mat[corr_lab][output_list[j]] += 1
        print(conf_mat)
        #####

elif(VALIDATION_METHOD == 3):

    #####################
    #File-by file testing







    




