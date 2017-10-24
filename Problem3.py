"""
本文件完成任务3：想办法调整参数让准确率达到80%以上
"""
#去掉无用的提示
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import pickle
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


#%%调入存储的数据文件
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # Free up memory


print('Data and modules loaded.')

#%%定义神经网络
#固定输入参数以避免前面的错误传递
features_count = 784
labels_count = 10

features = tf.placeholder(tf.float32,shape=[None, features_count])
labels =tf.placeholder(tf.float32,shape=[None, labels_count])

#创建学习对象
weights = tf.Variable(tf.random_normal([features_count,labels_count]))
biases = tf.Variable(tf.random_normal([labels_count]))
#biases = tf.Variable(tf.zeros([labels_count]))

#%%训练，审核和测试用的输入字典
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

#%%网络创建过程
#创建线性网络
Mux1 = tf.matmul(features,weights)
logits = tf.add(Mux1,biases)

#生成交叉熵（训练目标）
#这里软max函数和交叉熵联合使用
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
loss = tf.reduce_mean(cross_entropy)

#指定优化器
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

#%%这相当与是一个新的网络，这个网路的输出反映了分类器的准确性
#找到logits当中最大的元素，并比较其位置是否与labels当中的位置相同
is_correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
#将相同与不同的结果转换为float32并作为评价指标计算
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

#%%创建Session
sess = tf.Session()

#%%初始化
init = tf.global_variables_initializer()
sess.run(init)


#%%开始训练
batch_size = 50
#反复训练多少次
epochs = 5

#计算一共有多少个分组
batch_count = int(math.ceil(len(train_features)/batch_size))

for epoch_i in range(epochs):
    # Progress bar
    batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
    for batch_i in batches_pbar:
        #取出训练集
        batch_start = batch_i*batch_size
        batch_features = train_features[batch_start:batch_start + batch_size]
        batch_labels = train_labels[batch_start:batch_start + batch_size]

        sess.run([optimizer],feed_dict={features: batch_features, labels: batch_labels})
        #计算本次训练成果
        validation_accuracy = sess.run(accuracy, feed_dict=valid_feed_dict)
        print('Test accuracy at {}'.format(validation_accuracy))


#做有效性测试
print("训练完成！")
test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
print('Test accuracy at {}'.format(test_accuracy))









