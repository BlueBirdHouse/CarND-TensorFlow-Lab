"""
指导文件文件保存在
http://nbviewer.jupyter.org/github/BlueBirdHouse/CarND-TensorFlow-Lab/blob/master/lab.ipynb
"""
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import PrepareData

#A3 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255],[255, 255, 255, 255, 255, 255, 255, 255, 255,255,255,255]])
#train_features1,NG = PrepareData.normalize_grayscale(A3)

#Temp = 0

#%%执行导入数据过程
#导入训练集合
train_features, train_labels = PrepareData.uncompress_features_labels('notMNIST_train.zip')

#导入测试集合
test_features, test_labels = PrepareData.uncompress_features_labels('notMNIST_test.zip')


#%%问题1：数据归一化过程
# Problem 1 - Implement Min-Max scaling for grayscale image data
train_features,train_features_NG = PrepareData.normalize_grayscale(train_features)
test_features,test_features_NG = PrepareData.normalize_grayscale(test_features)

#%%清除问题数据
train_features,train_labels = PrepareData.deleteNan(train_features,train_labels,train_features_NG)
test_features,test_labels = PrepareData.deleteNan(test_features,test_labels,test_features_NG)

#%% 取原有数据集合中的一部分,原作者的目的是为了配合虚拟空间的使用
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)

#%% 将字符类型的labels转换为OneHot类型的
train_labels = PrepareData.MakeOneHotLabel(train_labels)
test_labels = PrepareData.MakeOneHotLabel(test_labels)

#%% 在训练集合里面进一步生成测试集合
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

#%% 保存文件供下一次访问
PrepareData.SaveFiles('notMNIST.pickle',train_features,
                            train_labels,
                            valid_features,
                            valid_labels,
                            test_features,
                            test_labels)














