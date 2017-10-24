"""
本文件定义从udacity的lab.ipynb的来准备训练数据的定义
"""
from zipfile import ZipFile
from tqdm import tqdm
#from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import pickle

def uncompress_features_labels(file):
    """
    直接从ZIP文件里面读取图片的方法比较省空间
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # 长生一个zip内文件名集合
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # 循环产生学习数据
        for filename in filenames_pbar:
            # 检查是否是目录，防止有其他东西
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    #PIL这个包与新版本的Python不兼容，所以改用其他包
                    #image = Image.open(image_file)
                    #image.load()
                    image = plt.imread(image_file,'L')
                    #image = plt.imread(image_file)
                    #plt.imshow(image)
                    #plt.show()
                    feature = np.array(image, dtype=np.float32).flatten()
                    # 找到文件的第一个字母
                    label = os.path.split(filename)[1][0]
                    features.append(feature)
                    labels.append(label)
    return np.array(features), np.array(labels)

#%%问题1：数据归一化过程
# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    将数据归一化到 [0.1, 0.9]的范围内

    """
    a = 0.1
    b = 0.9
    #求出每一行元素的最大值
    try:
        MaxRow = np.max(image_data,axis=1,keepdims=True)
        MinRow = np.min(image_data,axis=1,keepdims=True)
    except:
        MaxRow = np.max(image_data,axis=0)
        MinRow = np.min(image_data,axis=0)
    ba = b - a
    MaxMin = MaxRow - MinRow
    #找到奇异值部分
    Singular = np.where(MaxMin == 0)
    XXmin = image_data - MinRow
    Answer = np.divide(XXmin,MaxMin)
    return Answer * ba + a , Singular


### DON'T MODIFY ANYTHING BELOW ###
# Test Cases
"""
A1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])
B1 = [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     0.125098039216, 0.128235294118, 0.13137254902, 0.9]
Answer1,_ = normalize_grayscale(A1)
np.testing.assert_array_almost_equal(Answer1,B1,decimal=3)

A2 = np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])
B2 = [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
     0.896862745098, 0.9]
Answer2,_ = normalize_grayscale(A2)
np.testing.assert_array_almost_equal(Answer2,B2)

A3 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255],[0, 1, 10, 20, 30, 40, 233, 244, 254,255,255,255]])
B3 = [[0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     0.125098039216, 0.128235294118, 0.13137254902, 0.9],[0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
     0.896862745098, 0.9,0.9,0.9]]
Answer3,_ = normalize_grayscale(A3)
np.testing.assert_array_almost_equal(Answer3,B3)

print('Tests Passed!')
"""

def deleteNan(features,labels,Singular):
    """
    删除features中对应的行Nan数据，同时删除labels中对应数据
    """
    features = np.delete(features,Singular[0],axis=0)
    labels = np.delete(labels,Singular[0],axis=0)
    return features,labels
        
"""
A4 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255],[255, 255, 255, 255, 255, 255, 255, 255, 255,255,255,255],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255],[255, 255, 255, 255, 255, 255, 255, 255, 255,255,255,255]])
A4l = np.array([[8],[7],[6],[5]])
Answer3,NG = normalize_grayscale(A4)

features,labels = deleteNan(Answer3,A4l,NG)
Temp = 0
"""

def DisplayFig(Num,features,labels):
    """
    用来测试features和labels的对应关系，并展示图片
    """
    AFig = features[Num,:]
    AFig = np.reshape(AFig,[28,28])
    print(labels[Num])
    plt.imshow(AFig)
    plt.show()

def MakeOneHotLabel(labels):
    """
    用来将字符类型的labels转换为OneHot类型的
    """
    encoder = LabelBinarizer()
    encoder.fit(labels)
    labels = encoder.transform(labels)

    # 转换过来的label改为float32
    return labels.astype(np.float32)

#%% 保存数据文件
def SaveFiles(pickle_file,train_features,train_labels,valid_features,valid_labels,test_features,test_labels):
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                        {
                            'train_dataset': train_features,
                            'train_labels': train_labels,
                            'valid_dataset': valid_features,
                            'valid_labels': valid_labels,
                            'test_dataset': test_features,
                            'test_labels': test_labels,
                        },
                pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
                raise
    
    
    
    





