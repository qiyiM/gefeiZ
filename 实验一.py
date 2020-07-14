import gzip
def unpack_gz(file_lis):
    '''解压缩gs文件'''
    for f in file_lis:
        with open(f, 'rb') as fin:
            with open(f.replace('.gz',''),'wb') as fout:
                fout.write(gzip.decompress(fin.read()))

if _name_ == '_main_':
    #待解压的文件路径列表
    gz_file_lis = ['./MNIST/train-images-idx3-ubyte.gz',
        './MNIST/train-labels-idxl-ubyte.gz',
        './MNIST/t10k-images-udx3-ubyte.gz',
        './MNIST/t10k-labels-idxl-ubyte.gz']

    unpack_gz(gz_file_lis)

import numpy as np
import os

class MNISTLoader(object):
    '''MNIST数据加载器'''
    def _init_(self, path, dtype, sample_size, offset):
        '''MNIST数据加载器初始化函数

        Args:
            data_path: str 文件路径
            dtype: str 数据类型描述，与Numpy兼容
            sample_size:int 数据文件中每个样例的大小，以字节计算
            offset: int 文件头偏移量 以字节计算
        '''
        self.path = os.path.realpath(path)
        self.dtype = dtype
        self.sample_size = sample_size
        self.offset = offset
    def load(self):
        '''加载数据'''
        data = np.fromfile(self.path, dtype = self.dtype)[self.offset:]
        return data.reshape(-1, self.sample_size)


if _name_ == '_main_':
    train_data_path = './MNIST/train-images-idx3-ubyte'
    train_label_path = './MNIST/train-labels-idx1-ubyte'
    test_data_path = './MNIST/t10k-images-idx3-ubyte'
    test_label_path = './MNIST/t10k-labels-idx1-ubyte'
    #图像数据大小为28像素*28像素，类型‘u1’为无符号单字节，文件前16字节为无关数据
    train_loader = MNISTLoader(train_data_path, 'u1', 28*28, 16)
    #图像标签数据大小为1字节的整数，文件的前8字节为无关数据
    train_label_loader = MNISTLoader(train_label_path, 'u1', 1, 8)
    #图像数据大小为28像素*28像素，类型‘u1’为无符号单字节，文件前16字节为无关数据
    test_loader = MNISTLoader(test_data_path, 'u1', 28*28, 16)
    #图像标签数据大小为1字节的整数，文件的前8字节为无关数据
    test_label_loader = MNISTLoader(test_label_path, 'u1', 1, 8)

    train_data = train_loader.load()
    trainlabels = train_label_loader.load()
    test_data = test_loader.load()
    test_labels = test_label_loader.load()

    #绘制图像数字
    from matplotlib import pyplot as plt 
    plt.subplot(121)
    plt.imshow(train_data[0].reshape(28, 28))
    plt.subplot(122)
    plt.imshow(test_data[0].reshape(28, 28))
    plt.show()
    print(train_labels[0], test_labels[0])

# coding = utf8
import numpy as np

from MINSTLoader import MNISTLoader
from SequentialModel import SequentialModel

def categorical_to_onehot(label_lis, class_num):
    '''将此类编码转换为onehot编码'''
    label_lis = np.squeeze(label_lis).astype(int)
    onehot = np.zeros((len(label_lis),class_num))
    onehot[range(len(label_lis)),label_lis] = 1
    return onehot

def load_data():
    '''加载所有数据以及标签'''
    train_data_path = './MNIST/train-images-idx3-ubyte'
    train_label_path = './MNIST/train-labels-idx1-ubyte'
    test_data_path = './MNIST/t10k-images-idx3-ubyte'
    test_label_path = './MNIST/t10k-labels-idx1-ubyte'
    
    #图像数据大小为28像素*28像素，文件的前8字节为无关字节
    train_loader = MNISTLoader(train_data_path, 'u1', 28*28, 16)
    #图像标签数据大小为1字节的整数，文件前12字节为无关数据
    train_label_loader = MNISTLoader(train_label_path, 'u1', 1, 8)
    #图像数据大小为28像素*28像素，文件前16字节为无关数据
    train_loader = MNISTLoader(test_data_path, 'u1', 28*28, 16)
    #图像标签数据大小为1字节的整数，文件的前8字节为无关数据
    test_label_loader = MNISTLoader(test_label_path, 'u1', 1, 8)

    train_data = train_loader.load()
    train_labels = train_label_loader.load()
    test_data = test_loader.load()
    test_labels = test_label_loader.load()

    #转换标签为onehot类型
    train_labels = categorical_to_onehot(train_labels, 10)
    test_labels = categorical_to_onehot(test_labels, 10)

    return (train_data, train_labels),(test_data, test_labels)


def create_model():
    '''构建基于全连接层的手写数字识别模型'''
    model = SequentialModel()
    #构建输入层
    model.add_fc_layer(28*28)
    #构建第一个隐层
    model.add_fc_layer(400, use_bias=True, activation='sigmoid')
    #构建输出层
    model.add_fc_layer(10, use_bias=True, activation='sigmoid')

    return model

if _name_=='_main_':
    #获取数据和标签
    (train_x, train_y), (test_x, test_y) = load_data()
    #构建基于全连接网络的序贯模型
    model = create_model()
    #输出模型信息
    print(model)
    #训练模型
    model.fit(train_x, train_y, 10, learning_rate=0.1)
    #评估模型在测试集上的表现结果
    model.evaluate(test_x, test_y)


