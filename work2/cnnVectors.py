'''
这里返回卷积神经网络向量
'''
import tensorflow as tf
from tensorflow.keras import layers



def buildMyVGG():#9层,为了凑4096个特征，乱调.。。。
    input_shape = [112,112,3]
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'))

    model.add(layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    model.add(layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'))

    model.add(layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    model.add(layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'))

    model.add(layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    model.add(layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'))


    model.add(layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    # model.add(layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    # model.add(layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu))
    # model.add(layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'))
    model.add(layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'))
    model.add(layers.Flatten())  # 拉直
    return model

def buildFullConnect():#全连接层
     input_shape = [4363]#注意最后放进全连接层有4363个特征，这是根据之前构建的模型得来的
     model = tf.keras.Sequential()
     model.add(layers.Dense(256, activation=tf.nn.relu,input_shape=input_shape))
     model.add(layers.Dense(128, activation=tf.nn.relu))
     model.add(layers.Dense(3,activation=tf.nn.softmax))#最后的结果为三个
     return model


def buildFullConnect2():#只有直方向量的时候

    input_shape = [4096]
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(3, activation=tf.nn.softmax))  # 最后的结果为三个
    return model

def buildFullConnect3():#只有颜色矩的时候
    input_shape = [267]
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(3, activation=tf.nn.softmax))  # 最后的结果为三个
    return model
