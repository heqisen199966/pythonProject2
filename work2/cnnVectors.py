'''
这里返回卷积神经网络向量
'''
import tensorflow as tf
from tensorflow.keras import layers



def buildMyVGG():#9层,为了凑4096个特征，乱调.。。。

    input_shape = [112, 112, 3]

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
    model.add(layers.Dropout(rate=0.5))

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
     model.add(layers.Dropout(rate=0.5))
     model.add(layers.Dense(128, activation=tf.nn.relu))
     model.add(layers.Dense(3,activation=tf.nn.softmax))#最后的结果为三个
     return model


def buildFullConnect2():#只有颜色直方向量的时候

    input_shape = [4354]
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(3, activation=tf.nn.softmax))  # 最后的结果为三个
    return model

def buildFullConnect3():#只有颜色矩的时候
    input_shape = [4105]
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(3, activation=tf.nn.softmax))  # 最后的结果为三个
    return model

def buildFullConnect4():#仅卷积
    input_shape = [4096]
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dropout(rate=0.7))
    model.add(layers.Dense(3, activation=tf.nn.softmax))  # 最后的结果为三个
    return model

def buildAlexNet():
    input_shape = [112,112,3]
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=[11,11], strides=(4,4),padding="valid", activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_1_3x3_1"))
    model.add(layers.Conv2D(256, kernel_size=[5,5], strides=(1,1),padding="same", activation = tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_1_3x3_2"))
    model.add(layers.Conv2D(384, kernel_size=[3,3], strides=(1,1),padding="same", activation = tf.nn.relu))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Conv2D(384, kernel_size=[3,3], strides=(1,1),padding="same", activation = tf.nn.relu))
    model.add(layers.Conv2D(256, kernel_size=[3,3], strides=(1,1),padding="same", activation = tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size = (3, 3), strides = (1, 1), padding="valid", name = "maxpool_1_3x3_3"))
    model.add(layers.Flatten())
    return model #2304


def buildAlexNetFullConnect():
    input_shape = [2571]
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.Dropout(rate=0.3))
    # model.add(layers.Dense(4096, activation=tf.nn.relu, input_shape=input_shape))
    model.add(layers.Dense(128, activation=tf.nn.relu))
    # model.add(layers.Dense(4096, activation=tf.nn.relu))
    model.add(layers.Dense(3, activation=tf.nn.softmax))  # 最后的结果为三个
    return model



