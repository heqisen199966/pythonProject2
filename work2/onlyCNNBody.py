'''
                                  融合特征向量
                                      ||（归一化）
                                  融合拼接向量
                                      ||
               ( 赋权卷积神经网络向量   拼接     赋权颜色特征向量)
                        //                        \\
              归一化卷积神经网络向量                归一化颜色特征向量
                //                                   \\
              卷积神经网络向量                         颜色特征向量

'''


import  tensorflow as tf
from tensorflow.keras import losses, optimizers


import cnnVectors as CV
import rawImages as RI
from tensorflow import keras
import os
import matplotlib.pyplot as mp

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#由于显卡不够好，这里用CPU

imagesAndLabels = RI.getRawImagesAndLabels()#获取输入图片及其标签

trainX_images = imagesAndLabels[0]#训练图片
trianY_list   = imagesAndLabels[1]#训练标签
trianY_list = tf.one_hot(trianY_list,depth = 3)#将训练标签变为独热码
testX_images = imagesAndLabels[2]#测试图片
testY_list   = imagesAndLabels[3]#测试标签
testY_list  = tf.one_hot(testY_list ,depth = 3)#将测试标签变为独热码
rawImagesListForTrain = trainX_images[:, 56:168, 56:168, :] #获取训练用的主要部分图像
rawImagesListForTest =  testX_images[:, 56:168, 56:168, :] #获取测试用的主要部分图像

optimizer = optimizers.Adam(learning_rate=0.001)#设置学习率
model = CV.buildMyVGG()#获取卷积模型
model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.CategoricalAccuracy()])
#获取训练的卷积神经网络向量
cnnVectorsForTrain = model.predict(rawImagesListForTrain)
#获取测试的卷积神经网络向量
cnnVectorsForTest = model.predict(rawImagesListForTest)


colorAndCnnForTrain =cnnVectorsForTrain



colorAndCnnForTest = cnnVectorsForTest



optimizer2 = optimizers.Adam(learning_rate=0.001)#设置学习率
model2 = CV.buildFullConnect2()
model2.compile(optimizer=optimizer2, loss="mse", metrics=[tf.keras.metrics.CategoricalAccuracy()])



ds_train = tf.data.Dataset.from_tensor_slices((colorAndCnnForTrain, trianY_list))
ds_test = tf.data.Dataset.from_tensor_slices((colorAndCnnForTest,testY_list))
ds_train = ds_train.shuffle(100).batch(32).repeat(1)
ds_test = ds_test.shuffle(100).batch(32).repeat(1)

history = model2.fit(ds_train, validation_data=ds_test, epochs=300)
'''
loss
categorical_accuracy
val_loss
val_categorical_accuracy
'''
mp.subplot(2,1,1)
mp.plot(history.history['categorical_accuracy'],linewidth="1",color="blue",label="categorical_accuracy")
mp.plot(history.history['val_categorical_accuracy'],linewidth="1",color="black",label="val_categorical_accuracy")
mp.legend()
ax = mp.gca()
ax.grid(axis="both",color="orangered",linewidth=0.75)
mp.title("OnlyCNN-Accuracy,Epoch=300")
mp.subplot(2,1,2)
mp.plot(history.history['loss'],linewidth="1",color="blue",label="loss")
mp.plot(history.history['val_loss'],linewidth="1",color="black",label="val_loss")
ax = mp.gca()
ax.grid(axis="both",color="orangered",linewidth=0.75)
mp.title("OnlyCNN-Loss,Epoch=300")
mp.legend()
mp.show()












