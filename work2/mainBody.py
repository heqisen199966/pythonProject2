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
from colorStraight import  getColorStraight
from colorMoment import  getColorMoment
import colorMix as CM
import cnnVectors as CV
import rawImages2 as RI #这里改了
from tensorflow import keras
import os
import matplotlib.pyplot as mp
from tensorflow.keras.callbacks import  TensorBoard
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#由于显卡不够好，这里用CPU

imagesAndLabels = RI.getRawImagesAndLabels()#获取输入图片及其标签

trainX_images = imagesAndLabels[0]#训练图片
trianY_list   = imagesAndLabels[1]#训练标签
trianY_list = tf.one_hot(trianY_list,depth = 3)#将训练标签变为独热码
testX_images = imagesAndLabels[2]#测试图片
testY_list   = imagesAndLabels[3]#测试标签
testY_list  = tf.one_hot(testY_list ,depth = 3)#将测试标签变为独热码

colorStraightForTrain = getColorStraight(trainX_images) #得到训练的直方向量
colorMomentforTrain = getColorMoment(trainX_images) #得到训练的颜色矩向量
colorStraightForTest = getColorStraight(testX_images) #得到测试的直方向量

colorMomentForTest = getColorMoment(testX_images) #得到测试的颜色矩向量


rawImagesListForTrain = trainX_images[:, 56:168, 56:168, :] #获取训练用的主要部分图像

rawImagesListForTest =  testX_images[:, 56:168, 56:168, :] #获取测试用的主要部分图像

colorStraightForTrain = CM.normalizeColor(colorStraightForTrain)#训练的直方向量归一化
colorMomentforTrain  = CM.normalizeColor(colorMomentforTrain)#训练的颜色矩向量归一化

colorStraightForTest  = CM.normalizeColor(colorStraightForTest)#测试的直方向量归一化
colorMomentForTest  = CM.normalizeColor(colorMomentForTest)#测试的颜色矩向量归一化


#将训练的归一化的直方向量和训练的归一化的颜色矩向量进行赋权值
colorStraightForTrain, colorMomentforTrain = CM.empowerNormalizedImagesList(colorStraightForTrain, colorMomentforTrain, 1,28)# 1  28
#将训练的赋权直方向量和训练的赋权颜色矩向量进行拼接
colorVectorsForTrain = CM.joinStraightAndColor(colorStraightForTrain, colorMomentforTrain)#获取颜色特征向量

#将测试的归一化的直方向量和测试的归一化的颜色矩向量进行赋权值
colorStraightForTest, colorMomentForTest = CM.empowerNormalizedImagesList(colorStraightForTest, colorMomentForTest, 1,28)#权值
#将测试的赋权直方向量和测试的赋权颜色矩向量进行拼接
colorVectorsForTest = CM.joinStraightAndColor(colorStraightForTest, colorMomentForTest)#获取颜色特征向量


optimizer = optimizers.Adam(learning_rate = 0.001)#设置学习率
model = CV.buildMyVGG();


model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_crossentropy','categorical_accuracy'])

#获取训练的卷积神经网络向量
cnnVectorsForTrain = model.predict(rawImagesListForTrain)#这里记住输入的是（N，112，112,3）返回一个nX4096的矩阵
#对训练的卷积神经网络向量归一化
cnnVectorsForTrain = CM.normalizeColor(cnnVectorsForTrain)#对卷积向量进行归一化

#获取测试的卷积神经网络向量
cnnVectorsForTest = model.predict(rawImagesListForTest)
#对测试的卷积神经网络向量归一化
cnnVectorsForTest = CM.normalizeColor(cnnVectorsForTest)

'''
    根据专利，
       对归一化之后的卷积向量和 颜色特征向量 分别赋权，再将颜色特征向量拼接到卷积向量里面
'''

#对训练的归一化的神经网络向量和训练的归一化的颜色矩向量进行赋权值
colorVectorsForTrain, cnnVectorsForTrain = CM.empowerNormalizedImagesList(colorVectorsForTrain, cnnVectorsForTrain, 2, 1)#2 1
#对训练的赋权神经网络向量和训练的赋权颜色矩向量进行拼接得到训练的融合拼接向量
colorAndCnnForTrain = CM.joinStraightAndColor(cnnVectorsForTrain, colorVectorsForTrain)
#训练的融合拼接向量进行归一化，最终得到融合特征向量
colorAndCnnForTrain = CM.normalizeColor(colorAndCnnForTrain)

#下面同上，获取的是测试用的融合特征向量
colorVectorsForTest, cnnVectorsForTest = CM.empowerNormalizedImagesList(colorVectorsForTest, cnnVectorsForTest,2, 1)#赋权
colorAndCnnForTest = CM.joinStraightAndColor(cnnVectorsForTest, colorVectorsForTest)#获取融合拼接向量
colorAndCnnForTest = CM.normalizeColor(colorAndCnnForTest)#再次归一化

optimizer2 = optimizers.Adam(learning_rate = 0.001)#设置学习率
model2 = CV.buildFullConnect()
# model2.compile(optimizer=optimizer2, loss=losses.CategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.CategoricalAccuracy()])
model2.compile(optimizer = optimizer2, loss="categorical_crossentropy", metrics=['categorical_crossentropy','categorical_accuracy'])
'''
    输入的数含有nan
    过大导致NAN？
    过小
'''

x_eval = colorAndCnnForTest[:480,:];#从验证数据中拿走一半来做测试数据
colorAndCnnForTest = colorAndCnnForTest[480:,:];
y_eval = testY_list[:480,:];
testY_list = testY_list[480:,:];


ds_train = tf.data.Dataset.from_tensor_slices((colorAndCnnForTrain, trianY_list))
ds_test = tf.data.Dataset.from_tensor_slices((colorAndCnnForTest,testY_list))
ds_eval = tf.data.Dataset.from_tensor_slices((x_eval,y_eval))
# #必须batch()，否则少了一个维度，shapeError。
ds_train = ds_train.shuffle(100).batch(32).repeat(1)
ds_test = ds_test.shuffle(100).batch(32).repeat(1)
ds_eval = ds_eval.shuffle(100).batch(32).repeat(1)


# TensorBoard = TensorBoard(log_dir="../model3", histogram_freq=1, write_grads=True)
history = model2.fit(ds_train, validation_data = ds_test, epochs = 400)
model2.evaluate(ds_eval)
'''
loss
categorical_accuracy
val_loss
val_categorical_accuracy
'''
mp.subplot(2,1,1)
mp.plot(history.history['categorical_accuracy'],linewidth = "1",color = "blue",label = "categorical_accuracy")
mp.plot(history.history['val_categorical_accuracy'],linewidth = "1",color = "black",label = "val_categorical_accuracy")
mp.legend()
ax = mp.gca()
ax.grid(axis="both",color="orangered",linewidth=0.75)
mp.title("Original-Accuracy,Epoch=400")




mp.subplot(2,1,2)
mp.plot(history.history['categorical_crossentropy'],linewidth="1",color="blue",label="categorical_crossentropy")
mp.plot(history.history['val_categorical_crossentropy'],linewidth="1",color="black",label="val_categorical_crossentropy")
ax = mp.gca()
ax.grid(axis="both",color="orangered",linewidth = 0.75)
mp.title("Original-categorical_crossentropy,Epoch=400")
mp.legend()
mp.show()




# model2.fit(colorAndCnnForTrain, trianY_list, validation_data=(colorAndCnnForTest, colorAndCnnForTest), epochs=20)








