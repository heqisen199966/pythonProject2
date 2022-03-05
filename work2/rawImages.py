'''
这里是获得输入图片，
思路：
    先读取图片路径并且对其分类， 然后根据图片的路径读取图片

'''
import  os
import tensorflow as tf
import numpy as np
def getRawImagesAndLabels():
    '''
           炮姜 pj,干姜gj,碳姜tj
           '''
    pj = []
    label_pj = []#炮姜对应的标签0
    gj = []
    label_gj = []#干姜对应的标签1
    tj = []
    label_tj = []#碳姜对应的标签2

    file_dir = "D:\\pics"
    file_dir2 = "D:\\pics\\"  # 后面拼接地址的时候忘记加多个\

    #这里先拿出每张图片的地址
    for file in os.listdir(file_dir):
        name = file.split(sep="(")
        x = name[0].strip(" ")  # 分割之后的字符含有空格,之后要注意输入图片的名字
        if x == "pj":
            pj.append(file_dir2 + file)  # 图片绝对路径
            label_pj.append(0)
        elif x == "gj":
            gj.append(file_dir2 + file)  # 图片绝对路径
            label_gj.append(1)
        elif x == "tj":
            tj.append(file_dir2 + file)  # 图片绝对路径
            label_tj.append(2)

    image_list = np.hstack((pj, gj, tj))  # 将图片路径数据以水平方向拼接，形成长度为3600的数组
    label_list = np.hstack((label_pj, label_gj, label_tj))#拼接标签，注意这里拼接的顺序不能乱
    temp = np.array([image_list, label_list])  # 2X3600矩阵
    temp = temp.transpose()  # 转置3600X2
    np.random.shuffle(temp)  # 打乱顺序
    image_list = list(temp[:, 0])  # 遍历所有行，取每一行的第0列数据出来，这里就是取出图片地址
    label_list = list(temp[:, 1])  # 取出标签
    label_list = [int(i) for i in
                  label_list]  # 列表生成式 ,遍历label_list取出i ,然后执行int(i)再向label_list列表添加int(i)
    trainX_list = image_list[:int(len(image_list)*0.8)]  # 由于3600张照片将前百分之80的数据拿出来做训练
    testX_list = image_list[int(len(image_list)*0.8):]
    trianY_list = label_list[:int(len(image_list)*0.8)]  # 后百分之20数据做测试
    testY_list = label_list[int(len(image_list)*0.8):]

    trainX_list = tf.cast(trainX_list, tf.string)
    testX_list = tf.cast(testX_list, tf.string)
    trianY_list = tf.cast(trianY_list, tf.int32)
    testY_list = tf.cast(testY_list, tf.int32)
    trainX_images = []  # 这里保存图片数据
    testX_images = []

    # 这里根据图片地址开始拿出图片
    for step, x in enumerate(trainX_list):#训练图片
        image_contents = tf.io.read_file(x)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        image = tf.image.resize(image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        trainX_images.append(image)  # 收集每张图片

    for step, x in enumerate(testX_list):#测试图片
        image_contents = tf.io.read_file(x)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        image = tf.image.resize(image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        testX_images.append(image)  # 收集每张图片

    trainX_images = tf.convert_to_tensor(trainX_images)  # 将图片转为张量
    testX_images = tf.convert_to_tensor(testX_images)

    return [trainX_images,trianY_list,testX_images,testY_list]
