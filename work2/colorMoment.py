'''

                                        颜色特征向量
                                            ||
                          （赋权颜色直方向量   拼接   赋权颜色矩向量）
                                 //                     \\
                      归一化颜色直方向量                 归一化颜色矩向量
                             //                             \\
                      颜色直方向量                         颜色矩向量√
                                    这里获取颜色矩向量

                    涉及幂函数或者开根号，此时需要加上一个很小的数1e-8,防止出错
'''
import tensorflow as tf
import numpy as np

def getColorMoment(rawImagesList):
    '''
    每张图片的颜色矩向量由其R/G/B三部分组成形成1X9的矩阵,那么3600张图片应该得到的矩阵为3600X9
    先各自分别计算出图片的颜色矩向量单独的R/G/B分量 3600X3 ,最后再拼接起来
    '''
    colorMomentR=[] #颜色矩向量中的R
    colorMomentG=[]#颜色矩向量中的G
    colorMomentB=[]#颜色矩向量中的B
    colorMomentList = [colorMomentR,colorMomentG,colorMomentB]#为了方便下面存储数据
    for i in range(3):

        '''
            3600张图片一次性拿出R/G/B分量，按照专利，选取图片中心的4个部分作为主要图片
            将图片划分为16个等份的字块，输入图片为224X224，
            划分16等份，每一份应该为56X56，中心4份组成主要图片，那么
            majorImagesList.shape = (3600,112,112)
        '''

        #获取中心图片
        majorImagesList = rawImagesList[:, 56:168, 56:168, i]

        majorImagesList = tf.cast(majorImagesList,"float32")#为下面计算做准备

        # totalPixels = 12544
        totalPixels = 50176
        '''
        根据专利，一阶矩 = 当前通道中(R/G/B)的像素值和 除以 像素个数总和
        出现负数，导致后面卷积出现错误，专利没提到怎么解决负数，
        先尝试下将负数直接搞为0
        '''
        #pixelValuesSum最后是一个含有3600个数的数组，如果不设置axis=1，那么会默认全部加起来，只有一个值，不符合
        pixelValuesSum = tf.reduce_sum(majorImagesList, axis=1)
        pixelValuesSum = tf.reduce_sum(pixelValuesSum, axis=1)

        '''
            思路: 先取绝对值得到 absPixelValuesSum ，它再和PixelValuesSum相加
            那么原来负数变为0，正数翻了2倍，然后再除以2
        '''
        absPixelValuesSum  = tf.abs(pixelValuesSum)#绝对值
        pixelValuesSum = pixelValuesSum + absPixelValuesSum#抵消
        pixelValuesSum = pixelValuesSum / 2
        oneMoment = pixelValuesSum/totalPixels # 计算一阶矩,每张图片为112X112，那么总的像素个数为224
        colorMomentList[i].append(oneMoment)#将一阶矩放到当前通道的颜色矩数组中

        '''
            根据专利，
                先求出差值平方矩阵：
                    也就是将主图片（1,112,112）中每个像素值减去之前计算得到的一阶矩值OneMoment，
                    然后再对每个像素值进行平方
                二阶矩 = 开平方根[差值平方矩阵像素值之和/像素点之和]
        '''

        differenceSquareMatrix = []#存放差值平方矩阵
        for y in range(len(rawImagesList)):#依据输入图片的数量来控制循环次数
           #现在能想到的办法是,依次拿出每个矩阵再和其对应的一阶矩值OneMoment相减
            differenceSquareMatrix.append(majorImagesList[y]-oneMoment[y])#矩阵和其对应的一阶矩值是对齐的，打乱就不能这样了
        differenceSquareMatrix = tf.convert_to_tensor(differenceSquareMatrix)#转为张量
        differenceSquareMatrix = tf.pow(differenceSquareMatrix+1e-8,2)
        #统计像素值
        pixelValuesSum = tf.reduce_sum(differenceSquareMatrix, axis=1)
        pixelValuesSum = tf.reduce_sum(pixelValuesSum, axis=1)

        twoMoment = np.sqrt((pixelValuesSum/totalPixels)+1e-8)  #计算二阶矩
        colorMomentList[i].append(twoMoment)


        '''
            根据专利求三阶矩:
                   求出差值立方矩阵，方法同上，只不过这次是求立方
                   三阶矩 = 开3次方根[差值立方矩阵像素值之和/像素点之和]
        '''
        differenceCubeMatrix = []  # 存放差值立方矩阵
        for l in range(len(rawImagesList)):
            differenceCubeMatrix.append(majorImagesList[l] - oneMoment[l])  # 矩阵和其对应的一阶矩值是对齐的，打乱就不能这样了
        differenceCubeMatrix = tf.convert_to_tensor(differenceCubeMatrix)  # 转为张量
        differenceCubeMatrix = tf.pow(differenceCubeMatrix+1e-8,3) # pow函数，differenceCubeMatrix的三次方

        '''
               发现负数开立方根会变为NAN,那么我直接将将负数变为正数来统计，可能会有偏差吧
               或者将原本负数变为0
           '''
        differenceCubeMatrix = tf.pow(differenceCubeMatrix+1e-8, 2)  # 先来个平方简单粗暴
        differenceCubeMatrix = tf.pow(differenceCubeMatrix+1e-8, 1 / 2)  # 来开根号，让负数变为正数

        # 统计像素值
        pixelValuesSum = tf.reduce_sum(differenceCubeMatrix, axis=1)
        pixelValuesSum = tf.reduce_sum(pixelValuesSum, axis=1)
        threeMoment = tf.pow((pixelValuesSum / totalPixels)+1e-8,1/3 ) # 最后开立方根计算三阶矩
        colorMomentList[i].append(threeMoment)

    for i in range(3):
        colorMomentList[i] = tf.transpose(colorMomentList[i])#将3X3600转置为3600X3

    colorMoment = tf.concat([colorMomentList[0],colorMomentList[1],colorMomentList[2]],axis=1)#将R/G/B三通道拼接起来3600X9

    return colorMoment