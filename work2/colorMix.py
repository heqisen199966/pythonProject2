'''

                                         颜色特征向量
                                              ||
                          （赋权颜色直方向量√   拼接   赋权颜色矩向量√）
                                 //                     \\
                      归一化颜色直方向量√                归一化颜色矩向量√
                             //                             \\
                      颜色直方向量                         颜色矩向量
                                    归一化，赋权，融合拼接
                        涉及幂函数或者开根号，此时需要加上一个很小的数1e-8,防止出错
'''
import tensorflow as tf
def normalizeColor(ImagesList):
    '''
        根据专利，归一化方法:
                    将每张图片的特征平方然后对它们求和之后再开根号得到A
                    归一化向量 = (1/A) *V(未归一化向量)
    '''
    ImagesList = tf.cast(ImagesList,dtype="float32")#得先转为浮点
    '''
            坑：当0去开根号的时候会出现nan
    '''
    squartImagesList = tf.pow(ImagesList+1e-8,2)#对每个特征先来个平方
    # ImagesList是3600X258/3600X9，3600张图片，每个图片的特征数为 258/9, 那么对每张图片的特征数进行求和
    sumImagesList = tf.reduce_sum(squartImagesList,axis=1)#得到(3600,0)
    sumImagesList = tf.pow(sumImagesList+1e-8,1/2)#开根号
    '''
      思想:
         sumImagesList 得到的是一个3600数组，每个元素代表着每张图片即将被除的数
         那么为让ImagesList 3600X258/9  能够除以对应要除的数
         例如 ImagesList[0] 代表着第0张图片的向量，
         它要除以 sumImagesList[0] ，也就是 ImagesList[0]/sumImagesList[0] = 第0张图片归一化的向量
         同理 [1],[2]....
         那么这里将sumImagesList原本shape为(3600,) 转为 (1,3600) 再转为 (3600,1)
         然后令 ImagesList /sumImagesList 就能够一次性计算，不用循环拿出来
    '''
    sumImagesList =tf.expand_dims(sumImagesList,axis=0)#最前面插入一个维度，shape为(3600,) 转为 (1,3600)
    sumImagesList = tf.transpose(sumImagesList)#转置(1,3600) 再转为 (3600,1)
    normalizeImagesList = ImagesList/sumImagesList
    return normalizeImagesList

def empowerNormalizedImagesList(straight,color,valueForStraight,valueForColor):
    #赋权
    '''
        根据专利，
             直方向量*1(权值)
             颜色矩向量*28(权值)
    '''
    return straight*valueForStraight,color*valueForColor

def joinStraightAndColor(straight,color):
    #最后一个步骤，将颜色矩拼接到直方向量后面
    result = tf.concat([straight,color],axis=1)
    return result



