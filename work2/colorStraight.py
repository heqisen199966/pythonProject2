'''
                                        颜色特征向量
                                            ||
                          （赋权颜色直方向量   拼接   赋权颜色矩向量）
                                 //                     \\
                      归一化颜色直方向量                 归一化颜色矩向量
                             //                             \\
                      颜色直方向量√                        颜色矩向量
                                    这里获取颜色直方向量
'''
import tensorflow as tf

'''
    下面涉及到的3600，只是假设有3600，图片张数可以变
'''
def getColorStraight(rawImagesList):
    picsStraightR = []  # 3600张图片的颜色直方向量R分量 3600X86
    picsStraightG = []  # 3600张图片的颜色直方向量G分量 3600X86
    picsStraightB = []  # 3600张图片的颜色直方向量B分量 3600X86
    colorStraightList = []  # 颜色直方向量3600X258，3600张图片，每张图片有258个颜色直方特征
    picsStraightRGBList = [picsStraightR, picsStraightG, picsStraightB]  # 为下面放picsStraightR/G/B值做准备
    for i in range(3):
        # 3600张图片一次性拿出R/G/B分量，按照专利，选取图片中心的112X112部分作为主要图片
        majorImagesList = rawImagesList[:, 56:168, 56:168, i]

        for j in range(86):
            # tempListForRGB=[]#存储计算R/G/B各自分量时的数据，3600X86
            if j == 0:
                # 先统计有多少个0
                totoalZero = 0
                majorImagesListCmpWithZero = tf.math.equal(majorImagesList, 0)  # 看看每个数是否为0,真的为T
                out = tf.cast(majorImagesListCmpWithZero, dtype=tf.float32)  # True的时候会变为1，False的时候会变为0
                # 统计0的个数
                '''
                    下面两行代码意思: 上面的out是一个shape为(3600,112,112)3600张图片，宽高112X112，
                    而一张图片的shape为（1,112,112）所以要单独统计3600张图片中 的0的个数
                    使得最后的totoalZero 的shape为(3600,)，3600个数，每个数代表每张图片0的个数
                    再将其添加到tempListForRGB
                    而最后picsStraightR/G/B 先得到的是 86X3600的矩阵,可以之后再转置
                '''
                totoalZero = tf.reduce_sum(out, axis=1)
                totoalZero = tf.reduce_sum(totoalZero, axis=1)

                picsStraightRGBList[i].append(totoalZero)  # 将0的个数放到列表的第一个位置

            else:  # 统计非0的个数
                ''' 
                根据专利，将0-255划分为85个区间，也就是每3个数为一个区间，在端点处的值归到其左侧的区间
                每个数都必须 j<x<=j+2  例如    1<x<=3
                但是当j为1的时候，由于1的左边是0，没有左侧区间所以当J为1的时候
                将1统计到其右侧区间
                '''
                if j == 1:
                    # 当j是1的时候将其放到右侧区间也就是[1-3]这个区间
                    majorImagesListGT = tf.math.greater_equal(majorImagesList, (j * 3) - 2)
                else:
                    # J不是1的时候取开区间
                    majorImagesListGT = tf.math.greater(majorImagesList, (j * 3) - 2)
                # 由于要将端点值放到其左侧区间，也就是说 [1-3],[4,6],当端点值为4的时候，它是归到[1-3]区间的以此类推
                majorImagesListLE = tf.math.less(majorImagesList, (j * 3) + 1)
                # 还没找到能够一次性提取语句  比如    x>i&& x<=i*3 这种提取方法现在想到的是将他们各自求出来然后再找共同部分
                # 经过比较之后的majorImagesListGT，majorImagesListLE，里面的元素是bool类型，只需要两部分均为ture
                majorImagesGTAndLE = tf.math.equal(majorImagesListGT, majorImagesListLE)
                # 转为float
                out = tf.cast(majorImagesGTAndLE, dtype=tf.float32)
                # 统计  j-2 < x <=j*3+1 的个数
                sumMajorImagesGTAndLE = tf.reduce_sum(out, axis=1)
                sumMajorImagesGTAndLE = tf.reduce_sum(sumMajorImagesGTAndLE, axis=1)
                picsStraightRGBList[i].append(sumMajorImagesGTAndLE)  # 将统计到的数据放进去

    picsStraightR = tf.convert_to_tensor(picsStraightRGBList[0])#得到 86X3600
    picsStraightR = tf.transpose(picsStraightR)#转置变为3600x86
    picsStraightG = tf.convert_to_tensor(picsStraightRGBList[1])
    picsStraightG = tf.transpose(picsStraightG)
    picsStraightB = tf.convert_to_tensor(picsStraightRGBList[2])
    picsStraightB = tf.transpose(picsStraightB)
    #3个3600x86拼接起来得到3600X258这就是颜色直方向量
    colorStraightList = tf.concat([picsStraightR,picsStraightG,picsStraightB],axis=1)
    return colorStraightList


