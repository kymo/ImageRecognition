#encoding:utf-8
from numpy import *
import cv2
import os
from pybrain.tools.shortcuts        import buildNetwork
from pybrain.datasets               import SupervisedDataSet
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.utilities              import percentError
from pybrain.datasets               import ClassificationDataSet
from numpy.random import multivariate_normal
#read image

class ExtractFeature:

    def __init__(self):
        self.feature_path = "f.in"
        self.image_path = "./xlj"

    def pca(self, data_mat, top_n = 5):
        """ component analysis for dimentionality reduction

        Args:
            matrix: matrix which's gone to be reducted

        Returns:
            new vector
        """
        meanVals = mean(data_mat, axis=0)
        meanRemoved = data_mat - meanVals #减去均值
        stded = meanRemoved / std(data_mat) #用标准差归一化
        covMat = cov(stded,  rowvar=0) #求协方差方阵
        eigVals, eigVects = linalg.eig(mat(covMat)) #求特征值和特征向量
        eigValInd = argsort(eigVals)  #对特征值进行排序
        eigValInd = eigValInd[:-(top_n + 1):-1]
        redEigVects = eigVects[:, eigValInd]       # 除去不需要的特征向量
        lowDDataMat = stded * redEigVects    #求新的数据矩阵
        reconMat = (lowDDataMat * redEigVects.T) * std(data_mat) + meanVals
        return lowDDataMat, reconMat

    def extract_feature(self, image_file):
        """extract feature from the file

        Args:
            None
        Return:
            feature_dict,just like the following structure shows:
            {'key_count' : 1, 'feature':[[1,2,3],[2,3,4]]}
        """
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        detector = cv2.SIFT()
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        keypoints = detector.detect(gray,None)
        # descripters
        k1,d1 = descriptor.compute(gray, keypoints)
        # PCA
        r, s = self.pca(d1, 32)
        A = float32(r.A)
        st = cv2.kmeans(A, 32, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
        feature = []
        for item in st[2]:
            feature.extend(item)
        return feature


    def resize_img(self, img_path, out_path, new_width):
        import Image
        #读取图像
        im = Image.open(img_path)
        #获得图像的宽度和高度
        width,height = im.size
        #计算高宽比
        ratio = 1.0 * height / width
        #计算新的高度
        new_height = 128
        new_size = (new_width, new_height)
        #插值缩放图像，
        out = im.resize(new_size, Image.ANTIALIAS)
        #保存图像
        out.save(out_path)

    def extract_all(self):
        """ extract all the features of the sample images

        Args:
            None

        Return:
            None
        """
        feature_file = open(self.feature_path, "w")
        for i in range(1, 26):
            image_path = self.image_path + str(i)
            images = os.listdir(image_path)
            for item in images:
                feature = self.extract_feature(image_path + "/" + item)
                print len(feature), type(feature)
                feature_file.write('\t'.join([str(_i) for _i in feature]))
                feature_file.write('\t' + str(i) + '\n')
        feature_file.close()


class Classifier():

    def __init__(self):
        pass

    def init_net(self):
        [ass]

    def load_data(self):
        datas = open(self.FN, "r").readlines()
        for item in datas:
            data = item.strip('\n').split('\t')
            input = [float(val) for val in data[:self.IN]]
            output = [int(data[self.IN]) - 1]
            self.ds.addSample(input, output)
        self.tstdata, self.trndata = self.ds.splitWithProportion(self.RT)

        self.trndata._convertToOneOfMany()
        self.tstdata._convertToOneOfMany()

    def init_iri(self):
        self.IN = 4
        self.HN = 12
        self.ON = 3
        self.D = 150
        self.RT = 0.25
        self.FN=  "iris.data"

    def init_image(self):
        self.IN = 1024
        self.HN = 128
        self.D = 125
        self.RT = 0.25
        self.FN = "f.in"

    def load_iris(self):
        datas = open("iris.data", "r").readlines()
        typ = {"Iris-virginica" : 3, "Iris-versicolor":2, "Iris-setosa" : 1}

        for item in datas:
            data = item.strip('\n').split(',')
            input = [float(val) for val in data[:-1]]
            output = [typ[data[-1]] - 1]
            self.ds.addSample(input, output)

        self.tstdata, self.trndata = self.ds.splitWithProportion(self.RT)
        self.trndata._convertToOneOfMany()
        self.tstdata._convertToOneOfMany()


    def train(self):

        #self.init_iri()
        self.init_image()
        self.ds = ClassificationDataSet(self.IN, 1, nb_classes=128)
        #classifier.init_image()
        self.load_data()
        print "Number of trianing patterns: ", len(self.trndata)
        print "Input and output dimensions: ", self.trndata.indim, self.trndata.outdim
        print "First sample (input, target, class):"
        print self.trndata['input'][0], self.trndata['target'][0], self.trndata['class'][0]
        print self.trndata.indim, self.trndata.outdim
        self.net = buildNetwork(self.trndata.indim, 7, self.trndata.outdim)


        trainer = BackpropTrainer(self.net, dataset=self.trndata, momentum=0.1, verbose=True, weightdecay=0.01)

        """
        for i in range(200):
            trainer.trainEpochs(1)
            trnresult = percentError(trainer.testOnClassData(), self.trndata['class'])
            tstresult = percentError(trainer.testOnClassData(dataset = self.tstdata), self.tstdata["class"])
            print "epch: %4d" %  trainer.totalepochs, \
                " train error: %5.2f%%" % trnresult, \
                " test error: %5.2f%%" % tstresult
        """
        trainer.trainUntilConvergence()
        trnresult = percentError(trainer.testOnClassData(), self.trndata['class'])
        tstresult = percentError(trainer.testOnClassData(dataset = self.tstdata), self.tstdata["class"])
        print "epch: %4d" %  trainer.totalepochs, \
            " train error: %5.2f%%" % trnresult, \
            " test error: %5.2f%%" % tstresult

if __name__ == "__main__":
    classifier = Classifier()
    classifier.train()


