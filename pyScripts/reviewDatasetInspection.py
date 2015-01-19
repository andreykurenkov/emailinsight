import string
import cPickle
from scipy.stats import pearsonr
from os import system, listdir
from collections import Counter
from sklearn.covariance import EmpiricalCovariance
from sklearn.feature_extraction import DictVectorizer
from reviewHelpfulnessClassification import FullTextRSTFeatureExtractor, FullPickledRSTFeatureExtractor
from reviewDatasetParsing import getLabeledRSTData, getLabel, getDatumText
from reviewDatasetParsing import getSavedRSTParsedData, labels , getParsedData, getPickledRSTSciKitDataLists

def makeOctavePlot(run = True):
    data = ([],[])
    count = 0

    for d in getParsedData():
        ratio,numPos,numTotal = getLabel(d, discreteLabel=False, minHelpResponses = 6)
        text = d['text']
        #ratioc = len([x for x in d['text'] if x.isupper()])/(1.0*len(d['text']))
        if ratio!='few' and ratio>0 and ratio<1 and len(text)<10000:
            data[0].append(ratio)
            #data[1].append(len([c for c in d['text'] if c in string.punctuation]))
            data[1].append(len(text))
            count+=1
            if count>10000:
                break
    to = open('octavePlot.m','w')
    to.write('x=%s;\ny=%s;\nh=figure;\nscatter(x,y);title("Length vs helpfulness ratio");\n'%(str(data[0]),str(data[1])))
    to.write('h=figure;\nhist(x,3);title("Histogram of helpfulness ratio");\n')
    to.write('print(h,"ratio.png");\n')
    to.write('pause')
    to.close()
    if run:
        system('octave octavePlot.m')

def checkFeatureCorr(threshold = 0.02):
      X = []
      y = []
      dictts = [] 
      labels = []
      #dictmaker = FullTextRSTFeatureExtractor(None)
      dictmaker = FullPickledRSTFeatureExtractor(None)
      features = set([])
      nums, texts, ilabels = getPickledRSTSciKitDataLists(False)
      for num,text,label in zip(nums,texts,ilabels):
          dictmaker.setInstanceNums([num])
          dictt = dictmaker.getFeatures(text)
          dictts.append(dictt)
          for key in dictt.keys():
              features.add(key)
          labels.append(label)
      mapping = {}
      for i, feature in enumerate(features):
          mapping[feature] = i
          X.append([])
      for dictt,label in zip(dictts, labels):
          for feature in features:
              if feature in dictt:
                  X[mapping[feature]].append(dictt[feature])
              else:
                  X[mapping[feature]].append(0)
          y.append(labels.index(label))
      print 'Feature,Correlation,Probability of uncorellated data getting correlation'
      for feature in features:
          c,p = pearsonr(X[mapping[feature]],y)

          if p < threshold:
              print '%s,%f,%f'%(feature,c,p)
      #to = open('octaveCorrelate.m','w')
      #to.write('pkg load statistics\n')
      #to.write('x=%s;\ny=%s;\nh=figure;\ncorr(x,y)\n'%(str(X).replace('], [','];['),str(y).replace(',',';')))
      #to.close()
      #if run:
      #    system('octave octaveCorrelate.m')

class RSTCovarianceMatrixMaker(FullTextRSTFeatureExtractor):
    def __init__(self):
        self.vectorizer = DictVectorizer(dtype=float, sparse=True)
        self.atInstance = 0 
        self.instanceNums = []

    def fit(self, X, y=None):
        self.vectorizer.fit([self.getDict(d) for d in X],y)
        return self

    def getDict(self, datum):
        base = {'ratio':datum['ratio']}
        base.update(self.getTextFeatures(getDatumText(datum)))
        return base

    def setInstanceNums(self, nums):
        self.instanceNums = nums

    def setInstance(self, num):
        self.atInstance = num

    def transform(self, X, y=None):
        return self.vectorizer.transform([self.getDict(d) for d in X])

def printSciKitCovarianceMatrixs():
      #does not work, ValueError: setting an array element with a sequence.
      xMaker = RSTCovarianceMatrixMaker()
      nums, data, ilabels = getLabeledRSTData(False)
      for i,d in enumerate(data):
          d['ratio'] = ilabels[i]
      xMaker.setInstanceNums(nums)
      xMaker.fit(data)
      X = xMaker.transform(data)
      correlator = EmpiricalCovariance()
      correlator.fit(X)

      print correlator.covariance_
 
if __name__ == '__main__':
    checkFeatureCorr()
    #makeOctavePlot()
