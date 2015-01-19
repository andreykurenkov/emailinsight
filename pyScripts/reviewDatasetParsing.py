import nltk
from os import listdir
from collections import Counter
import cPickle 
import gzip
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
labels = ['bad','okay','good']
keys = ['productId', 'userId','profileName', 'helpfulness', 'score', 'time', 'summary', 'text']

def getWords(text):
    text = text.replace('(','').replace(')','')
    text = ''.join(filter(lambda w: ord(w)<128, text))
    return list(itertools.chain.from_iterable([nltk.word_tokenize(sent[0:len(sent)-1]) for sent in sent_detector.tokenize(text)]))

def getNGrams(words, n = 1):
    if n==1:
        return Counter(words)
    #if n==2:
    #    return Counter(nltk.bigrams(words))
    #if n==3:
    #    return Counter(nltk.trigrams(words))
    return Counter([' '.join(ngrams) for ngrams in map(lambda x: words[x:x+n],xrange(len(words)-n+1))])

def getParsedData(stateProgress = True):
    count = 0
    example = {}
    moviesFile = open('movies.txt')
    dontAdd = False
    for line in moviesFile:
        if line=='\n':
            count+=1
            if not ('text' in example and 'summary' in example):
                dontAdd = True
            if not dontAdd:
                yield example
            dontAdd = False
            example = {}
            if stateProgress and count%10000==0:
                print 'At review #%d'%count
        else:
            split = line.split(':')
            if len(split)<2:
                if stateProgress:
                    print 'Problem with review #%d'%count
                dontAdd = True
            else:
                key = split[0].split('/')[1]
                text = ''.join(split[1:]).lstrip()
                example[key] = text[0:len(text)-1].replace('<br />','')

def saveParsedData(numInstances, minHelpfulnessResponses = 3, name="parsed"):
    toFile = open(name,'w')
    data = [x for x in getParsedData(numInstances) if getLabel(x)[2]>=minHelpfulnessResponses]
    cPickle.dump(data,toFile)
    toFile.close()

def getSavedParsedData(name="parsed"):
    fromFile = open(name,'r')
    data = cPickle.load(fromFile)
    fromFile.close()
    return data

def getFeatures(instanceData, maxN):
    features = {}
    text = instanceData['text']
    textWords = getWords(text)
    for n in xrange(1,maxN):
        features.update(getNGrams(textWords,n))
    summary = instanceData['summary']
    sWords = getWords(summary)
    for n in xrange(1,maxN):
        counter = getNGrams(sWords,n)
        scounter = {}
        for key in counter:
            scounter[('summary',key)] = counter[key]
        features.update(scounter)
    features['textlength'] = len(text)
    return features

def getLabel(instanceData, discreteLabel=False, minHelpResponses = 3):
    if 'helpfulness' not in instanceData:
      return float('nan'),0,0
    hVals = instanceData['helpfulness'].split('/')
    numPos = float(hVals[0])
    numTotal = float(hVals[1])
    if numTotal < minHelpResponses:
        return 'few',numPos,numTotal
    ratio = numPos/numTotal
    if not discreteLabel:
        return ratio,numPos,numTotal
    labelType = int(round(ratio/0.33333))
    if labelType>3:
        return float('nan'),numPos,numTotal
    label = labels[labelType-1]
    return label,numPos,numTotal

def getLabeledData(stateProgress = True, discreteLabels=False, allowFew = False):
    dataset = []
    count = 0
    for datum in getParsedData(stateProgress=stateProgress):
        label,numPos,numTotal = getLabel(datum, discreteLabels)
        if allowFew or label!='few':
            count = count + 1
            if stateProgress and count%1000==0:
                print 'At %d instances'%count
            datum['total'] = numTotal
            yield (datum,label)

def getDatumText(datum):
    return datum['text']+' '+datum['summary'].replace(' ','-s s-')

def getSciKitData(stateProgress = True, discreteLabels=False):
    for datum,label in getLabeledData(stateProgress = stateProgress, discreteLabels=discreteLabels):
        text = getDatumText(datum)#getFeatures(datum,maxN)
        #text = ''.join(filter(lambda w: ord(w)<128, text))
        yield (text,label)

def getDictInstances(maxN = 3, stateProgress = True, discreteLabels=False):
    for datum,label in getLabeledData(stateProgress = stateProgress, discreteLabels=discreteLabels):
        features = getFeatures(datum,maxN)
        #text = ''.join(filter(lambda w: ord(w)<128, text))
        yield (features,label)

def getSciKitInstanceFromDict(instances):
    encoder = LabelEncoder()
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    X, y = list(compat.izip(*instances))
    X = vectorizer.fit_transform(X)
    y = encoder.fit_transform(y)
    return X,y


def getAsSciKit(texts,labels,vectorizer, encoder = None,selector = None):
    X = vectorizer.transform(texts)
    y = np.array(labels).astype(float) if encoder is None else encoder.transform(labels)

    #This is now done with FeautureUnion
    #extraFeatures = [getExtraSciKitFeatures(text) for text in texts]
    #Xextra = dictvectorizer.transform(extraFeatures)
    #X = sc.sparse.hstack([X,Xextra])

    if selector!=None:
         X = selector.transform(X)
         vocab = np.asarray(vectorizer.get_feature_names())[selector.get_support()]
         vectorizer.vocabulary = vocab
    return X,y

def getBestWords(numInstances=1000,num=1000):
    instances = []
    for instance in getDictInstances(stateProgress = True, discreteLabels=True):
        instances.append(instance)
        if len(instances)==numInstances:
            selection = naivebayes.NaiveBayesClassifier.train(instances)
            return selection.most_informative_features(n=num)

def getLabeledRSTData(discreteHelpfulness, stateProgress=False, useFew = False):
    nums = []
    files = listdir('./rstParsed')
    for f in files:
        parts = f.split('.')
        if parts[1] == 'brackets':
            num = int(parts[0].replace('review',''))
            nums.append(num)
    nums = sorted(nums)

    count = 0
    count2 = 0
    for datum,label in getLabeledData(stateProgress = stateProgress, discreteLabels=discreteHelpfulness, allowFew=True):
        count = count+1
        if count > nums[-1]:
            break
        if count == nums[count2]:
          if label == 'few' and not useFew:
            nums.remove(nums[count2])
          else:
            yield nums[count2],datum,label
            count2 += 1

def getLabeledPickledRSTData(discreteHelpfulness, stateProgress=False, useFew = False):
    nums = []
    files = listdir('./output_trees')
    for f in files:
        parts = f.split('.')
        if parts[-1]=='gz':
            num = int(parts[0].replace('review',''))
            nums.append(num)
    nums = sorted(nums)

    count = 0
    count2 = 0
    for datum,label in getLabeledData(stateProgress = stateProgress, discreteLabels=discreteHelpfulness, allowFew=True):
        count = count+1
        if count > nums[-1]:
            break
        if count == nums[count2]: 
          if label == 'few' and not useFew:
            nums.remove(nums[count2])
          else:
            yield nums[count2],datum,label
            count2 += 1

def getLabeledRSTLists(discreteHelpfulness):
    nums = []
    data = []
    labels = []
    for num,datum,label in getRSTSciKitData(discreteHelpfulness):
        nums.append(num)
        data.append(datum)
        labels.append(label)
    return nums,data,labels

def getLabeledPickledRSTLists(discreteHelpfulness):
    nums = []
    data = []
    labels = []
    for num,datum,label in getPickledLabeledRSTData(discreteHelpfulness):
        nums.append(num)
        data.append(datum)
        labels.append(label)
    return nums,data,labels

def getRSTSciKitData(discreteHelpfulness):
    for num,datum,label in getLabeledRSTData(discreteHelpfulness):
        text = getDatumText(datum)
        yield num,text,label

def getPickledRSTSciKitData(discreteHelpfulness):
    for num,datum,label in getLabeledPickledRSTData(discreteHelpfulness):
        text = getDatumText(datum)
        yield num,text,label

def getRSTSciKitDataLists(discreteHelpfulness):
    nums = []
    texts = []
    labels = []
    for num,text,label in getRSTSciKitData(discreteHelpfulness):
        nums.append(num)
        texts.append(text)
        labels.append(label)
    return nums,texts,labels

def getPickledRSTSciKitDataLists(discreteHelpfulness):
    nums = []
    texts = []
    labels = []
    for num,text,label in getPickledRSTSciKitData(discreteHelpfulness):
        nums.append(num)
        texts.append(text)
        labels.append(label)
    return nums,texts,labels

def saveRSTParsedData(discrete, name='rstdata'):
    toFile = open(name,'w')
    num,text,label = getRSTSciKitDataLists(discrete)
    cPickle.dump((num,text,label),toFile)
    toFile.close()

def getSavedRSTParsedData(name='rstdata'):
    fromFile = open(name,'r')
    data = cPickle.load(fromFile)
    fromFile.close()
    return data

def savePickeldRSTParsedData(numInstances, name='rstdata'):
    toFile = open(name,'w')
    num,text,label = getRSTSciKitDataLists(discrete)
    cPickle.dump((num,text,label),toFile)
    toFile.close()

def getSavedPickeldRSTParsedData(name='rstdata'):
    fromFile = open(name,'r')
    data = cPickle.load(fromFile)
    fromFile.close()
    return data

def getPickledTree(path):
    try:
      fromF = gzip.open(path,'r')
      tree = cPickle.load(fromF)
      return tree 
    except:
      return None

