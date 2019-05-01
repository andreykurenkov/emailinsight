from kerasClassify import *
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import time

# Dataset tsv file path. Each line is an email
csvEmailsFilePath = "D:/views/Collage.Topics/Reports/classify/enron_6_email_folders_KAMINSKI.tsv";



def select_best_features(dataset, train_labels, num_best, verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset
    if verbose:
        print('\nSelecting %d best features\n'%num_best)
    selector = SelectKBest(chi2, k=num_best)
    X_train = selector.fit_transform(X_train,train_labels)
    X_test = selector.transform(X_test)
    return ((X_train, Y_train), (X_test, Y_test)),selector.scores_

def plot_feature_scores(feature_names,scores,limit_to=None,save_to=None,best=True):
    plt.figure()
    if best:
        plt.title("Best features")
    else:
        plt.title("Worst features")
    if limit_to is None:
        limit_to = len(features_names)
    #for some reason index 0 always wrong
    scores = np.nan_to_num(scores)
    if best:
        indices = np.argsort(scores)[-limit_to:][::-1]
    else:
        indices = np.argsort(scores)[:limit_to]
    #indices = np.argpartition(scores,-limit_to)[-limit_to:]
    plt.bar(range(limit_to), scores[indices],color="r", align="center")
    plt.xticks(range(limit_to),np.array(feature_names)[indices],rotation='vertical')
    plt.xlim([-1, limit_to])
    plt.ylabel('Score')
    plt.xlabel('Word')
    plt.show(block=False)
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight')

def plot_confusion_matrix(cm, label_names, title='Confusion matrix', cmap=plt.cm.Blues, save_to = None):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation='vertical')
    plt.yticks(tick_marks, label_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight')

def make_plot(x,y,title=None,x_name=None,y_name=None,save_to=None,color='b',new_fig=True):
    if new_fig:
        plt.figure()
    plot = plt.plot(x,y,color)
    if title is not None:
        plt.title(title)
    if x_name is not None:
        plt.xlabel(x_name)
    if y_name is not None:
        plt.ylabel(y_name)
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight')
    return plot

def make_plots(xs,ys,labels,title=None,x_name=None,y_name=None,y_bounds=None,save_to=None):
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    handles = []
    plt.figure()
    plt.hold(True)
    for i in range(len(labels)):
        plot, = make_plot(xs[i],ys[i],color=colors[i%len(colors)],new_fig=False)
        handles.append(plot)
    plt.legend(handles,labels)
    if title is not None:
        plt.title(title)
    if x_name is not None:
        plt.xlabel(x_name)
    if y_name is not None:
        plt.ylabel(y_name)
    if y_bounds is not None:
        plt.ylim(y_bounds)
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight')
    plt.hold(False)

def get_baseline_dummy(dataset,train_label_list,test_label_list,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset
    dummy = DummyClassifier()
    dummy.fit(X_train,train_label_list)
    predictions = dummy.predict(X_test)
    accuracy = accuracy_score(test_label_list,predictions)
    
    if verbose:
        print('Got baseline of %f with dummy classifier'%accuracy)

    return accuracy

def get_baseline_svm(dataset,train_label_list,test_label_list,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset
    linear = LinearSVC(penalty='l1',dual=False)
    grid_linear = GridSearchCV(linear, {'C':[0.1, 0.5, 1, 5, 10]}, cv=5)
    grid_linear.fit(X_train,train_label_list)
    accuracy = grid_linear.score(X_test, test_label_list)
    
    if verbose:
        print('Got baseline of %f with svm classifier'%accuracy)

    return accuracy

def get_baseline_knn(dataset,train_label_list,test_label_list,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset
    knn = KNeighborsClassifier(n_neighbors=100,n_jobs=-1)
    knn.fit(X_train,train_label_list)
    predictions = np.round(knn.predict(X_test))
    accuracy = accuracy_score(test_label_list,predictions)

    if verbose:
        print('Got baseline of %f with linear regression '%accuracy)

    return accuracy

def get_baseline_pa(dataset,train_label_list,test_label_list,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset
    classifier = PassiveAggressiveClassifier(n_jobs=-1,fit_intercept=True)
    classifier.fit(X_train,train_label_list)
    accuracy = classifier.score(X_test,test_label_list)
    
    if verbose:
        print('Got baseline of %f with Passive Aggressive classifier'%accuracy)

    return accuracy

def run_once(verbose=True,test_split=0.1,ftype='binary',num_words=10000,select_best=4000,num_hidden=512,dropout=0.5, plot=True,plot_prefix='',graph_to=None,extra_layers=0):
    features,labels,feature_names,label_names = get_ngram_data(num_words=num_words,matrix_type=ftype,verbose=verbose)
    num_labels = len(label_names)
    dataset,train_label_list,test_label_list = make_dataset(features,labels,num_labels,test_split=test_split)
    if select_best and select_best<num_words:
        dataset,scores = select_best_features(dataset,train_label_list,select_best,verbose=verbose)
    if plot and select_best:
        plot_feature_scores(feature_names, scores,limit_to=25, save_to=plot_prefix+'scores_best.png')
        plot_feature_scores(feature_names, scores,limit_to=25, save_to=plot_prefix+'scores_worst.png',best=False)
    predictions,acc = evaluate_mlp_model(dataset,num_labels,num_hidden=num_hidden,dropout=dropout,graph_to=graph_to, verbose=verbose,extra_layers=extra_layers)
    conf = confusion_matrix(test_label_list,predictions)
    conf_normalized = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    if verbose:
        print('\nConfusion matrix:')
        print(conf)
    if plot:
        plot_confusion_matrix(conf, label_names,save_to=plot_prefix+'conf.png')
        plot_confusion_matrix(conf_normalized, label_names, save_to=plot_prefix+'conf_normalized.png',title='Normalized Confusion Matrix')
    return dataset,train_label_list,test_label_list,acc

def test_features_words():
    #get emails once to pickle
    emails = get_emails(csvEmailsFilePath,verbose=False)

    types = ['binary','count','freq','tfidf']
    all_accs = []
    all_counts = []
    all_times = []
    all_baselines = []
    maxacc = 0
    maxtype = None
    for ftype in types:
        word_counts = range(500,3600,250)
        all_counts.append(word_counts)
        accs=[]
        times=[]
        baselines = []
        print('\nTesting learning for type %s with word counts %s\n'%(ftype,str(word_counts)))
        for word_count in word_counts:
            start = time.time()
            dataset,train_label_list,test_label_list,acc_one = run_once(num_words=word_count,ftype=ftype,plot=False,verbose=False,select_best=None)
            acc = (acc_one+sum([run_once(num_words=word_count,ftype=ftype,plot=False,verbose=False,select_best=None)[3] for i in range(4)]))/5.0
            end = time.time()
            elapsed = (end-start)/5.0
            times.append(elapsed)
            print('\nGot acc %f for word count %d in %d seconds'%(acc,word_count,elapsed))

            start = time.time()
            baseline = get_baseline_dummy(dataset,train_label_list,test_label_list,verbose=False) 
            baselines.append(baseline)
            end = time.time()
            belapsed = end-start
            print('Got baseline acc %f in %d seconds'%(baseline,belapsed))

            if acc>maxacc:
                maxacc = acc
                maxtype = ftype
            accs.append(acc)
        all_baselines.append(baselines)
        all_times.append(times)
        all_accs.append(accs)
        print('\nWord count accuracies:%s\n'%str(accs))
    make_plots(all_counts,all_accs,types,title='Test accuracy vs max words',y_name='Test accuracy',x_name='Max most frequent words',save_to='word_accs.png',y_bounds=(0,1))
    make_plots(all_counts,all_accs,types,title='Test accuracy vs max words',y_name='Test accuracy',x_name='Max most frequent words',save_to='word_accs_zoomed.png',y_bounds=(0.6,0.95))
    make_plots(all_counts,all_baselines,types,title='Baseline accuracy vs max words',y_name='Baseline accuracy',x_name='Max most frequent words',save_to='word_baseline_accs.png',y_bounds=(0,1))
    make_plots(all_counts,all_times,types,title='Time vs max words',y_name='Parse+test+train time (seconds)',x_name='Max most frequent words',save_to='word_times.png')
    print('\nBest word accuracy %f with features %s\n'%(maxacc,maxtype))

def test_hidden_dropout():
    #get emails once to pickle
    emails = get_emails(csvEmailsFilePath,verbose=False)

    dropouts = [0.25,0.5,0.75]
    all_accs = []
    all_counts = []
    all_times = []
    maxacc = 0
    maxh = 0
    for d in dropouts:
        hidden = [32,64,128,256,512,1024,2048]
        all_counts.append(hidden)
        accs=[]
        times=[]
        print('\nTesting learning for dropout %f with hidden counts %s\n'%(d,str(hidden)))
        for h in hidden:
            start = time.time()
            acc = sum([run_once(dropout=d,num_words=2500,num_hidden=h,plot=False,verbose=False,select_best=None)[3] for i in range(5)])/5.0
            end = time.time()
            elapsed = (end-start)/5.0
            times.append(elapsed)
            print('\nGot acc %f for hidden count %d in %d seconds'%(acc,h,elapsed))
            if acc>maxacc:
                maxacc = acc
                maxh = h
            accs.append(acc)
        all_times.append(times)
        all_accs.append(accs)
        print('\nWord count accuracies:%s\n'%str(accs))
    make_plots(all_counts,all_accs,['Droupout=%f'%d for d in dropouts],title='Test accuracy vs num hidden',y_name='Test accuracy',x_name='Number of hidden units',save_to='hidden_accs.png',y_bounds=(0,1))
    make_plots(all_counts,all_accs,['Droupout=%f'%d for d in dropouts],title='Test accuracy vs num hidden',y_name='Test accuracy',x_name='Number of hidden units',save_to='hidden_accs_zoomed.png',y_bounds=(0.8,1))
    make_plots(all_counts,all_times,['Droupout=%f'%d for d in dropouts],title='Time vs max words',y_name='Parse+test+train time (seconds)',x_name='Number of hidden units',save_to='hidden_times.png')
    print('\nBest word accuracy %f with hidden %d\n'%(maxacc,maxh))

def test_select_words(num_hidden=512):
    #get emails once to pickle
    emails = get_emails(csvEmailsFilePath,verbose=False)

    word_counts = [2500,3500,4500,5500]
    all_accs = []
    all_counts = []
    all_times = []
    maxacc = 0
    maxs = None
    for word_count in word_counts:
        select = [0.5,0.6,0.7,0.8,0.9]
        all_counts.append(select)
        accs=[]
        times=[]
        print('\nTesting learning for word count %d with selects %s\n'%(word_count,str(select)))
        for s in select:
            start = time.time()
            acc = sum([run_once(num_hidden=num_hidden,dropout=0.1,num_words=word_count,plot=False,verbose=False,select_best=int(s*word_count))[3] for i in range(5)])/5.0
            end = time.time()
            elapsed = (end-start)/5.0
            times.append(elapsed)
            print('\nGot acc %f for select ratio %f in %d seconds'%(acc,s,elapsed))
            if acc>maxacc:
                maxacc = acc
                maxs = s
            accs.append(acc)
        all_times.append(times)
        all_accs.append(accs)
        print('\nWord count accuracies:%s\n'%str(accs))
    make_plots(all_counts,all_accs,['Words=%d'%w for w in word_counts],title='Test accuracy vs ratio of words kept',y_name='Test accuracy',x_name='Ratio of best words kept',save_to='select_accs_%d.png'%num_hidden,y_bounds=(0,1))
    make_plots(all_counts,all_accs,['Words=%d'%w for w in word_counts],title='Test accuracy vs ratio of words kept',y_name='Test accuracy',x_name='Ratio of best words kept',save_to='select_accs_zoomed_%d.png'%num_hidden,y_bounds=(0.8,1))
    make_plots(all_counts,all_times,['Words=%d'%w for w in word_counts],title='Time vs ratio of words kept',y_name='Parse+test+train time (seconds)',x_name='Ratio of best words kept',save_to='select_times_%d.png'%num_hidden,y_bounds=(0,65))
    print('\nBest word accuracy %f with select %f\n'%(maxacc,maxs))

#test_features_words()
#test_hidden_dropout()
#test_select_words(128)
#test_select_words(32)
#test_select_words(16)
#run_once(num_words=10000,dropout=0.5,num_hidden=512, extra_layers=0,plot=True,verbose=True,select_best=4000)
features,labels,feature_names,label_names = get_ngram_data(emailsFilePath=csvEmailsFilePath,num_words=5000,matrix_type='tfidf',verbose=True,max_n=1)
#features,labels,label_names = get_sequence_data()
num_labels = len(label_names)
dataset,train_label_list,test_label_list = make_dataset(features,labels,num_labels,test_split=0.1)

# Feature selection (best 4000 features)
dataset,scores = select_best_features(dataset,train_label_list,4000,verbose=True)

# Unrem for baseline svm 
baseline = get_baseline_svm(dataset,train_label_list,test_label_list,verbose=True) 

# Unrem for keras 
# predictions,acc = evaluate_conv_model(dataset,num_labels,num_hidden=512,verbose=True,with_lstm=True)
