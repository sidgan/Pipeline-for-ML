#!usr/bin/env python 


from sklearn.kernel_approximation import RBFSampler
import sklearn.cluster 
import optparse
import copper 
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation 
from sklearn.cross_validation import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import random
import csv as csv
import pandas as pd 
import numpy as np 
import warnings 
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', SyntaxWarning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt 
import sklearn
from sklearn.ensemble import AdaBoostClassifier



def cal_score(method, clf, features_test, target_test):
		scores = cross_val_score(clf, features_test, target_test)
		print method + " : %f " % scores.max()
		#print scores.max()		

def main():
	type_of_problem = ""
	split = 0.3
	su_train = []
	su_test = [] 
	p = optparse.OptionParser()
	#take path of training data set 
	p.add_option('--path_train', '-p', default='/home/user')
	#take path of test data set
	p.add_option('--path_test', '-s', default ='/home/user')
	#what type of problem is it? regression/classification/clustering/dimensionality reduction
	p.add_option('--type_of_problem', '-t', default = 'no_input')
	#include cross validation true/false
	p.add_option('--cross_validation', '-v', default ='True')
	#take the numerical values 
	#p.add_option('--numerical_values', '-n')
	#specify target column
	p.add_option('--target', '-y')	
	options, arguments = p.parse_args()
	#when user does not provide type of problem 
	if options.type_of_problem == 'no_input':
		global type_of_problem
		#ask the user explicitly for type of problem
		print "Depending on type of problem, enter c for classification, clu for clustering, r for regression and d for dimensionality reduction."
		type_of_problem = raw_input("")
	#ask for value of cross validation if previously true 
	if options.cross_validation=='True':
		global split
		print "How much cross validation would you like? Enter a number between 0-1."
		split = raw_input("")
		split = float(split)

	#Ask for numerical values 
	print "Does the data set require imputation? Enter 1 for affirmative."
	imputation = raw_input()
	imputation = int(imputation)

	print "Enter labels for numerical values to be used for analysis, seperated by space"
	num_values = raw_input()
	num_values = num_values.split()
	
	#load from files 
	train = pd.read_csv(options.path_train)
	test = pd.read_csv(options.path_test)

	#Any categories for conversion
	#EXAMPLE:  sex mapped to male and female 
	print "Enter labels for categorical values to be converted into numerical values. If none then enter none"
	cat = raw_input()
	cat = cat.split()
	if cat == 'none': 
		# no categorical values 
		print ""
	else: 
		print "Preparing data."
		print "Converting categorical to numerical values."
		for each in cat: 
			global su_train 
		#	print each 
			i = train.get(each)
			i = pd.get_dummies(i, prefix=each)
			#print i 
			#print i.Sex_female.mean()
			if each == cat[0]:
				su_train = i 
			else: 
				su_train = pd.concat([su_train, i], axis=1)
			#print su_train
			#FOR TEST DATA 			
			t = test.get(each)
			t = pd.get_dummies(t, prefix=each)
			if each == cat[0]:
				su_test = i 
			else: 
				su_test = pd.concat([su_test, i], axis=1)
			#print su_train
			
	#load target values 
	target = train[options.target]
	
	#TRAINING DATA SET 
	#final data frame with categorical and numerical values 
	data = pd.concat([train.get(num_values), su_train], axis=1)
	
	#perform imputation if allowed - TRAINING DATA SET 
	if imputation == 1: 
		imp = data.dropna().mean()
		data = data.fillna(imp)
	
	#TEST DATA SET 
	#final data frame with categorical and numerical values 
	test = pd.concat([test.get(num_values), su_test], axis=1)

	#perform imputation if allowed 
	if imputation == 1: 
		print "Performing imputation."
		imp = test.dropna().mean()
		test = test.fillna(imp)

	#split the training data for cross validation 
	if options.cross_validation == 'True': 
		#perform cross validation if v==true 
		print "Splitting the training data with %f." % split 
		features_train, features_test, target_train, target_test = train_test_split(data, target, test_size=split, random_state=0)
		
	else: 
		#no cross validation 
		features_train = data 
		features_test = test 
		target_train = target
		target_test = 0
	print "Generating Model"	
	#diffrentiate on the basis of type of problem
	if type_of_problem == 'c':
		prob = 1
		#Naive Bayes 
		nb_estimator = GaussianNB()
		nb_estimator.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("NAIVE BAYES CLASSIFICATION",nb_estimator, features_test, target_test)
		#predictions = nb_estimator.predict(test)
		#SVC Ensemble

		#Ada boost 
		clf_ada = AdaBoostClassifier(n_estimators=100)
		params = {
			'learning_rate': [.05, .1,.2,.3,2,3, 5],
			'max_features': [.25,.50,.75,1],
			'max_depth': [3,4,5],
			}
		gs = GridSearchCV(clf_ada, params, cv=5, scoring ='accuracy', n_jobs=4)
		clf_ada.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("ADABOOST",clf_ada, features_test, target_test)
		#predictions = clf_ada.predict_proba(test)

		#RANDOM FOREST CLASSIFIER 
		rf = RandomForestClassifier(n_estimators=100)
		rf = rf.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("RANDOM FOREST CLASSIFIER",rf, features_test, target_test)
		#predictions = rf.predict_proba(test)

		#Gradient Boosting
		gb = GradientBoostingClassifier(n_estimators=100, subsample=.8)
		params = {
			'learning_rate': [.05, .1,.2,.3,2,3, 5],
			'max_features': [.25,.50,.75,1],
			'max_depth': [3,4,5],
		}
		gs = GridSearchCV(gb, params, cv=5, scoring ='accuracy', n_jobs=4)
		gs.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("GRADIENT BOOSTING",gs, features_test, target_test)
		#sorted(gs.grid_scores_, key = lambda x: x.mean_validation_score)
		#print gs.best_score_
		#print gs.best_params_
		#predictions = gs.predict_proba(test)
		#KERNEL APPROXIMATIONS - RBF 		
		rbf_feature = RBFSampler(gamma=1, random_state=1)
		X_features = rbf_feature.fit_transform(data)
		
		#SGD CLASSIFIER		
		clf = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
       		fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, verbose=0,
       warm_start=False)
		clf.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("SGD Regression",clf, features_test, target_test)


		#KN Classifier
		neigh = KNeighborsClassifier(n_neighbors = 1)
		neigh.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("KN CLASSIFICATION",neigh, features_test, target_test)
		#predictions = neigh.predict_proba(test)
	
		

	
		#Decision Tree classifier
		clf_tree = tree.DecisionTreeClassifier(max_depth=10)
		clf_tree.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("DECISION TREE CLASSIFIER",clf_tree, features_test, target_test)
		#predictions = clf_tree.predict_proba(test)
	
	if type_of_problem == 'r':
		prob = 2
		#LOGISTIC REGRESSION 
		logreg = LogisticRegression(C=3)
		logreg.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("LOGISTIC REGRESSION",logreg, features_test, target_test)
		#predictions = logreg.predict(test)
		# SUPPORT VECTOR MACHINES 
		clf = svm.SVC(kernel = 'linear')
		clf.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("LINEAR KERNEL",clf, features_test, target_test)
		#print clf.kernel
		#for sigmoid kernel
		clf= svm.SVC(kernel='rbf', C=2).fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("SVM RBF KERNEL",clf, features_test, target_test)		
		#predictions = clf.predict(test)
		#Lasso 
		clf = linear_model.Lasso(alpha=.1)
		clf.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("LASSO",clf, features_test, target_test)
		#elastic net 
		clf = linear_model.ElasticNet(alpha=.1, l1_ratio=.5, fit_intercept=True, normalize=False, precompute='auto',max_iter=1000, copy_X=True, tol =.0001, warm_start=False, positive=False)
		clf.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("ELASTIC NET",clf, features_test, target_test)
		#SGD REGRESSION	
		clf = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
       		fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, verbose=0,
       warm_start=False)
		clf.fit(features_train, target_train)
		if options.cross_validation == 'True': 
			cal_score("SGD Regression",clf, features_test, target_test)


	if type_of_problem == 'clu':
		prob = 3
		#MINI BATCH K MEANS CLUSTERING
		clf = sklearn.cluster.MiniBatchKMeans(init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
		clf.fit(features_train, target_train)

		#MEAN SHIFT
			
		clf = sklearn.cluster.MeanShift(bandwidth=None, seeds=[features_train, target_train], bin_seeding=False, min_bin_freq=1, cluster_all=True)
		#clf.fit([features_train, target_train])
		#clf.fit(data, target)		
		#if options.cross_validation == 'True': 
		#	cal_score("MEAN SHIFT CLUSTERING",clf, features_test, target_test)
		#K MEANS CLUSTERING	
		clf = sklearn.cluster.KMeans( init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)
		clf.fit(data)
		#if options.cross_validation == 'True': 
		#	cal_score("K MEANS CLUSTERING",clf, features_test, target_test)


	if type_of_problem == 'd':
		prob = 4
		#PCA
		pca = PCA(n_components=1)
		pca_train = pca.fit(data)
		pca_test = pca.transform(test)

	#perform classification 
	
	

#main ends here 
if __name__ == '__main__':
	main()
