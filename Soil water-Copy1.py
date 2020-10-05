#!/usr/bin/env python
# coding: utf-8

# In[1]:


#predictiin soil water using machine learning algorithms
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[2]:


import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy,pandas
import io


# In[3]:


sim=pd.read_csv('lucy_py.csv', encoding='latin')
sim.shape


# In[4]:


sim.describe()


# In[5]:


sim["depth"]=pd.Categorical(sim["depth"])


# In[6]:


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[7]:


names = ['ripper', 'plough', 'basins','depth']


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import countplot
from matplotlib.pyplot import figure ,show
sns.set()
sns.set(style="ticks",color_codes=True)


# In[9]:


# class distribution
print(sim.groupby('depth').describe())


# In[10]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


# box and whisker plots
width=17
height=20
figure(figsize=(width,height))
sim.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[12]:


## histograms

sim.hist()
plt.show()


# In[13]:


# scatter plot matrix
scatter_matrix(sim)
plt.show()


# In[61]:


# Split-out validation dataset
array = sim.values
X = array[:,0:3]
Y = array[:,3]
validation_size = 0.2
seed = 1234
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[62]:


# Test options and evaluation metric
seed = 1234
scoring = 'accuracy'


# In[63]:


# Spot Check Algorithms// appending all algorithms as a single function
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    


# In[64]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[66]:


# Make predictions on validation dataset linear discriminant analysis Linear discriminant analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[63]:


## creating visuals...
##### count plot.
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import countplot
from matplotlib.pyplot import figure ,show
sns.set()
sns.set(style="ticks",color_codes=True)
###################################################################
width=4
height=5
figure(figsize=(width,height))
sns.countplot(predictions);
plt.show()
print(predictions)


# In[67]:


# K Neighbors Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

########################################

width=4
height=5
figure(figsize=(width,height))
sns.countplot(predictions);
plt.show()
print(predictions)


# In[68]:


# Create confusion matrix
confusion_mat = confusion_matrix(Y_validation, predictions)


# Visualize confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()


# In[69]:


# Support Vector machine
svm=SVC(gamma='auto')
svm.fit(X_train, Y_train)
predictions =svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[70]:


# DecisionTreeClassifier()
lr=LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train, Y_train)
predictions =lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[71]:


###LogisticRegression(solver='liblinear', multi_class='ovr')
lr=DecisionTreeClassifier()
lr.fit(X_train, Y_train)
predictions =lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[72]:


#####GaussianNB()
nb=GaussianNB()
nb.fit(X_train, Y_train)
predictions =nb.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




