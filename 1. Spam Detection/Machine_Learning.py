import numpy as np
import sys
import sklearn
import pandas as pd
import nltk
import pickle

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import VotingClassifier


print("Python {}".format(sys.version))
print("NLTK {}".format(nltk.__version__))
print("sklearn {}".format(sklearn.__version__))
print("np {}".format(np.__version__))
print("pd {}".format(pd.__version__))

#Importing Dataset
print("Libraries are imported Successfully\nLoading Dataset")
df =pd.read_csv('Dataset\\spam2.csv',header=None,encoding='ISO-8859-1')

#Preprocessing the data
classes=df[0]
text_message=df[1]

encoder=LabelEncoder()
Y=encoder.fit_transform(classes)

print("Data Loaded\n2. Preprocessing the Data")

def Preprocessing(text_message):
    #use regulare expression email,phone number, other number symbols

    # replace email address with 'email'

    processed=text_message.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddr')

    # replace web address with 'webaddress'

    processed=processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

    #replacing money symbols with 'moneysymb'

    processed=processed.str.replace(r'£|\$', 'moneysymb')

    #replace 10 digit phone number with phonenumber

    processed=processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')

    #replace normal number with 'numbr'

    processed=processed.str.replace(r'\d+(\.\d+)?', 'numbr')

    # reomve punctuation
    processed=processed.str.replace(r'[^\w\d\s]',' ')

    #replace whitespace between terms with single space
    processed=processed.str.replace(r'\s+', ' ')

    #replace leading and triling whitespaces
    processed=processed.str.replace(r'^\s+|\s+?$',' ')

    #changing the world in lower case
    processed=processed.str.lower()

    #remove stop words from text message
    stop_words=set(stopwords.words('english'))

    processed=processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

    # remove word strems using a Porter stemer(ing,ed,etc)
    ps=nltk.PorterStemmer()

    processed=processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

    return processed
print(type(text_message))
processed=Preprocessing(text_message)
print("Preprocessing Done\n3.Generating Features")

#create a bag of word
all_words=[]
for message in processed:
    words=word_tokenize(message)
    for w in words:
        all_words.append(w)
all_words=nltk.FreqDist(all_words)

#print the total number of words and 15 most common words

print('Number of Words: {}'.format(len(all_words)))
print('Most Common Words:{}'.format(all_words.most_common(15)))

# all_words=all_words.most_common(2000)
all_words={k: v for k, v in sorted(all_words.items(), key=lambda item: item[1])}
word_features=[]
for x in list(reversed(list(all_words))):
    word_features.append(x)

f1 = open('my_feature.pickle', 'wb')
pickle.dump(word_features, f1)
f1.close()
# Define a feature function

def find_features(message):
    words=word_tokenize(message)
    features={}
    for word in word_features:
        features[word]=(word in words)
    return features
# Lets see some results
features=find_features(processed[0])
for key,value in features.items():
    if value==True:
        print(key,value)


# find freatures for all message
messages = list(zip(processed,Y))

#define a seed for reproducibility
seed=1
np.random.seed=seed
np.random.shuffle(messages)

# call find_features function for each sms message
features_sets=[(find_features(text),label) for (text,label) in messages]

training,testing=model_selection.train_test_split(features_sets,test_size=0.25,random_state=42)


# Scikit-learn Classifiers with NLTK
print("Feature generated \n4.Scikit-learn Classifiers with NLTK")



names=['K Nearest Neighbores','Decision Tree','Random Forest','Logistic Regression','SGD Classifier','Naive Bayes','SVM Linear']
classifiers=[
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel="linear")
    ]

models=list(zip(names,classifiers))

nltk_ensemble=SklearnClassifier(VotingClassifier(estimators=models,voting='hard',n_jobs=-1))
nltk_ensemble.train(training)
accuracy=nltk.classify.accuracy(nltk_ensemble,testing)*100
print('Ensemble Method Accuracy:{} '.format(accuracy))

txt_features,labels=zip(*testing)
prediction=nltk_ensemble.classify_many(txt_features)

#print a confusion matrix and classification reports

print(classification_report(labels,prediction))

pd.DataFrame(
    confusion_matrix(labels,prediction),
    index=[['actual','actual'],['ham','spam']],
    columns=[['Predicted','Predicted'],['ham','spam']]
)


f = open('my_classifier.pickle', 'wb')
pickle.dump(nltk_ensemble, f)
f.close()

