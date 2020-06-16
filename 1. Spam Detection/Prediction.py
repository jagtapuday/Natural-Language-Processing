import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
f1=open('my_feature.pickle', 'rb')
word_features = pickle.load(f1)
f1.close()
def find_features(message):
    words=word_tokenize(message)
    features={}
    for word in word_features:
        features[word]=(word in words)
    return features

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

