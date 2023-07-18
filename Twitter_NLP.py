import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import spacy

# Define vectorizers and classifiers
vectorizers = {
    'Bag of Words': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

classifiers = {
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': MultinomialNB(),
    'K-NN': KNeighborsClassifier(),
    'SVM': SVC(probability=True)
}


# Download the English language model
#spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm",disable=['parser', 'ner'])

def remove_punc_dig(text : str):
    '''
    text : str 
    This function will remove all the punctuations and digits from the "text"
    '''
    to_remove = string.punctuation + string.digits
    cur_text = ""
    for i in range(len(text)):
        if text[i] in to_remove:
            cur_text += " "
        else:
            cur_text += text[i].lower()
    cur_text = " ".join(cur_text.split())
    return cur_text

def remove_stop_words(text: str):
    '''
    text : str
    This function will remove stop words like I,my,myself etc
    '''
    filtered_sentence = []
    for word in text.split(' '):
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 
    return " ".join(filtered_sentence)


def lemmatizer(text : str):
    '''
    text : str
    Applying lemmatization for all words of "text"
    '''
    return " ".join([token.lemma_ for token in nlp(text)])

#Load the data
data = pd.read_csv(r'D:\\Guvi\\Final_Project\\FinalBalancedDataset.csv')
#data=data.head(1000)
# print(data)

data['Tweet'] = data['Tweet'].apply(lambda x : remove_punc_dig(x))

data['Tweet'] = data['Tweet'].apply(lambda x : remove_stop_words(x))

data['Tweet'] = data['Tweet'].apply(lambda x : lemmatizer(x))

# Create a variable to capture the output
captured_output = ""

# Iterate over vectorizers and classifiers
for vectorizer_name, vectorizer in vectorizers.items():
    for classifier_name, classifier in classifiers.items():
        print(f'{vectorizer_name} + {classifier_name}')

        # Convert text data using the vectorizer
        features = vectorizer.fit_transform(data['Tweet'])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, data['Toxicity'], test_size=0.2, random_state=42)

        # Train the classifier and make predictions
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        # Evaluate the performance
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
        print('ROC-AUC:', roc_auc_score(y_test, predictions))
        fpr, tpr, _ = roc_curve(y_test, predictions)
        plt.plot(fpr, tpr)
        plt.show()

        print('-' * 50)
