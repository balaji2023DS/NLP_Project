import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# import gensim
# from gensim import corpora
# import numpy as np

# nltk.download('stopwords')
# nltk.download('wordnet')

warnings.filterwarnings('ignore')

# Function for text preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase and tokenize
    words = text.lower().split()
    # Remove stopwords and perform stemming and lemmatization
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
    return " ".join(words)

# # Function to perform LDA on the preprocessed text data
# def perform_lda(text, num_topics=5):
#     dictionary = corpora.Dictionary(text)
#     corpus = [dictionary.doc2bow(doc) for doc in text]
#     lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
#     return lda_model

# Function to load the dataset
def load_dataset():
    return pd.read_csv(r'D:\\Guvi\\Final_Project\\FinalBalancedDataset.csv')

# Function to convert text using different NLP methods
def convert_text(df, method):
    df['processed_tweet'] = df['Tweet'].apply(preprocess_text)
    text = df['processed_tweet']
    if method == 'Bag of Words':
        vectorizer = CountVectorizer()
    elif method == 'TF-IDF':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid NLP method selected.")

    X = vectorizer.fit_transform(text)
    return X


# Function to train and predict using different classifiers
def train_predict(classifier, X_train, y_train, X_test):
    if classifier == 'Decision Trees':
        clf = DecisionTreeClassifier()
    elif classifier == 'Random forest':
        clf = RandomForestClassifier()
    elif classifier == 'Naive Bayes Model':
        clf = MultinomialNB()
    elif classifier == 'K-NN Classifier':
        clf = KNeighborsClassifier()
    elif classifier == 'SVM':
        clf = SVC(probability=True)
    else:
        raise ValueError("Invalid classifier selected.")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


# Function to calculate metrics
def calculate_metrics(y_test, y_pred, y_prob):
    precision, recall, f1_score, _ = classification_report(y_test, y_pred, output_dict=True)['1'].values()
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    return precision, recall, f1_score, conf_matrix, roc_auc

# Function to plot metrics for each classifier
def plot_metrics(metrics_dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metrics_dict.keys(), metrics_dict.values())
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Metrics for each Classifier")
    plt.ylim(0, 1)
    st.pyplot(fig)

# Function to plot ROC curve for selected classifier
def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot(fig)

# Function to plot confusion matrix as heatmap
def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', square=True, cbar=False,
                xticklabels=['Non-Toxic', 'Toxic'],
                yticklabels=['Non-Toxic', 'Toxic'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("Toxic Tweets Classifier")

    # Load the dataset
    df=load_dataset()
    #df=df.head(10000)

    # Dropdown to choose NLP method
    nlp_method = st.sidebar.selectbox("Choose NLP Method", ('Select','Bag of Words', 'TF-IDF'))
    if(nlp_method == 'Select'):
        st.warning('Please select a NLP Method')
        return

    # Convert text using selected NLP method
    X = convert_text(df, nlp_method)
    y = df['Toxicity']

    # Perform LDA on the preprocessed text data
    # if nlp_method == 'Bag of Words' or nlp_method == 'TF-IDF':
    #     lda_input = [tweet.split() for tweet in df['processed_tweet']]  # Assuming 'processed_tweet' column contains preprocessed text
    #     lda_model = perform_lda(lda_input)  # Perform LDA with list of tokenized texts
    #
    #     # Get the topic distribution for each document
    #     corpus = [lda_model.id2word.doc2bow(doc) for doc in lda_input]
    #     topic_distribution = [lda_model.get_document_topics(doc) for doc in corpus]
    #
    #     # Convert LDA topic distribution to a dense array
    #     X_lda = np.zeros((len(topic_distribution), lda_model.num_topics))
    #     for i, doc_topics in enumerate(topic_distribution):
    #         for topic, prob in doc_topics:
    #             X_lda[i, topic] = prob

        # # Combine LDA features with existing features
        # X_combined = np.hstack((X.toarray(), X_lda))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dropdown to choose classifier
    classifier = st.sidebar.selectbox("Choose Classifier", ('Select',
    'Decision Trees', 'Random forest', 'Naive Bayes Model', 'K-NN Classifier', 'SVM'))

    if classifier == 'Select':
        st.warning("Please select a classifier.")
        return

    # Train and predict using selected classifier
    y_pred, y_prob = train_predict(classifier, X_train, y_train, X_test)

    # Calculate metrics
    precision, recall, f1_score, conf_matrix, roc_auc = calculate_metrics(y_test, y_pred, y_prob)

    # Display results
    st.write(f"**Classifier:** {classifier}")
    st.write(f"**NLP Method:** {nlp_method}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1-Score:** {f1_score:.2f}")
    # st.write("**Confusion Matrix:**")
    # st.write(conf_matrix)
    st.write(f"**RoC - AUC curve:** {roc_auc:.2f}")

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)

    # # Plot metrics for each classifier
    # metrics_dict = {
    #     'Precision': precision,
    #     'Recall': recall,
    #     'F1-Score': f1_score
    # }
    # plot_metrics(metrics_dict)

    # Plot ROC curve for selected classifier
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_roc_curve(fpr, tpr, roc_auc)

if __name__ == "__main__":
    main()
