import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    auc, roc_curve
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def main():
    st.title("Classifier Evaluation with PCA and ROC Analysis")

    # Load the features dataset
    features_path = r"D:\\Guvi\\Final_Project\\Kannada_MNIST\\Kannada_MNIST_npz\\Dig_MNIST\\X_dig_MNIST.npz"
    features_data = np.load(features_path)
    X = features_data['arr_0']

    # Load the labels dataset
    labels_path = r"D:\\Guvi\\Final_Project\\Kannada_MNIST\\Kannada_MNIST_npz\\Dig_MNIST\\y_dig_MNIST.npz"
    labels_data = np.load(labels_path)
    y = labels_data['arr_0']

    # Convert the labels to integers
    y = y.astype(int)

    # Define the component sizes to loop through
    component_sizes = [15, 20, 25, 30]

    selected_classifier = st.sidebar.selectbox("Select Classifier", ['Decision Trees', 'Random Forest', 'Naive Bayes', 'K-NN Classifier', 'SVM'])
    selected_component_size = st.sidebar.selectbox("Select Component Size", component_sizes)

    st.write(f"Selected Classifier: {selected_classifier}")
    st.write(f"Selected Component Size: {selected_component_size}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the training data to (num_samples, num_features)
    num_samples, height, width = X_train.shape
    X_train_reshaped = X_train.reshape(X_train.shape[0], height * width)

    # Perform PCA on training data
    pca = PCA(n_components=selected_component_size)
    X_train_pca = pca.fit_transform(X_train_reshaped)

    # Reshape the test data to (num_samples, num_features)
    X_test_reshaped = X_test.reshape(X_test.shape[0], height * width)

    # Transform the test data using the same PCA transformation
    X_test_pca = pca.transform(X_test_reshaped)

    # Get the selected classifier
    classifiers = {
        'Decision Trees': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB(),
        'K-NN Classifier': KNeighborsClassifier(),
        'SVM': SVC()
    }

    classifier = classifiers[selected_classifier]

    st.write(f"Classifier: {selected_classifier}")

    # Fit the classifier on the training data
    classifier.fit(X_train_pca, y_train)

    # Predict labels for the test data
    y_pred_dt = classifier.predict(X_test_pca)

    # Calculate evaluation metrics
    # Compute the probabilities for each class using the one-hot encoded format
    y_scores_dt = np.eye(10)[y_pred_dt]

    # Reshape y_test to match the shape of y_scores_dt
    y_test_reshaped = np.eye(10)[y_test]

    # Calculate Precision, Recall, and F1-Score
    if (selected_classifier == 'Decision Trees'):
        precision = precision_score(y_test, y_pred_dt, average='macro')
        recall = recall_score(y_test, y_pred_dt, average='macro')
        f1 = f1_score(y_test, y_pred_dt, average='macro')
        accuracy = accuracy_score(y_test, y_pred_dt)
    else:
        precision = precision_score(y_test, y_pred_dt, average='weighted')
        recall = recall_score(y_test, y_pred_dt, average='weighted')
        f1 = f1_score(y_test, y_pred_dt, average='weighted')
        accuracy = accuracy_score(y_test, y_pred_dt)

    # Calculate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_dt)

    # Calculate ROC-AUC score with one-vs-rest strategy
    roc_auc = roc_auc_score(y_test_reshaped, y_scores_dt, multi_class='ovr')

    # Compute the ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc_dict = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_test_reshaped[:, i], y_scores_dt[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curves for each class
    fig, ax = plt.subplots()
    for i in range(10):
        ax.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc_dict[i]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    # Print the results
    st.write("--------------------------------")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1-Score: {f1}")
    st.write(f"Confusion Matrix:\n{cm}")
    st.write("--------------------------------")

if __name__ == "__main__":
    main()
