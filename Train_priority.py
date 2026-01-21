from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

nltk.download('stopwords', quiet=True)

# Caricamento dataset
df = pd.read_csv('data/dataset_clean.csv')
print(df.head())

X = df["full_text"]
y_priority = df["priority"]

# Train-test split
X_train_prio, X_test_prio, y_train_prio, y_test_prio = train_test_split(X, y_priority, test_size=0.2, random_state=42, stratify=y_priority)

# TF-IDF
italian_stopwords = stopwords.words('italian')
vectorizer = TfidfVectorizer(stop_words=italian_stopwords, ngram_range=(1,2), min_df=2, max_df=0.9)

X_train_prio_tfidf = vectorizer.fit_transform(X_train_prio)
X_test_prio_tfidf = vectorizer.transform(X_test_prio)

# SVM Lineare
svm_linear_model_prio = LinearSVC(random_state=42, class_weight='balanced')
svm_linear_model_prio.fit(X_train_prio_tfidf, y_train_prio)
y_pred_prio_linear = svm_linear_model_prio.predict(X_test_prio_tfidf)

accuracy_prio_linear = accuracy_score(y_test_prio, y_pred_prio_linear)
f1_prio_linear = f1_score(y_test_prio, y_pred_prio_linear, average='macro') 
cm_linear = confusion_matrix(y_test_prio, y_pred_prio_linear)

print("SVM Lineare - Priorità")
print("Accuracy:", accuracy_prio_linear)
print("F1 Score (Macro):", f1_prio_linear)
print("Confusion Matrix:\n", cm_linear)
print("\nClassification Report:\n", classification_report(y_test_prio, y_pred_prio_linear))

# SVM RBF
svm_rbf_model_prio = SVC(kernel='rbf', C=1, random_state=42, class_weight='balanced')
svm_rbf_model_prio.fit(X_train_prio_tfidf, y_train_prio)
y_pred_prio_rbf = svm_rbf_model_prio.predict(X_test_prio_tfidf)

accuracy_prio_rbf = accuracy_score(y_test_prio, y_pred_prio_rbf)
f1_prio_rbf = f1_score(y_test_prio, y_pred_prio_rbf, average='macro')
cm_rbf = confusion_matrix(y_test_prio, y_pred_prio_rbf)

print("SVM RBF - Priorità")
print("Accuracy:", accuracy_prio_rbf)
print("F1 Score (Macro):", f1_prio_rbf)
print("Confusion Matrix:\n", cm_rbf)
print("\nClassification Report:\n", classification_report(y_test_prio, y_pred_prio_rbf))

# MLP (Rete Neurale)
mlp_model_prio = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_model_prio.fit(X_train_prio_tfidf, y_train_prio)
y_pred_prio_mlp = mlp_model_prio.predict(X_test_prio_tfidf)

accuracy_prio_mlp = accuracy_score(y_test_prio, y_pred_prio_mlp)
f1_prio_mlp = f1_score(y_test_prio, y_pred_prio_mlp, average='macro')
cm_mlp = confusion_matrix(y_test_prio, y_pred_prio_mlp)

print("MLP Classifier - Priorità")
print("Accuracy:", accuracy_prio_mlp)
print("F1 Score (Macro):", f1_prio_mlp)
print("Confusion Matrix:\n", cm_mlp)
print("\nClassification Report:\n", classification_report(y_test_prio, y_pred_prio_mlp))

# Confronto modelli
results = {
    'SVM Linear': {'accuracy': accuracy_prio_linear, 'f1_score': f1_prio_linear, 'confusion_matrix': cm_linear},
    'SVM RBF': {'accuracy': accuracy_prio_rbf, 'f1_score': f1_prio_rbf, 'confusion_matrix': cm_rbf},
    'MLP': {'accuracy': accuracy_prio_mlp, 'f1_score': f1_prio_mlp, 'confusion_matrix': cm_mlp}
}

# Bar Chart
model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]

plt.figure()
plt.bar(model_names, accuracies)
plt.title("Confronto accuracy modelli - Priorità")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()

# Confusion Matrix
for model in model_names:
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=results[model]['confusion_matrix'])
    disp.plot()
    plt.title(f"Confusion Matrix - {model}")
    plt.show()

# Salvataggio modelli
Path("models").mkdir(exist_ok=True)

joblib.dump(vectorizer, 'models/tfidf_priority.pkl')
joblib.dump(svm_linear_model_prio, 'models/svm_priority.pkl')
joblib.dump(svm_rbf_model_prio, 'models/svm_rbf_priority.pkl')
joblib.dump(mlp_model_prio, 'models/mlp_priority.pkl')
