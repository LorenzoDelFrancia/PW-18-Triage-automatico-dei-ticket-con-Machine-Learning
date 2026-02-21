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
y_category = df["category"]

# Train-test split
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_category, test_size=0.2, random_state=42, stratify=y_category)

# TF-IDF
italian_stopwords = stopwords.words('italian')
vectorizer = TfidfVectorizer(stop_words=italian_stopwords, ngram_range=(1,2), min_df=2, max_df=0.9)

X_train_cat_tfidf = vectorizer.fit_transform(X_train_cat)
X_test_cat_tfidf = vectorizer.transform(X_test_cat)

# SVM Lineare
svm_linear_model_cat = LinearSVC(random_state=42, class_weight='balanced')
svm_linear_model_cat.fit(X_train_cat_tfidf, y_train_cat)
y_pred_cat_linear = svm_linear_model_cat.predict(X_test_cat_tfidf)

accuracy_cat_linear = accuracy_score(y_test_cat, y_pred_cat_linear)
f1_cat_linear = f1_score(y_test_cat, y_pred_cat_linear, average='macro') 
cm_linear = confusion_matrix(y_test_cat, y_pred_cat_linear)

print("SVM Lineare - Categoria")
print("Accuracy:", accuracy_cat_linear)
print("F1 Score (Macro):", f1_cat_linear)
print("Matrice di confusione:\n", cm_linear)
print("\nReport di classificazione:\n", classification_report(y_test_cat, y_pred_cat_linear))

# SVM RBF
svm_rbf_model_cat = SVC(kernel='rbf', C=1, random_state=42, class_weight='balanced')
svm_rbf_model_cat.fit(X_train_cat_tfidf, y_train_cat)
y_pred_cat_rbf = svm_rbf_model_cat.predict(X_test_cat_tfidf)

accuracy_cat_rbf = accuracy_score(y_test_cat, y_pred_cat_rbf)
f1_cat_rbf = f1_score(y_test_cat, y_pred_cat_rbf, average='macro')
cm_rbf = confusion_matrix(y_test_cat, y_pred_cat_rbf)

print("SVM RBF - Categoria")
print("Accuracy:", accuracy_cat_rbf)
print("F1 Score (Macro):", f1_cat_rbf)
print("Matrice di confusione:\n", cm_rbf)
print("\nReport di classificazione:\n", classification_report(y_test_cat, y_pred_cat_rbf))

# MLP (Rete Neurale)
mlp_model_cat = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_model_cat.fit(X_train_cat_tfidf, y_train_cat)
y_pred_cat_mlp = mlp_model_cat.predict(X_test_cat_tfidf)

accuracy_cat_mlp = accuracy_score(y_test_cat, y_pred_cat_mlp)
f1_cat_mlp = f1_score(y_test_cat, y_pred_cat_mlp, average='macro')
cm_mlp = confusion_matrix(y_test_cat, y_pred_cat_mlp)

print("MLP Classifier - Categoria")
print("Accuracy:", accuracy_cat_mlp)
print("F1 Score (Macro):", f1_cat_mlp)
print("Matrice di confusione:\n", cm_mlp)
print("\nReport di classificazione:\n", classification_report(y_test_cat, y_pred_cat_mlp))

# Confronto modelli
results = {
    'SVM Lineare': {'accuracy': accuracy_cat_linear, 'f1_score': f1_cat_linear, 'confusion_matrix': cm_linear},
    'SVM RBF': {'accuracy': accuracy_cat_rbf, 'f1_score': f1_cat_rbf, 'confusion_matrix': cm_rbf},
    'MLP Classifier': {'accuracy': accuracy_cat_mlp, 'f1_score': f1_cat_mlp, 'confusion_matrix': cm_mlp}
}

# Grafico a barre
model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]

plt.figure()
plt.bar(model_names, accuracies)
plt.title("Confronto accuracy modelli - Categoria")
plt.xlabel("Modello")
plt.ylabel("Accuracy")
plt.ylim(0.95, 1.01)
plt.xticks(rotation=15)
plt.show()

# Matrice di confusione
for model in model_names:
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=results[model]['confusion_matrix'], display_labels=svm_linear_model_cat.classes_)
    disp.plot()
    disp.ax_.set_xlabel("Classe predetta")
    disp.ax_.set_ylabel("Classe reale")
    plt.title(f"Matrice di confusione - {model}")
    plt.show()

# Salvataggio modelli
Path("models").mkdir(exist_ok=True)

joblib.dump(vectorizer, 'models/tfidf_category.pkl')
joblib.dump(svm_linear_model_cat, 'models/svm_category.pkl')
joblib.dump(svm_rbf_model_cat, 'models/svm_rbf_category.pkl')
joblib.dump(mlp_model_cat, 'models/mlp_category.pkl')
