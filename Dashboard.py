import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Caricamento dei modelli salvati
tfidf_cat = joblib.load("models/tfidf_category.pkl")
tfidf_prio = joblib.load("models/tfidf_priority.pkl")

models_cat = {
    "SVM Lineare": joblib.load("models/svm_category.pkl"),
    "SVM RBF": joblib.load("models/svm_rbf_category.pkl"),
    "MLP Classifier": joblib.load("models/mlp_category.pkl")
}

models_prio = {
    "SVM Lineare": joblib.load("models/svm_priority.pkl"),
    "SVM RBF": joblib.load("models/svm_rbf_priority.pkl"),
    "MLP Classifier": joblib.load("models/mlp_priority.pkl")
}

# Funzione per trovare le parole piu influenti
def top_words(text, tfidf, model, top_n=5):
    vec = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    pred = model.predict(vec)[0]
    class_pos = list(model.classes_).index(pred)
    
    weights = model.coef_[class_pos]
    scores = vec.toarray()[0] * weights
    top_idx = np.argsort(np.abs(scores))[-top_n:][::-1]
    
    return [feature_names[i] for i in top_idx if scores[i] != 0]


# Titolo
st.title("Classificazione automatica dei ticket")

# --- SEZIONE 1: PREDIZIONE SINGOLA ---
st.header("Predizione singolo ticket")

testo = st.text_area("Inserisci il testo del ticket")

if st.button("Predici"):
    if testo.strip() == "":
        st.warning("Inserisci un testo")
    else:
        # Trasformo il testo
        X_cat = tfidf_cat.transform([testo])
        X_prio = tfidf_prio.transform([testo])
        
        # Predizioni categoria
        st.subheader("Categoria")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred = models_cat["SVM Lineare"].predict(X_cat)[0]
            st.metric("SVM Lineare", pred)
        with col2:
            pred = models_cat["SVM RBF"].predict(X_cat)[0]
            st.metric("SVM RBF", pred)
        with col3:
            pred = models_cat["MLP Classifier"].predict(X_cat)[0]
            st.metric("MLP Classifier", pred)
        
        # Predizioni priorita
        st.subheader("Priorità")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred = models_prio["SVM Lineare"].predict(X_prio)[0]
            st.metric("SVM Lineare", pred)
        with col2:
            pred = models_prio["SVM RBF"].predict(X_prio)[0]
            st.metric("SVM RBF", pred)
        with col3:
            pred = models_prio["MLP Classifier"].predict(X_prio)[0]
            st.metric("MLP Classifier", pred)
        
        # Parole influenti
        parole = top_words(testo, tfidf_cat, models_cat["SVM Lineare"])
        if parole:
            st.write("Parole più influenti:", ", ".join(parole))


# --- SEZIONE 2: PREDIZIONE BATCH ---
st.header("Predizione batch (CSV)")

with st.expander("Formato del dataset"):
    st.markdown("""
    **Opzione 1 - Dataset grezzo** (consigliato):
    | title | body | category* | priority* |
    |-------|------|-----------|-----------|
    | Problema stampante | La stampante non funziona | Tecnico | alta |
    
    **Opzione 2 - Dataset preprocessato**:
    | full_text | category* | priority* |
    |-----------|-----------|-----------|
    | problema stampante non funziona | Tecnico | alta |
    
    *Le colonne `category` e `priority` sono opzionali: se presenti, verranno mostrate le metriche di valutazione.*
    """)

file = st.file_uploader("Carica CSV", type="csv")

if file:
    df = pd.read_csv(file)
    
    # Controllo formato e preprocessing se necessario
    if "full_text" in df.columns:
        st.success("Rilevato dataset preprocessato")
    elif "title" in df.columns and "body" in df.columns:
        st.warning("Rilevato dataset grezzo - applico preprocessing...")
        # Preprocessing (stesso di Preprocessing.py)
        df["full_text"] = df["title"].astype(str) + " " + df["body"].astype(str)
        df["full_text"] = df["full_text"].str.lower()
        pattern = r"[!?,.;:()\[\]{}@%*/\\|=+]"
        df["full_text"] = df["full_text"].str.replace(pattern, " ", regex=True)
        df["full_text"] = df["full_text"].str.replace(r"\s+", " ", regex=True).str.strip()
        st.success("Preprocessing completato")
    else:
        st.error("Il CSV deve contenere la colonna 'full_text' oppure le colonne 'title' e 'body'")
        st.stop()
    
    X_cat = tfidf_cat.transform(df["full_text"])
    X_prio = tfidf_prio.transform(df["full_text"])
    
    # Aggiungo le predizioni al dataframe
    df["categoria_predetta"] = models_cat["SVM Lineare"].predict(X_cat)
    df["priorità_predetta"] = models_prio["SVM Lineare"].predict(X_prio)
    
    st.subheader("Anteprima")
    st.dataframe(df.head(10))
    
    # Se ci sono le etichette "category" o "priority", calcolo le metriche
    if "category" in df.columns or "priority" in df.columns:
        
        st.subheader("Valutazione modelli")
        
        # Categoria
        if "category" in df.columns:
            st.markdown("### Categoria")
            
            # Calcolo metriche per ogni modello
            risultati_cat = {}
            for nome, modello in models_cat.items():
                pred = modello.predict(X_cat)
                acc = accuracy_score(df["category"], pred)
                f1 = f1_score(df["category"], pred, average="macro")
                cm = confusion_matrix(df["category"], pred)
                risultati_cat[nome] = {"accuracy": acc, "f1": f1, "cm": cm}
            
            # Grafico a barre
            fig, ax = plt.subplots()
            nomi = list(risultati_cat.keys())
            acc_values = [risultati_cat[n]["accuracy"] for n in nomi]
            f1_values = [risultati_cat[n]["f1"] for n in nomi]
            
            x = np.arange(len(nomi))
            ax.bar(x - 0.2, acc_values, 0.4, label="Accuracy")
            ax.bar(x + 0.2, f1_values, 0.4, label="F1 Score")
            ax.set_xticks(x)
            ax.set_xticklabels(nomi)
            ax.set_ylim(0, 1)
            ax.set_title("Confronto modelli - Categoria")
            ax.legend()
            st.pyplot(fig)
            
            # Metriche
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, nome in enumerate(nomi):
                with cols[i]:
                    st.write(f"**{nome}**")
                    st.metric("Accuracy", f"{risultati_cat[nome]['accuracy']:.2%}")
                    st.metric("F1", f"{risultati_cat[nome]['f1']:.2%}")
            
            # Matrice di confusione
            st.write("Matrice di confusione")
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, nome in enumerate(nomi):
                with cols[i]:
                    fig, ax = plt.subplots()
                    disp = ConfusionMatrixDisplay(risultati_cat[nome]["cm"], display_labels=models_cat[nome].classes_)
                    disp.plot(ax=ax, cmap="Blues", colorbar=False)
                    ax.set_title(nome)
                    st.pyplot(fig)
        
        # Priorità
        if "priority" in df.columns:
            st.markdown("### Priorità")
            
            risultati_prio = {}
            for nome, modello in models_prio.items():
                pred = modello.predict(X_prio)
                acc = accuracy_score(df["priority"], pred)
                f1 = f1_score(df["priority"], pred, average="macro")
                cm = confusion_matrix(df["priority"], pred)
                risultati_prio[nome] = {"accuracy": acc, "f1": f1, "cm": cm}
            
            # Grafico a barre
            fig, ax = plt.subplots()
            nomi = list(risultati_prio.keys())
            acc_values = [risultati_prio[n]["accuracy"] for n in nomi]
            f1_values = [risultati_prio[n]["f1"] for n in nomi]
            
            x = np.arange(len(nomi))
            ax.bar(x - 0.2, acc_values, 0.4, label="Accuracy")
            ax.bar(x + 0.2, f1_values, 0.4, label="F1 Score")
            ax.set_xticks(x)
            ax.set_xticklabels(nomi)
            ax.set_ylim(0, 1)
            ax.set_title("Confronto modelli - Priorità")
            ax.legend()
            st.pyplot(fig)
            
            # Metriche
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, nome in enumerate(nomi):
                with cols[i]:
                    st.write(f"**{nome}**")
                    st.metric("Accuracy", f"{risultati_prio[nome]['accuracy']:.2%}")
                    st.metric("F1", f"{risultati_prio[nome]['f1']:.2%}")
            
            # Matrice di confusione
            st.write("Matrice di confusione")
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, nome in enumerate(nomi):
                with cols[i]:
                    fig, ax = plt.subplots()
                    disp = ConfusionMatrixDisplay(risultati_prio[nome]["cm"], display_labels=models_prio[nome].classes_)
                    disp.plot(ax=ax, cmap="Oranges", colorbar=False)
                    ax.set_title(nome)
                    st.pyplot(fig)
    
    else:
        st.info("Per vedere le metriche, carica un CSV con le colonne 'category' e 'priority'")
    
    # Download
    st.subheader("Download")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica CSV con predizioni", csv, "predizioni.csv", "text/csv")
