# Triage Automatico dei Ticket con Machine Learning

Progetto universitario per la classificazione automatica di ticket di supporto.

## Cos'è?

Un sistema che legge i ticket e predice automaticamente:
- **Categoria**: Tecnico, Amministrazione o Commerciale
- **Priorità**: Alta, Media o Bassa

## Prova il progetto

👉 **[Dashboard](https://triage-ticket.streamlit.app/)** - puoi testarlo direttamente online

## Come funziona

1. L'utente inserisce titolo e descrizione del ticket
2. Il testo viene pulito e trasformato in vettori (TF-IDF)
3. I modelli ML (SVM e MLP) fanno la predizione
4. Viene mostrata categoria e priorità

## File principali

- `Dashboard.py` - Interfaccia web con Streamlit
- `Preprocessing.ipynb` - Pulizia del dataset  
- `Train_category.ipynb` - Training modelli per categoria
- `Train_priority.ipynb` - Training modelli per priorità
- `models/` - Modelli già allenati

## Eseguire in locale

```bash
# Clona il repo
git clone https://github.com/LorenzoDelFrancia/PW-18-Triage-automatico-dei-ticket-con-Machine-Learning.git
cd PW-18-Triage-automatico-dei-ticket-con-Machine-Learning

# Installa dipendenze
pip install -r requirements.txt

# Avvia la dashboard
streamlit run Dashboard.py

```

## Risultati

**Categoria** → ~90-95% accuracy (funziona bene!)

**Priorità** → ~65-73% accuracy (più difficile perché la priorità è soggettiva)

## Tecnologie

- Python
- Scikit-learn (SVM, MLP, TF-IDF)
- Streamlit
- NLTK
- Pandas

---

*Progetto universitario - Lorenzo Del Francia*
