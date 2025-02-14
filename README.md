# RAG Chatbot

Dieses Projekt implementiert einen voll funktionsfähigen Chatbot basierend auf einem Retrieval-Augmented Generation (RAG) Modell. Der Chatbot unterstützt OCR zur Texterkennung, die Integration interner Daten (Schreibgutobjekte) über FAISS und eine finale Prüfschleife für verwaltungsrechtlich einwandfreie Texte.

## Installation

1. Repository klonen:
   ```bash
   git clone https://github.com/Moenogram/RAG.git
   cd RAG
   ```

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. Anwendung starten:
   ```bash
   python app.py
   ```

## Projektstruktur

- **app.py**: Flask-Anwendung als Frontend.
- **ocr.py**: Integration von Tesseract OCR.
- **index.py**: Erstellung und Verwaltung des FAISS-Index.
- **rag_model.py**: Generierung der Antworten mittels RAG-Modell.
- **templates/index.html**: HTML-Interface.
