import torch
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import json

class DocumentProcessor:
    def __init__(self, model_path="german-bert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embedder = SentenceTransformer(model_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def process_document(self, file_path):
        """OCR und Textextraktion aus Dokumenten"""
        if file_path.endswith('.pdf'):
            images = convert_from_path(file_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image, lang='deu')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return text

class RAGSystem:
    def __init__(self, api_key):
        self.openai_api_key = api_key
        self.doc_processor = DocumentProcessor()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.vector_store = None
        self.qa_chain = None
        
    def train(self, document_dir):
        """Training des RAG Systems mit Verwaltungsdokumenten"""
        # Dokumente laden und verarbeiten
        loader = DirectoryLoader(document_dir)
        documents = loader.load()
        
        # Texte in Chunks aufteilen
        texts = self.doc_processor.text_splitter.split_documents(documents)
        
        # Vektorstore erstellen
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./data/chroma_db"
        )
        
        # QA Chain initialisieren
        llm = OpenAI(api_key=self.openai_api_key)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )
        
    def generate_response(self, input_document, template_type="standard"):
        """Generiert Verwaltungsantwort basierend auf Input"""
        # Dokument verarbeiten
        input_text = self.doc_processor.process_document(input_document)
        
        # Rechtliche Prüfung und Antwortgenerierung
        prompt = f"""
        Analysiere folgendes Dokument und erstelle eine rechtssichere Verwaltungsantwort.
        Beachte dabei:
        - Verwaltungsrechtliche Grundlagen
        - Formelle Anforderungen
        - Rechtsmittelbelehrung wenn nötig
        
        Dokument:
        {input_text}
        """
        
        response = self.qa_chain({"query": prompt})
        return response["result"]

class QualityCheck:
    def __init__(self):
        """Initialisiert Prüfmodule für Verwaltungstexte"""
        self.legal_requirements = self._load_legal_requirements()
        
    def _load_legal_requirements(self):
        """Lädt rechtliche Anforderungen aus Konfigurationsdatei"""
        with open('legal_requirements.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def check_document(self, text):
        """Prüft generierte Dokumente auf rechtliche Konformität"""
        issues = []
        
        # Formale Prüfung
        if not self._check_formal_requirements(text):
            issues.append("Formale Anforderungen nicht erfüllt")
            
        # Rechtliche Prüfung
        legal_issues = self._check_legal_compliance(text)
        if legal_issues:
            issues.extend(legal_issues)
            
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "text": text
        }
    
    def _check_formal_requirements(self, text):
        """Prüft formale Anforderungen"""
        # Implementierung der formalen Prüfung
        return True
    
    def _check_legal_compliance(self, text):
        """Prüft rechtliche Konformität"""
        issues = []
        # Implementierung der rechtlichen Prüfung
        return issues

# Beispiel Verwendung:
if __name__ == "__main__":
    # System initialisieren
    rag_system = RAGSystem(api_key="your-openai-api-key")
    
    # System mit vorhandenen Dokumenten trainieren
    rag_system.train("path/to/document/directory")
    
    # Qualitätsprüfung initialisieren
    quality_checker = QualityCheck()
    
    # Dokument verarbeiten und Antwort generieren
    response = rag_system.generate_response("path/to/input/document.pdf")
    
    # Qualitätsprüfung durchführen
    check_result = quality_checker.check_document(response)
    
    if check_result["passed"]:
        print("Dokument kann versendet werden")
    else:
        print("Folgende Probleme wurden gefunden:", check_result["issues"])