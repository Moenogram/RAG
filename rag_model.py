from transformers import RagTokenizer, RagTokenForGeneration

# Laden des Tokenizers und Modells
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

def generate_answer(input_text):
    """
    Generiert eine Antwort basierend auf der kombinierten Eingabe (Frage + Kontext).
    """
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]
