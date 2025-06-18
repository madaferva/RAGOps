from ragas.evaluation import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
from langchain_openai import ChatOpenAI
import os

# LLM basado en LangChain con endpoint personalizado
llm = ChatOpenAI(
    base_url="http://10.10.78.11:8079/v1",
    api_key="none",  # No se usa pero es requerido por interfaz
    model="google/gemma-3-27b-it",
    temperature=0.0,
)

# Dataset mínimo
dataset = Dataset.from_list([{
    "question": "How old is Harry Potter?",
    "answer": "Harry is 14 years old.",
    "contexts": [
        "Harry wants to be a normal fourteenyearold wizard.",
        "Epilogue: Nineteen years later, Harry is now married and has kids."
    ]
}])

# Evaluar
print("🚀 Evaluando con faithfulness...")
results = evaluate(
    dataset,
    metrics=[faithfulness],
    llm=llm
)

# Mostrar resultado
print("\n✅ Resultados:")
print(results.to_pandas().to_string(index=False))
