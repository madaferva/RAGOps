import json
import argparse
import requests
from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings


# === Embeddings personalizados usando tu API ===
class MyRemoteEmbeddings(Embeddings):
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.endpoint,
            json={"input": texts, "model": self.model, "input_type": "passage"}
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            self.endpoint,
            json={"input": [text], "model": self.model, "input_type": "query"}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


# === Carga JSON a Dataset ===
def load_input_as_dataset(json_path: str) -> Dataset:
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    print(f"📂 Cargados {len(data)} ejemplo(s)")

    dataset = Dataset.from_list([
        {
            "question": item["query"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "reference": item.get("reference") or item.get("ground_truth"),
        }
        for item in data
    ])
    return dataset


# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True, help="Fichero JSON con query + contextos + respuesta")
    parser.add_argument("--output_validation", required=True, help="Fichero salida con métricas")
    args = parser.parse_args()

    # Configuración de tus endpoints locales
    EMBEDDING_URL = "http://10.10.78.12:8001/v1/embeddings"
    EMBEDDING_MODEL = "nvidia/nv-embedqa-mistral-7b-v2"

    LLM_URL = "http://10.10.78.11:8079/v1"
    LLM_MODEL = "google/gemma-3-27b-it"

    # Instancias
    embedding_fn = MyRemoteEmbeddings(EMBEDDING_URL, EMBEDDING_MODEL)
    llm_fn = ChatOpenAI(base_url=LLM_URL, api_key="none", model=LLM_MODEL, temperature=0.0)

    # Dataset
    dataset = load_input_as_dataset(args.answer)

    # Elegir métricas según si hay "reference"
    has_ref = all(x.get("reference") for x in dataset)
    metrics = [faithfulness, answer_relevancy]
    if has_ref:
        metrics += [context_precision, context_recall]

    print(f"🚀 Ejecutando evaluación con {len(dataset)} ejemplo(s)...")
    results = evaluate(dataset, metrics=metrics, llm=llm_fn, embeddings=embedding_fn)

    results_dict = results.to_pandas().to_dict(orient="records")
    with open(args.output_validation, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✅ Evaluación completada. Resultados en: {args.output_validation}")


if __name__ == "__main__":
    main()



