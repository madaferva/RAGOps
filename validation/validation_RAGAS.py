import json
import argparse
import requests
import os
import pandas as pd
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


def load_input_as_dataset(json_path: str) -> Dataset:
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    print(f"Cargados {len(data)} ejemplo(s)")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", help="Fichero JSON con query + contextos + respuesta", default=os.getenv("ANSWER_FILE"))
    parser.add_argument("--output_validation", help="Fichero salida con métricas", default=os.getenv("OUTPUT_VALIDATION"))
    parser.add_argument("--embeddings_api_url", help="URL del modelo de embeddings", default=os.getenv("EMBEDDINGS_API_URL"))
    parser.add_argument("--embeddings_api_key", help="API KEY del modelo de embeddings", default=os.getenv("EMBEDDINGS_API_KEY"))
    parser.add_argument("--embeddings_model_name", help="Nombre del modelo de embedding", default=os.getenv("EMBEDDINGS_MODEL_NAME"))
    parser.add_argument("--llm_api_url", help="URL del modelo de embeddings", default=os.getenv("LLM_API_URL"))
    parser.add_argument("--llm_api_key", help="API KEY del modelo de embeddings", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--llm_model_name", help="Nombre del modelo de embedding", default=os.getenv("LLM_MODEL_NAME"))
    args = parser.parse_args()

    # EMBEDDING_URL = "http://10.10.78.12:8001/v1/embeddings"
    # EMBEDDING_MODEL = "nvidia/nv-embedqa-mistral-7b-v2"

    # LLM_URL = "http://10.10.78.11:8079/v1"
    # LLM_MODEL = "google/gemma-3-27b-it"

    embeddings_api_url = args.embeddings_api_url
    embeddings_model_name = args.embeddings_model_name
    embeddings_api_key= args. embeddings_api_key

    llm_api_url = args.llm_api_url
    llm_model_name = args.llm_model_name
    llm_api_key = args.llm_api_key

    answer_file = args.answer_file
    output_validation = args.output_validation

    # Instancias
    embedding_fn = MyRemoteEmbeddings(embeddings_api_url, embeddings_model_name)
    llm_fn = ChatOpenAI(base_url=llm_api_url, api_key=llm_api_key   , model=llm_model_name, temperature=0.0)

    # embedding_fn = MyRemoteEmbeddings(EMBEDDING_URL, EMBEDDING_MODEL)
    # llm_fn = ChatOpenAI(base_url=llm_api_url, api_key="none", model=llm_model_name, temperature=0.0)

    dataset = load_input_as_dataset(answer_file)

    has_ref = all(x.get("reference") for x in dataset)
    metrics = [faithfulness, answer_relevancy]
    if has_ref:
        metrics += [context_precision, context_recall]

    print(f"Ejecutando evaluación con {len(dataset)} ejemplo(s)...")
    results = evaluate(dataset, metrics=metrics, llm=llm_fn, embeddings=embedding_fn)

    df = results.to_pandas()
    results_list = df.to_dict(orient="records")

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available_metrics = [m for m in metric_names if m in df.columns]
    global_metrics = df[available_metrics].mean(numeric_only=True).to_dict()

    results_list.append({"global_metrics": global_metrics})

    with open(output_validation, "w") as f:
        json.dump(results_list, f, indent=2)

    print(f"\n Evaluación completada. Resultados guardados en: {args.output_validation}")
    print("\n Métricas globales:")
    for k, v in global_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

