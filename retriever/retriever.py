import argparse
import requests
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

# === Función para generar el embedding de la consulta ===
def get_query_embedding(query: str, api_url: str, api_key: str, model_name: str) -> list:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "input": [query],
        "input_type": "query"
    }

    response = requests.post(
        f"{api_url}/v1/embeddings",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# === Main ===
parser = argparse.ArgumentParser(description="Retrieval semántico desde Qdrant con múltiples queries")
parser.add_argument("--input_file", help="Ruta al archivo JSON con las preguntas (array de strings)", default=os.getenv("INPUT_FILE"))
parser.add_argument("--api_url", help="URL del modelo de embeddings", default=os.getenv("API_URL"))
parser.add_argument("--api_key", help="API KEY del modelo de embeddings", default=os.getenv("API_KEY"))
parser.add_argument("--model_name", help="Nombre del modelo de embedding", default=os.getenv("MODEL_NAME"))
parser.add_argument("--host", help="Host de la base de datos", default=os.getenv("HOST"))
parser.add_argument("--port", type=int, help="Puerto Qdrant", default=os.getenv("PORT"))
parser.add_argument("--collection", help="Nombre de la colección en Qdrant", default=os.getenv("COLLECTION"))
parser.add_argument("--top_k", type=int, help="Número de documentos relevantes a recuperar", default=os.getenv("TOP_K"))
parser.add_argument("--output_dir", help="Directorio donde guardar el archivo JSON de resultados", default=os.getenv("OUTPUT_DIR"))
parser.add_argument("--output_file", help="Nombre del archivo de salida JSON", default=os.getenv("OUTPUT_FILE"))

args = parser.parse_args()

# Leer queries desde archivo
with open(args.input_file, "r", encoding="utf-8") as f:
    queries = json.load(f)

# Conexión a Qdrant
client = QdrantClient(host=args.host, port=args.port)
search_params = SearchParams(hnsw_ef=128, exact=False)

# Resultados acumulados
all_results = []

for query in queries:
    print(f"\nProcesando query: \"{query}\"")
    try:
        embedding = get_query_embedding(query, args.api_url, args.api_key, args.model_name)

        results = client.search(
            collection_name=args.collection,
            query_vector=embedding,
            limit=args.top_k,
            search_params=search_params,
            with_payload=True
        )

        query_result = {
            "query": query,
            "results": []
        }

        for i, result in enumerate(results):
            print(f"  #{i+1} - Score: {result.score:.4f}")
            print(f"     Fuente: {result.payload.get('source')}")
            print(f"     Chunk: {result.payload.get('chunk')}")

            query_result["results"].append({
                "score": result.score,
                "source": result.payload.get("source"),
                "chunk": result.payload.get("chunk"),
                "text": result.payload.get("text")
            })

        all_results.append(query_result)

    except Exception as e:
        print(f"Error en la query: \"{query}\": {e}")

# Guardar resultados
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, args.output_file)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n Todos los resultados guardados en: {output_path}")
