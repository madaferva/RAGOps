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
parser = argparse.ArgumentParser(description="Consulta semántica y almacenamiento de resultados en JSON")
parser.add_argument("--query", help="Texto de la pregunta que se quiere recuperar", default=os.getenv("QUERY"))
parser.add_argument("--api_url", help="URL del modelo de embeddings", default=os.getenv("API_URL"))
parser.add_argument("--api_key", help="API KEY del modelo de embeddings", default=os.getenv("API_KEY"))
parser.add_argument("--model_name", help="Nombre del modelo de embedding", default=os.getenv("MODEL_NAME"))
parser.add_argument("--host", help="Host de la base de datos", default=os.getenv("HOST"))
parser.add_argument("--port", type=int, help="Puerto Qdrant", default=os.getenv("PORT"))
parser.add_argument("--collection", help="Nombre de la colección en la base de datos", default=os.getenv("COLLECTION"))
parser.add_argument("--top_k", type=int, help="Número de documentos relevantes a recuperar", default=os.getenv("TOP_K"))
parser.add_argument("--output_dir", help="Directorio donde guardar el archivo JSON de resultados", default=os.getenv("OUTPUT_DIR"))
parser.add_argument("--output_file", help="Nombre del archivo JSON (ej. resultados.json)", default=os.getenv("OUTPUT_FILE"))

args = parser.parse_args()

# Obtener embedding de la query
embedding = get_query_embedding(args.query, args.api_url, args.api_key, args.model_name)

# Conexión a Qdrant
client = QdrantClient(host=args.host, port=args.port)
search_params = SearchParams(hnsw_ef=128, exact=False)

# Realizar búsqueda
results = client.search(
    collection_name=args.collection,
    query_vector=embedding,
    limit=args.top_k,
    search_params=search_params,
    with_payload=True
)

# Estructura de salida
output_data = {
    "query": args.query,
    "results": []
}

print(f"\nResultados para: \"{args.query}\"\n")

for i, result in enumerate(results):
    print(f"#{i+1} - Score: {result.score:.4f}")
    print(f"Fuente: {result.payload.get('source')}")
    print(f"Chunk: {result.payload.get('chunk')}")
    print(f"Texto: {result.payload.get('text')}\n")

    output_data["results"].append({
        "score": result.score,
        "source": result.payload.get("source"),
        "chunk": result.payload.get("chunk"),
        "text": result.payload.get("text")
    })

# Asegurar que el directorio existe
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, args.output_file)

# Guardar resultados en JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"Resultados guardados en: {output_path}")
