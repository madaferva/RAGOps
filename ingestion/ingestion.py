import json
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings


##### main ####

### Capturamos los distintos parámetros

# Configuración de argparse
parser = argparse.ArgumentParser(description="Conversión a embeddings de ficheros en modo texto (chunks)")
parser.add_argument("--collection_name", help="URL del modelo de embeddings")
parser.add_argument("--json_path", help="API del modelo de embeddins")
parser.add_argument("--host_port", help="Directorio donde están los ficheros de chunks en texto")
parser.add_argument("--host", help="Directorio donde se escribirá el fichero json con los embeddings")

args = parser.parse_args()


collection_name = args.collection_name
json_path = args.json_path
host_port = args.host_port
host = args.host


batch_size = 50
timeout_sec = 60.0

# Desactivar embeddings por defecto en LlamaIndex ===
Settings.embed_model = None

# Conexión a Qdrant
print("Conectando a Qdrant...")
qdrant_client = QdrantClient(host=host, port=host_port, timeout=timeout_sec)

# Cargar JSON previo
print(f"Cargando datos desde {json_path}")
with open(json_path, "r") as f:
    data = json.load(f)

vector_dim = len(data[0]["embedding"])
total_points = len(data)
print(f"📏 Dimensión del vector: {vector_dim}, total: {total_points} puntos")

# Crear colección en Qdrant si no existe
if not qdrant_client.collection_exists(collection_name):
    print(f"🧱 Creando colección '{collection_name}'...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
    )
else:
    print(f"✅ Colección '{collection_name}' ya existe.")

# Inserción por lotes para evitar timeout
print("Iniciando inserción por lotes...")
for i in range(0, total_points, batch_size):
    batch = data[i:i + batch_size]
    points = []

    for j, item in enumerate(batch):
        try:
            points.append(PointStruct(
                id=i + j,
                vector=item["embedding"],
                payload={
                    # "text": item["text"],
                    # "source": item["source"],
                    # "chunk": item["chunk"]
                    "text": item["text"]
                }
            ))
        except Exception as e:
            print(f"Error preparando punto {i+j}: {e}")

    try:
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Batch {i}-{i + len(points) - 1} insertado.")
    except Exception as e:
        print(f"Error al insertar el batch {i}-{i + len(points) - 1}: {e}")

# # === CREAR ÍNDICE PARA CONSULTAS (OPCIONAL) ===
# vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

print("Inserción completada y lista para consultas.")


