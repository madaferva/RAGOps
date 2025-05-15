import requests
import os
import time
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field
from typing import List

def clean_text(text: str) -> str:
    lines = text.strip().splitlines()
    clean_lines = []

    for line in lines:
        if line.strip() == "":
            continue
        # Filtra líneas tipo consola o que parecen tabla
        if line.strip().startswith("NAME ") or line.strip().startswith("pod/"):
             continue
        if "AGE" in line and "READY" in line:
             continue
        clean_lines.append(line.strip())

    return " ".join(clean_lines)

class TEIEmbeddingModel(BaseEmbedding):
    api_url: str = Field()
    api_key: str = Field()
    model_name: str = Field(default="nvidia/nv-embedqa-mistral-7b-v2")

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text, input_type="passage")

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query, input_type="query")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_embedding(self, text: str, input_type: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "input": [text],
            "input_type": input_type
        }

        response = requests.post(
            f"{self.api_url}/v1/embeddings",
            headers=headers,
            json=payload
        )

        #print("Respuesta cruda del servidor:", response.text)  # DEBUG

        response.raise_for_status()
        response_json = response.json()

        # Extraer vector desde respuesta del servidor
        embedding_vector = response_json["data"][0]["embedding"]
        return embedding_vector

# Crear instancia
embedding_model = TEIEmbeddingModel(
    api_url="http://10.10.78.12:8001",
    api_key="TU_API_KEY_DE_NVIDIA_NIM"
)

# Probar con un texto
texto = "Este es un texto de prueba."
vector = embedding_model._get_text_embedding(texto)

print(f"Embedding (longitud {len(vector)}):")
#print(vector)

# Ruta del directorio
chunk_dir = os.path.expanduser("/home/davidfv/datasets/OCP_chunks")
output = []
i=0
for filename in os.listdir(chunk_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(chunk_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            continue  # Saltar si el archivo está vacío

        # Extraer metadatos de la primera línea
        header = lines[0].strip()
        content = "".join(lines[1:]).strip()

        content = clean_text(content)

        if header.startswith("[SOURCE:") and "| CHUNK:" in header:
            try:
                source = header.split("[SOURCE:")[1].split("|")[0].strip()
                chunk = header.split("CHUNK:")[1].split("]")[0].strip()
                # print(source)
                # print(chunk)
            except Exception as e:
                print(f"Error al parsear header en {filename}: {e}")
                continue
        else:
            print(f"Formato inválido en {filename}, se omite.")
            continue

        # Obtener embedding
        try:
            # print(content)
            embedding = embedding_model._get_text_embedding(content)
            i=i+1
            lenemb = len(embedding)
            print(f'{i} Longitud: {lenemb}')
            time.sleep(1)
            #print(embedding)
            print(content)
        except Exception as e:
            print(f"Error al obtener embedding de {filename}: {e}")
            continue

        output.append({
            "source": source,
            "chunk": chunk,
            "text": content,
            "embedding": embedding
        })

# Guardar en JSON
with open("ocp_embeddings.json", "w", encoding="utf-8") as out_file:
    json.dump(output, out_file, ensure_ascii=False, indent=2)

print(f"Procesados {len(output)} chunks y guardados en 'ocp_embeddings.json'")
print(i)