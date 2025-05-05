from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser

# === Configuración de paths ===
input_dir = Path("/home/davidfv/datasets/OCP/")
output_dir = Path("/home/davidfv/datasets/OCP_chunks/")
output_dir.mkdir(parents=True, exist_ok=True)

# === Configura el parser con chunking optimizado para RAG ===
parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)

# === Cargar PDFs usando el loader de LlamaIndex ===
# Esto ya detecta PDFs y extrae texto automáticamente

print(input_dir)

#documents = SimpleDirectoryReader(str(input_dir)).load_data()

documents = SimpleDirectoryReader(input_dir="/home/davidfv/datasets/OCP", recursive = False ).load_data()

print("llego")

print(len(documents))

# === Procesar cada documento ===
for doc in documents:
    print(doc)
    base_name = Path(doc.metadata.get("file_name", "unknown")).stem
    print(f"Procesando: {base_name}")

    nodes = parser.get_nodes_from_documents([doc])

    for i, node in enumerate(nodes):
        chunk_text = node.text
        #print(node.text)
        metadata = f"[SOURCE: {base_name}.pdf | CHUNK: {i+1}/{len(nodes)}]\n\n"
        chunk_file = output_dir / f"{base_name}_chunk_{i+1}.txt"

        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(metadata + chunk_text)

print("✅ Chunking con metadatos completado con SimpleDirectoryReader.")
