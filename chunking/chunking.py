import os
from pathlib import Path
import fitz

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document

# === Configuracion ===
input_dir = Path("../../../datasets/OCP/")
output_dir = Path("../../../datasets/OCP_chunks/")
output_dir.mkdir(parents=True, exist_ok=True)

# Parser configurado para RAG
parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)

# === Funciones ===

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_pdf_and_save(pdf_path):
    base_name = pdf_path.stem
    print(f"Procesando: {base_name}")

    raw_text = extract_text_from_pdf(pdf_path)
    document = Document(text=raw_text)

    nodes = parser.get_nodes_from_documents([document])

    for i, node in enumerate(nodes):
        chunk_text = node.text
        metadata = f"[SOURCE: {base_name}.pdf | CHUNK: {i+1}/{len(nodes)}]\n\n"

        chunk_file = output_dir / f"{base_name}_chunk_{i+1}.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(metadata + chunk_text)

# === Proceso principal ===

for pdf_file in input_dir.glob("*.pdf"):
    chunk_pdf_and_save(pdf_file)

print("Chunking con metadatos completado.")

