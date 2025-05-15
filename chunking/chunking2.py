from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
import argparse

# Configuración de argparse
parser = argparse.ArgumentParser(description="Extración de parrafos de un fichero PDF a distintos ficheros de texto")
parser.add_argument("--pdf_folder", help="Directorio con PDFs")
parser.add_argument("--output_path", help="Directorio de salida de los ficheros de texto")

args = parser.parse_args()

# Opcional: Parámetros adicionales, como configuración de idioma o modo de extracción
# parser.add_argument("--encoding", default="utf-8", help="Codificación del archivo de salida (predeterminado: utf-8)")

pdf_folder = args.pdf_folder
output_path = args.output_path



# Configuración de paths

input_dir = Path(pdf_folder)
output_dir = Path(output_path)
output_dir.mkdir(parents=True, exist_ok=True)
chunk_size = 512
chunk_overlap = 50

# === Configura el parser con chunking optimizado para RAG ===
parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# === Cargar PDFs usando el loader de LlamaIndex ===
# Esto ya detecta PDFs y extrae texto automáticamente

# print(input_dir)

documents = SimpleDirectoryReader(str(input_dir)).load_data()


# === Procesar cada documento ===

# Numero global de chunk 

i=0

print ('Documents' + str(len(documents)))
      
# Trato de cada documento: En el caso el tratamiento de PDF de LLamaindex asume 
# que el documento es cada pagina del PDF

for doc in documents:

    base_name = Path(doc.metadata.get("file_name", "unknown")).stem

    # Nodos de cada pagina del PDF

    nodes = parser.get_nodes_from_documents([doc])

    # Escritura en fichero para la siguiente etapa de cada chunk

    for node in nodes:   
      i=i+1
      print(f'{base_name} - {i}')
      chunk_text = node.text

      # Añadimos el metadata en la primera línea del fichero para poder
      # recuperarlo posteriormente en 

      metadata = f"[SOURCE: {base_name}.pdf | CHUNK: {i}]\n\n"
      chunk_file = output_dir / f"{base_name}_chunk_{i}.txt"

      with open(chunk_file, "w", encoding="utf-8") as f:
        f.write(metadata + chunk_text)



print(f'Chunking con metadatos completado con SimpleDirectoryReader. Chunk Size: {chunk_size}. Chunk Overlap: {chunk_overlap} Chunks generated: {i}\n')

