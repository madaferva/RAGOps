
# Chunking

Etapa de chunking basada en directorios con PDFs y generando su contenido en ficheros de texto con el metadato como primera línea.

## Argumentos:

* pdf_folder: Directorio con los PDFs a añadir como contexto en el RAG
* output_path: Directorio donde se dejarán los ficheros en formato TXT de cada chunk


### Ejemplo de ejecución:

$ python3 chunking.py --pdf_folder=~/datasets/OCP/ --output_path=~/datasets/OCP_chunks/


## Containerización

### Creación del contenedor

docker build -t chunking:0.9 .

### Ejecución del contenedor

$ docker run --rm -v ~/datasets/OCP:/app/OCP -v ~/datasets/OCP_chunks/:/app/OCP_chunks chunking:0.9 --pdf_folder /app/OCP/ --output_path /app/OCP_chunks/

