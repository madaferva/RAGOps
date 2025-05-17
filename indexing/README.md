# Indexing

Etapa de obtención de vectores de embeddings obteniendo cada elemento en un fichero de texto y  generando su vector de embeddings. Con ello se contruye un JSON con los textos, los vectores y más información destinada a ser añadida a una base de datos vectorial

## Argumentos:

* api_url: URL base del motor de embeddings a usar
* api_key: Lleva del motor de embeddings a usar
* txt_folder: Directorio donde residen cada uno de los ficheros de texto de los chunks
* output_path: Directorio donde se dejará el JSON construido con los textos y sus embeddings correspondientes

### Ejemplo de ejecución:

$ python3 indexing.py --api_url="http://xx.xx.xx.xx:8001" --api_key="key" --txt_folder="~/datasets/OCP_chunks/" --output_path="~/datasets/OCP_emb/"

## Containerización

### Creación del contenedor

docker build -t indexing .

### Ejecución del contenedor

$ docker run --rm -v ~/datasets/OCP_chunks/:/app/OCP_chunks -v ~/datasets/OCP_emb:/app/OCP_emb  indexing --api_url="http://xx.xx.xx.xx:8001" --api_key="prueba" --txt_folder="/app/OCP_chunks/" --output_path="/app/OCP_emb/"