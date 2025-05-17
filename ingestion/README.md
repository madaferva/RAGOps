
# Chunking

Etapa de chunking basada en directorios con PDFs y generando su contenido en ficheros de texto con el metadato como primera línea.

## Argumentos:

* collection_name: Nombre de la colección a crear en la base de datos vectorial
* json_path: Path completo (incluido el archivo) del json con los chunks y los vectores previos
* host_port: Puerto de la base de datos vectorial
* host: Host de la base de datos vectorial


### Ejemplo de ejecución:

$ python3 ingestion.py  --json_path="~/datasets/OCP_emb/ocp_embeddings.json" --collection_name "test" --host_port=6333 --host=localhost

## Containerización

### Creación del contenedor

docker build -t ingestion .

### Ejecución del contenedor

$ docker run --rm -v /home/davidfv/datasets/OCP_emb/:/app/OCP_emb ingestion --json_path="/app/OCP_emb/ocp_embeddings.json" --collection_name "test" --host_port=6333 --host=qdrant

