
# Retriever

Etapa donde se pasa una pregunta y se devuelve un fichero JSON con las entradas de la base de datos más similares

## Argumentos:

* query: Pregunta realizada por el usuario
* collection_name: Nombre de la colección a crear en la base de datos vectorial
* host_port: Puerto de la base de datos vectorial
* host: Host de la base de datos vectorial
* top_k: Numero de entradas más relevantes devueltas


### Ejemplo de ejecución:

$ python3 retriever.py   --query "How can I access a OCP?"   --api_url http://xx.xx.xx.xx:8001   --api_key tu_api_key   --model_name nvidia/nv-embedqa-mistral-7b-v2   --host yy.yy.yy.yy   --port 6333   --collection ocp_collection   --top_k 5   --output_dir ~/datasets/OCP_retr   --output_file OCP_retr.json


## Containerización

### Creación del contenedor

docker build -t retriever .

### Ejecución del contenedor

$ docker run --rm -v ~/datasets/OCP_retr/:/app/OCP_retr retriever --query "How can create a MachineSet?"   --api_url http://xx.xx.xx.xx:8001   --api_key tu_api_key   --model_name nvidia/nv-embedqa-mistral-7b-v2   --host yy.yy.yy.yy   --port 6333   --collection ocp_collection   --top_k 5   --output_dir /app/OCP_retr   --output_file OCP_retr.json

