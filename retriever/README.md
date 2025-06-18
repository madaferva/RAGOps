
# Retriever

Etapa donde se pasa un fichero de preguntas y se devuelve un fichero JSON con las entradas de la base de datos más similares

## Argumentos:

* input_file: Pregunta realizada por el usuario
* api_url: URL del modelo de embeddings
* api_key: Key del modelo (si fuese necesario)
* model_name: Nombre del modelo a usar
* collection: Nombre de la colección a crear en la base de datos vectorial
* port: Puerto de la base de datos vectorial
* host: Host de la base de datos vectorial
* top_k: Numero de entradas más relevantes devueltas
* output_file: Fichero con cada una de las preguntas 


### Ejemplo de ejecución:

$ python3 retriever.py --input_file ~/datasets/harrypotter_retr/harry_potter_queries.json --api_url http://xx.xx.xx.xx:8001 --api_key tu_api_key --model_name nvidia/nv-embedqa-mistral-7b-v2 --host xx.xx.xx.xx --port 6333 --collection harry --top_k 10 --output_dir ~/datasets/harrypotter_retr/ --output_file harry_retr.json


## Containerización

### Creación del contenedor

docker build -t retriever .

### Ejecución del contenedor

$ docker run --rm -e INPUT_FILE="/app/harrypotter_retr/harry_potter_queries.json" -e API_URL="http://xx.xx.xx.xx:8001"   -e API_KEY="tu_api_key" -e MODEL_NAME="nvidia/nv-embedqa-mistral-7b-v2" -e HOST="192.168.1.68" -e PORT="6333" -e COLLECTION="harry"   -e TOP_K="10"   -e OUTPUT_DIR="/app/harrypotter_retr"  -e OUTPUT_FILE="harry_retr.json" -v ~/datasets/harrypotter_retr/:/app/harrypotter_retr/ retriever

