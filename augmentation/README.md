
# Augmentation

Etapa de aumentación basada el fichero JSON de pregunta y chunks recuperados donde se construye el prompt (realizando deduplicación y rankeo) y se obtiene la respuesta.

## Argumentos:

* output_answer: Fichero JSON que contiene la pregunta, la respuesta y los chunks utilizados.
* top_k: Los k chunks con más ranking (y deduplicados)
* model_name: nombre del modelo a usar
* api_url: URL del modelo de lenguaje a usar (OpenAI API)
* api_key: key (si fuese necesario)
* temperature: Temperatura del modelo
* max_tokens: Maximo de tokens permitido en la respuesta del modelo.

### Ejemplo de ejecución:

$ python3 augmentation.py --json_path ~/datasets/harrypotter_retr/harry_retr.json --output_answer ~/datasets/harrypotter_answer/harry_ans.json --top_k 5 --model_name "google/gemma-3-27b-it" --api_url "http://xx.xx.xx.xx:8079/v1" --api_key "none" --temperature 0.2 --max_tokens 256


## Containerización

### Creación del contenedor

docker build -t augmentation .

### Ejecución del contenedor

$ docker run --rm -e JSON_PATH="/app/harrypotter_retr/harry_retr.json" -e OUTPUT_ANSWER="/app/harrypotter_answer/harry_ans.json" -e TOP_K="5" -e MODEL_NAME="google/gemma-3-27b-it" -e API_URL="http://xx.xx.xx.xx:8079/v1" -e API_KEY="none" -e TEMPERATURE="0.2" -e MAX_TOKENS="256" -v ~/datasets/harrypotter_retr/:/app/harrypotter_retr/ -v ~/datasets/harrypotter_answer/:/app/harrypotter_answer/ augmentation


