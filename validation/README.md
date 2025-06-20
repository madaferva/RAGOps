
# Validator

Etapa donde evalua el resultado de la configuración del proceso RAG basandose en preguntas y las respuestas devueltas de manera individual y global. 

## Argumentos:

* answer_file: Fichero de respuestas que ha generado la fase de aumentación
* output_validation: Fichero de resultados con las métricas por pregunta y globales
* embeddings_api_url: URL del modelo de embeddings (compatible con OpenAI API)
* embeddings_api_key: Key del modelo de embeddings
* embeddings_model_name: Nombre del modelo de embeddings a usar
* llm_api_url: URL del modelo de lenguaje (compatible con OpenAI API)
* llm_api_key: Key del modelo de lenguaje
* llm_model_name: Nombre del modelo de lenguaje

### Ejemplo de ejecución:

$ python3 validation_RAGAS.py --answer_file ~/experiments/harry_full/answers/harry_ans.json --output_validation ~/experiments/harry_full/validation/harry_full_validation.json --embeddings_api_url "http://xx.xx.xx.xx:8001/v1/embeddings" --embeddings_api_key None --embeddings_model_name nvidia/nv-embedqa-mistral-7b-v2 --llm_api_url "http://xx.xx.xx.xx:8079/v1" --llm_api_key None --llm_model_name google/gemma-3-27b-it



## Containerización

### Creación del contenedor

docker build -t validation .

### Ejecución del contenedor

docker run --rm -e ANSWER_FILE="/app/answers/harry_ans.json" -e OUTPUT_VALIDATION="/app/validation/harry_full_validation.json" -e EMBEDDINGS_API_URL="http://xx.xx.xx.xx:8001/v1/embeddings" -e EMBEDDINGS_API_KEY="None" -e EMBEDDINGS_MODEL_NAME="nvidia/nv-embedqa-mistral-7b-v2" -e LLM_API_URL="http://xx.xx.xx.xx:8079/v1" -e LLM_API_KEY="None" -e LLM_MODEL_NAME="google/gemma-3-27b-it" -v ~/experiments/harry_full/answers/:/app/answers/ -v ~/experiments/harry_full/validation/:/app/validation   validation

