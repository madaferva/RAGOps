import argparse
import json
import openai
import os

# Carga el JSON con múltiples queries
def load_retrieved_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

# Deduplica y selecciona top_k entradas
def deduplicate_and_rank(results, top_k=3):
    sorted_items = sorted(results, key=lambda x: x['score'], reverse=True)
    top_chunks, seen = [], set()
    for item in sorted_items:
        text = item.get("text")
        if text and text not in seen:
            top_chunks.append(text)
            seen.add(text)
        if len(top_chunks) >= top_k:
            break
    return top_chunks

# Construye el prompt a partir del contexto y la pregunta
def build_prompt(context_chunks, question):
    context = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(context_chunks)])
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

# Consulta al modelo Gemma vía API OpenAI-compatible
def query_model(prompt, key, url, model, temperature, max_tokens):
    client = openai.OpenAI(
        api_key=key,
        base_url=url
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# Proceso principal para múltiples queries
def main(json_path, output_path, top_k, model_name, api_url, api_key, temperature, max_tokens):
    dataset = load_retrieved_json(json_path)
    all_results = []

    for i, item in enumerate(dataset):
        question = item["query"]
        results = item["results"]

        top_chunks = deduplicate_and_rank(results, top_k=top_k)
        prompt = build_prompt(top_chunks, question)
        answer = query_model(prompt, api_key, api_url, model_name, temperature, max_tokens)

        print(f"\n [{i+1}] Pregunta: {question}")
        print(f"Respuesta: {answer[:100]}...")  # Muestra un fragmento por brevedad

        all_results.append({
            "query": question,
            "prompt": prompt,
            "answer": answer,
            "contexts": top_chunks
        })

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nTodas las respuestas se han guardado en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", help="Ruta al fichero JSON con múltiples queries y chunks", default=os.getenv("JSON_PATH"))
    parser.add_argument("--output_answer", help="Ruta de salida para guardar respuestas y prompts", default=os.getenv("OUTPUT_ANSWER"))
    parser.add_argument("--top_k", type=int, help="Número máximo de entradas de contexto para el prompt", default=os.getenv("TOP_K"))
    parser.add_argument("--model_name", help="Modelo de LLM", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--api_url", help="URL AP LLM", default=os.getenv("API_URL"))
    parser.add_argument("--api_key", help="API KEY", default=os.getenv("API_KEY"))
    parser.add_argument("--temperature", help="Temperatura del LLM", default=os.getenv("TEMPERATURE"))
    parser.add_argument("--max_tokens", help="Tokens máximos de respuesta", default=os.getenv("MAX_TOKENS"))
    args = parser.parse_args()

    main(args.json_path, args.output_answer, args.top_k, args.model_name, args.api_url, args.api_key, args.temperature, args.max_tokens)
