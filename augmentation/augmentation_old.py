import argparse
import json
import openai
import os

# Carga el json de la etapa de retrieval con pregunta y elementos recuperados 

def load_retrieved_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


# Deduplica las entradas recuperadas en retrieval para no tener entradas recuperadas y selecciona de ellas top_k numeros

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

# Construye el prompt en base a la pregunta del usuario y a los items recuperados.

def build_prompt(context_chunks, question):
    context = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(context_chunks)])
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

def query_model(prompt, key, url, model, temperature, max_tokens):
    client = openai.OpenAI(
        api_key= key, 
        base_url= url
    )

    response = client.chat.completions.create(
        model= model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature= temperature,
        max_tokens= max_tokens
    )
    return response.choices[0].message.content

def main(json_path, output_path, top_k):
    data = load_retrieved_json(json_path)
    question = data["query"]
    top_chunks = deduplicate_and_rank(data["results"], top_k=top_k)
    prompt = build_prompt(top_chunks, question)
    answer = query_model(prompt, "none", "http://10.10.78.11:8079/v1", "google/gemma-3-27b-it", 0.1, 256)

    print("\nPrompt enviado al LLM:\n" + prompt)
    print("\nRespuesta del LLM:\n" + answer)

    result = {
        "query": question,
        "prompt": prompt,
        "answer": answer,
        "contexts": top_chunks
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Resultado guardado en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", help="Ruta al fichero JSON con los chunks recuperados")
    parser.add_argument("--output_answer", help="Ruta de salida para guardar respuesta y prompt")
    parser.add_argument("--top_k", help="Numero máximo de entradas de contexto para el prompt")
    args = parser.parse_args()

    main(args.json_path, args.output_answer,int(args.top_k))


