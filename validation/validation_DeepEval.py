import asyncio
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.evaluate import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from litellm import acompletion, completion, Provider
from pydantic import BaseModel
import instructor


# Modelo local que emula OpenAI API
class LocalLiteLLM(DeepEvalBaseLLM):
    def __init__(self, model="google/gemma-3-27b-it", base_url="http://10.10.78.11:8079/v1"):
        self.model = model
        self.base_url = base_url
        self.client = instructor.from_litellm(completion)
        self.async_client = instructor.from_litellm(acompletion)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        messages = [{"content": prompt, "role": "user"}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                provider=Provider.OpenAI,
                base_url=self.base_url,
                api_key="none",
                messages=messages,
                response_model=schema,
            )
        except Exception as e:
            print("[Sync Error]", e)
            response = schema.model_construct()

        return response

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        messages = [{"content": prompt, "role": "user"}]
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                provider=Provider.OpenAI,
                base_url=self.base_url,
                api_key="none",
                messages=messages,
                response_model=schema,
            )
        except Exception as e:
            print("[Async Error]", e)
            response = schema.model_construct()

        return response

    def get_model_name(self):
        return self.model


# 🚀 Main
if __name__ == "__main__":
    # Instancia modelo local
    llm = LocalLiteLLM()

    # 1 ejemplo sencillo
    test_case = LLMTestCase(
        input="¿Quién es Harry Potter?",
        actual_output="Harry Potter es un joven mago que estudia en Hogwarts.",
        context=["Harry Potter es un mago famoso por derrotar a Voldemort cuando era un bebé. Estudia en el colegio Hogwarts."],
    )

    # Métricas
    metrics = [
        FaithfulnessMetric(model=llm),
        AnswerRelevancyMetric(model=llm, strict=True)
    ]

    # Evaluar
    print("🔍 Evaluando...")
    results = asyncio.run(evaluate([test_case], metrics))

    # Mostrar resultados
    for result in results:
        print("\n📊 Resultado:")
        print(f"Faithfulness: {result.metric_results[0].score:.4f}")
        print(f"Answer Relevancy: {result.metric_results[1].score:.4f}")



