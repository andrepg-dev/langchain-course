from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def fake_llm(prompt: str) -> str:  # Fake LLM for the example
    return "completion"


dictionary = {
    "llm1": fake_llm,
    "llm2": fake_llm,
}

getterp = itemgetter("llm1")

runnable = dictionary | RunnablePassthrough.assign(hey=lambda x: x)

print(runnable.invoke(""))
# {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}
