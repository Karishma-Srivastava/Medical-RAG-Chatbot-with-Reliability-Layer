from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def check_hallucination(query, answer, context):
    prompt = f"""
You are an evaluator.

Given:
Question: {query}
Answer: {answer}
Context: {context}

Is the answer fully supported by the context?

Reply ONLY with:
- "YES" if grounded
- "NO" if hallucinated
"""

    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.choices[0].message.content.strip().upper()

    if "YES" in result:
        return True
    else:
        return False