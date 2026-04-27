from openai import OpenAI
import os

def generate_answer(query, context):

    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    prompt = f"""
    You are a medical assistant.

    If the question asks about management or treatment:
    - Focus ONLY on actionable steps (diet, lifestyle, control)
    - DO NOT give definitions unless necessary

    Use ONLY the context.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()
    print("PROMPT SENT TO LLM:", prompt[:500])