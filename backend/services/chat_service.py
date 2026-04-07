import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from groq import Groq
from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.ml.rag import retrieve, format_context

SYSTEM_PROMPT = """You are MedBot, an AI medical assistant integrated into a diagnostic system.
You speak Azerbaijani, English, and Russian fluently. Always respond in the same language the user used.

Your role:
- Greet users warmly and offer assistance at first time
- Answer questions about medical diagnosis results provided to you as context
- Explain medical terms clearly and simply using the knowledge base provided
- Help users understand what conditions like pneumonia, glioma, meningioma mean
- Suggest when to seek professional medical help

Strict rules:
- For non-medical unrelated questions (sports, politics, cooking etc.), politely decline
- Never provide specific treatment plans or prescriptions
- Always remind users that AI diagnosis is not a substitute for professional medical advice
- When answering, prefer information from the MEDICAL KNOWLEDGE BASE if provided

If user writes in Azerbaijani reply in Azerbaijani.
If user writes in Russian reply in Russian.
If user writes in English reply in English.

When user greets you, respond warmly and ask how you can help them with their health concerns.

Do not give answer related with out of medical. For example: "I break my head for cat in front of door. How can I pay compensation to cat" """

NON_MEDICAL_KEYWORDS = [
    "futbol", "basket", "idman", "musiqi", "film", "yemek", "resept",
    "siyaset", "iqtisad", "hava", "weather", "sport", "music", "movie",
    "recipe", "cooking", "политика", "спорт", "музыка", "кино",
]

GREETING_KEYWORDS = [
    "salam", "hello", "hi", "hey", "привет", "здравствуйте",
    "sabahın", "günün", "gecən", "xeyir",
]


def is_greeting(text: str) -> bool:
    text_lower = text.lower().strip()
    return any(kw in text_lower for kw in GREETING_KEYWORDS)


def is_non_medical(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in NON_MEDICAL_KEYWORDS)


def process_chat(message: str, history: list, diagnosis_context: str = "") -> dict:
    if not GROQ_API_KEY:
        return {
            "response": "GROQ_API_KEY təyin edilməyib.",
            "history":  history,
        }

    if is_non_medical(message) and not diagnosis_context:
        refuse = (
            "Üzr istəyirəm, bu mövzu mənim ixtisasım xaricindədir. "
            "Mən yalnız tibbi diaqnoz, simptomlar və sağlamlıq məsələləri ilə bağlı "
            "kömək edə bilərəm. Sağlamlığınızla bağlı sualınız varsa, məmnuniyyətlə kömək edərəm."
        )
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": refuse})
        return {"response": refuse, "history": history}

    rag_results = []
    if not is_greeting(message):
        rag_results = retrieve(message, top_k=3)

    rag_context = format_context(rag_results)

    client = Groq(api_key=GROQ_API_KEY)

    system = SYSTEM_PROMPT
    if rag_context:
        system += f"\n\n{rag_context}"
    if diagnosis_context:
        system += f"\n\nCURRENT PATIENT DIAGNOSTIC DATA:\n{diagnosis_context}"

    messages = [{"role": "system", "content": system}]
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.4,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"API xətası: {str(e)}"

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": answer})
    return {"response": answer, "history": history}