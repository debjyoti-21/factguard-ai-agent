# FACTGUARD – Multi-Agent Fake News Detection with Gemini API

import os
import numpy as np
import pandas as pd
print("✅ FactGuard started successfully")



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import google.generativeai as genai # type: ignore

api_key = input("Enter your Gemini API key (will NOT be saved): ").strip()

if not api_key:
    raise ValueError("API key is required.")

genai.configure(api_key=api_key)


# use a fast model; you can change to gemini-1.5-pro if you want
llm = genai.GenerativeModel("gemini-1.5-flash")


# =========================================================
# 1. LOAD DATASET  (Fake & Real news CSVs)
# =========================================================

true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true["label"] = 1  # REAL
fake["label"] = 0  # FAKE

df = pd.concat([true, fake]).reset_index(drop=True)

# combine title + text
df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

X = df["full_text"]
y = df["label"]


# =========================================================
# 2. TRAIN CLASSICAL ML MODEL (CUSTOM TOOL)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("\n✅ MODEL TRAINED")
print(f"Accuracy: {acc:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))


# =========================================================
# 3. KEY CONCEPT: SESSION & MEMORY + LOGGING
# =========================================================

session_state = {
    "history": [],   # list of {input, label, confidence}
    "count": 0
}

interaction_logs = []  # for observability


# =========================================================
# 4. AGENT 1 – CLASSIFIER TOOL (Custom Tool)
# =========================================================

def fake_news_classifier_agent(text: str) -> dict:
    """
    Custom tool agent:
    Uses TF-IDF + Logistic Regression to classify text as FAKE / REAL.
    """
    if not text or not text.strip():
        return {"error": "Empty text", "label": None, "confidence": None}

    vec = tfidf.transform([text])
    probs = clf.predict_proba(vec)[0]
    idx = int(np.argmax(probs))
    label_num = idx
    label_name = "REAL" if label_num == 1 else "FAKE"
    confidence = float(probs[idx]) * 100.0

    return {
        "label": label_name,
        "label_num": label_num,
        "confidence": round(confidence, 2)
    }


# =========================================================
# 5. AGENT 2 – (Optional) RESEARCH AGENT (stub tool)
# =========================================================

def research_agent_stub(text: str):
    """
    Stub research agent.
    In a full system this would call Google Search / fact-check APIs.
    Here we just return placeholders to demonstrate tool usage.
    """
    return [
        "Example source: BBC / Reuters / official websites.",
        "For real deployment, integrate Google Search or fact-check APIs here."
    ]


# =========================================================
# 6. AGENT 3 – EXPLANATION AGENT (Gemini LLM via API)
# =========================================================

def explanation_agent_llm(user_text: str, cls_result: dict, evidence: list, history: list) -> str:
    """
    LLM-powered agent using Gemini API.
    Takes:
      - original user text
      - classifier result (FAKE / REAL + confidence)
      - evidence list (stubbed for now)
      - session history (for context)
    Returns a natural-language explanation.
    """

    if cls_result.get("error"):
        return f"Could not analyze the text: {cls_result['error']}"

    verdict = cls_result["label"]
    confidence = cls_result["confidence"]

    # Shorten history for context
    recent_history = history[-3:] if history else []

    prompt = f"""
You are FactGuard, an AI assistant that helps users evaluate whether news looks fake or real.

User submitted this news text:
\"\"\"{user_text}\"\"\"


The classifier model output:
- Prediction: {verdict}
- Confidence: {confidence:.2f}%

Some example generic evidence notes:
{chr(10).join('- ' + e for e in evidence)}

Recent previous queries this user asked (for context):
{recent_history}

Your job:
- Explain in simple English why this news might look trustworthy or suspicious.
- Mention the model's confidence but remind the user this is not a final truth.
- Encourage the user to check trusted sources (official websites, major news outlets).
- Keep the answer under 200-250 words.

Now write your explanation as FactGuard:
"""

    response = llm.generate_content(prompt)
    return response.text.strip()


# =========================================================
# 7. ORCHESTRATOR AGENT – SEQUENTIAL MULTI-AGENT PIPELINE
# =========================================================

def factguard_orchestrator(user_text: str) -> str:
    """
    Multi-agent, SEQUENTIAL pipeline:
      1) classifier_agent (ML tool)
      2) research_agent_stub (tool)
      3) explanation_agent_llm (Gemini API)
    Also updates session memory + logs interaction.
    """

    # 1. Classifier agent
    cls_result = fake_news_classifier_agent(user_text)

    # 2. Research agent (stub)
    evidence = research_agent_stub(user_text)

    # 3. Explanation agent (LLM with API)
    explanation = explanation_agent_llm(
        user_text=user_text,
        cls_result=cls_result,
        evidence=evidence,
        history=session_state["history"]
    )

    # 4. Update session memory
    session_state["history"].append({
        "input": user_text,
        "label": cls_result.get("label"),
        "confidence": cls_result.get("confidence")
    })
    session_state["count"] += 1

    # 5. Log for observability
    interaction_logs.append({
        "input": user_text,
        "label": cls_result.get("label"),
        "confidence": cls_result.get("confidence")
    })

    return explanation


# =========================================================
# 8. SIMPLE CLI INTERFACE
# =========================================================

def print_menu():
    print("\n==============================")
    print(" FACTGUARD – FAKE NEWS AGENT ")
    print("==============================")
    print("Type any news text to analyze.")
    print("Commands:")
    print("  /memory  - show session history")
    print("  /logs    - show basic logs (count only)")
    print("  /exit    - quit")
    print("==============================\n")


if __name__ == "__main__":
    print_menu()

    while True:
        user = input("👉 Enter news text (or command): ").strip()

        if not user:
            continue

        if user.lower() == "/exit":
            print("👋 Goodbye!")
            break

        if user.lower() == "/memory":
            print("\n📚 SESSION MEMORY:")
            if not session_state["history"]:
                print("  (empty)")
            else:
                for i, item in enumerate(session_state["history"], start=1):
                    print(f"  {i}. {item['input'][:60]}... -> {item['label']} ({item['confidence']}%)")
            continue

        if user.lower() == "/logs":
            print("\n📊 LOG SUMMARY:")
            print("  Total interactions:", len(interaction_logs))
            continue

        # Normal path: run the multi-agent pipeline
        print("\n🤖 FACTGUARD ANALYSIS:")
        try:
            result = factguard_orchestrator(user)
            print(result)
        except Exception as e:
            print("Error during analysis:", e)
