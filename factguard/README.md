🛡️ FactGuard – AI-Powered Multi-Agent System for Fake News Detection

FactGuard is an intelligent multi-agent AI system that helps users analyze whether a news article is likely fake or real using a combination of machine learning and a large language model (LLM). The project is designed as an agent-based pipeline where each agent performs a specialized task, orchestrated by a central controller.

🎯 Project Goal

Misinformation spreads rapidly through social media and online platforms, making it difficult for users to identify what is trustworthy. FactGuard aims to:

Reduce misinformation exposure
downlode the data hear-
https://drive.google.com/drive/folders/14UsbkORA2Wyplt29SVG--LBDshDNvWOE?usp=drive_link
Provide explainable AI decisions

Assist learners and everyday users in evaluating online content

Demonstrate how agent-based AI systems solve real-world problems

🧠 System Architecture (Multi-Agent)

FactGuard uses a Sequential Multi-Agent System approach:

Agents in the System

Classifier Agent (Custom ML Tool)

Uses TF-IDF + Logistic Regression to classify news

Outputs FAKE or REAL with confidence score

Research Agent (Tool Agent)

Simulates evidence gathering from trusted sources

In production, this could be replaced with real search APIs

Explanation Agent (LLM Agent – Gemini API)

Uses Gemini to explain results in natural language

Generates a user-friendly report

Orchestrator Agent

Controls execution flow

Calls agents in sequence

Stores memory & logs activity

✅ Key Concepts Implemented (Capstone Requirements)
✔ Multi-Agent System

Sequential agents cooperating to solve a complex task.

✔ Agent Powered by an LLM

Gemini API generates explanations in real-time.

✔ Tools

Custom ML classifier

External API (Gemini)

Simulated research agent

✔ Sessions & Memory

Conversation history is stored per session and accessible using /memory.

✔ Observability / Logging

Every prediction is logged internally and can be reviewed with /logs.

✔ Agent Evaluation (Optional Bonus)

Aggregate statistics available through /eval (if implemented).

⚙️ Installation & Setup
1. Clone Repository
git clone https://github.com/debjyoti-21/factguard-ai-agent.git
cd factguard-ai-agent

2. Install Dependencies
pip install -r requirements.txt

3. Download Dataset

Download Fake and Real News Dataset by Clément Bisaillon from Kaggle.

Place these files into the project directory:

True.csv
Fake.csv

4. Run Project
python app.py

5. Enter API Key at Runtime

For security, the Gemini API key is entered during execution.

No API keys are stored in code or files.

🖥️ Commands Inside the App
Command	Function
/exit	Exit program
/memory	Show session history
/logs	Show usage stats
/eval	Summary statistics (optional)
📊 Tech Stack

Python

Scikit-learn

Pandas, NumPy

Google Gemini API

CLI-based Agent Interface

⚠️ Disclaimer

FactGuard is an AI assistant, not a certified fact-checking authority.
Always confirm critical information with trusted official sources.

🚀 Future Improvements

Real-time web search integration

Domain credibility analysis agent

Multilingual classification

Web interface

Agent deployment (Cloud Run)

👤 Author

Created by [debjyoti nath]
Capstone Project – AI Agents with Google (2025)