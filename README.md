### 📦 `requirements.txt`

```txt
streamlit
langchain
langchain-community
langchain-groq
```

Make sure these packages are installed in a compatible Python environment (preferably Python 3.10+).

---

### 📘 `README.md`

````markdown
# 📚 Text-to-Math Problem Solver & Knowledge Assistant

Welcome to the **Text-to-Math Problem Solver** powered by **Google Gemma2 LLM**!  
This app helps users solve math problems, conduct logical reasoning, and search for data from Wikipedia — all through a conversational chatbot built using **Streamlit** and **LangChain**.

---

## ✨ Features

- 🧮 Solve complex math problems step-by-step
- 🔍 Search accurate information using Wikipedia
- 🧠 Perform logical reasoning using language models
- 🔐 Secure API key input using Streamlit sidebar

---

## 🚀 Getting Started

### 1. Clone the repo:
```bash
git clone https://github.com/yourusername/text-to-math-assistant.git
cd text-to-math-assistant
````

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the app:

```bash
streamlit run main.py
```

### 4. Enter your 🔑 Groq API Key in the sidebar.

---

## 🛠️ Tech Stack

* `Streamlit` for the frontend UI
* `LangChain` for chaining LLM operations
* `Groq` for blazing-fast inference using Gemma2-9b-it
* `WikipediaAPIWrapper` for fetching knowledge-based data

---

## 🤝 Contributing

Pull requests are welcome! For significant changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---
