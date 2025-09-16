# Insights with Fida | AI-Powered Portfolio Assistant 🤖

[![HF Space](https://img.shields.io/badge/Try%20Demo-HuggingFace-blue.svg)](https://huggingface.co/spaces/chfida/ai-portfolio-assistant)

Insights with Fida is an AI-powered personal portfolio assistant that transforms a static resume into an interactive experience.
It allows recruiters, collaborators, and clients to ask questions about my career, skills, and projects — and get natural, first-person answers.

---

## 🚀 Live Demo

Run the model directly in your browser using Hugging Face Spaces:

🔗 **Demo Link**: [AI Portfolio Assistant](https://huggingface.co/spaces/chfida/ai-portfolio-assistant)

📦 **Embed iFrame** (for blogs or dashboards):

```html
<iframe
  src="https://chfida-ai-portfolio-assistant.hf.space/"
  frameborder="0"
  width="100%"
  height="500">
</iframe>
```

## 🛠️ Features
- 💬 **Conversational Resume** – Ask about my career, projects, and skills.
- 📚 **RAG Pipeline** – Combines semantic search (FAISS + embeddings) with GPT reasoning.
- 🧠 **Context Retention** – Handles multi-turn conversations seamlessly.
- 🎨 **Interactive UI** – Built with Streamlit for smooth chat-like experience.
- 🌐 **Publicly Accessible** – No installation needed, runs on Hugging Face Spaces.

🧠 Technical Stack

- **Vector Store**: FAISS
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM**: OpenAI GPT-4o-mini (`ChatOpenAI`)
- **Frameworks**: LangChain, Streamlit
- **Deployment**: Hugging Face Spaces

⚙️ How It Works
1. My documents (resume, portfolio, project notes) are converted into embeddings using Sentence-Transformers.
2. A FAISS index stores and retrieves the most relevant chunks based on user queries.
3. GPT-4o-mini takes the retrieved context and generates conversational answers as if I’m responding directly.
4. The assistant remembers prior exchanges for a natural, ongoing dialogue.

## 📁 Repository Contents

```bash
📦 ai-portfolio-assistant
├── src/                        # Core app source code
│   ├── streamlit_app.py        # Streamlit UI + logic
│   ├── streamlit_functions.py  # Helper functions
├── README.md                   # Project documentation

```
> **Note:**: Prompts and FAISS index are not included in this repository due to sensitive and personal data. The app fetches them securely from a private Hugging Face model repository.

## ✍️ Author
Fida Muhammad <br>
🔗 [Hugging Face](https://huggingface.co/chfida)<br>
🔗 [Hugging Face Space](https://huggingface.co/chfida/spaces)<br>
🔗 [Linkedin](https://www.linkedin.com/in/fida-m/)  
🔗 [Website](https://www.fidamuhammad.com)


## 📜 License
This project is licensed under the MIT License.
You are free to use, modify, and share the code for non-commercial and research purposes.

🙏 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [OpenAI](https://platform.openai.com/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Streamlit](https://streamlit.io/)

