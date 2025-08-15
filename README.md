# 🤖 Chatbot

This project is an **AI-powered chatbot** that can read and understand documents, extract structured information, and answer questions interactively.  
It is built with **Google Gemini**, **LangChain**, and **Streamlit**, and integrates with **Confluence** for document ingestion.

---

## 🚀 Features
- 📂 **Document Ingestion** – Supports DOCX and other formats.
- 📝 **Smart Formatting** – Reformats documents if needed, while keeping structure if already clean.
- 📊 **Table Preservation** – Extracts and maintains table structure.
- 🖼️ **OCR for Images** – Reads text and diagrams inside images.
- 📐 **Diagram Conversion** – Converts UML / sequence diagrams into ASCII format for engineers.
- 🔗 **Confluence Integration** – Automatically ingests documents hosted on Confluence.
- ⚡ **Optimized Performance** – Uses caching to reduce redundant processing and speed up answers.
- 🎨 **Streamlit UI** – Simple, interactive, and user-friendly interface.

---

## 🛠️ Tech Stack
- **Python 3**
- **Streamlit**
- **LangChain**
- **Google Gemini API**
- **Confluence API**

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/HassanMuhamed1/Chatbot.git
cd Chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage
1. Run the chatbot:
   ```bash
   streamlit run chatbot_app.py
   ```
2. Upload a document (DOCX, PDF, etc.).
3. Ask your questions in the chat interface.
4. Get structured, accurate responses instantly.

---

## 📌 Example Use Cases
- 📑 Quickly summarize large technical documents.
- 🛠️ Convert diagrams into ASCII-based explanations.
- 🔍 Extract tables and structured data for analysis.
- 🏢 Enterprise knowledge assistant with Confluence integration.

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo, submit issues, or create pull requests.

---

## 👤 Author
**Hassan Muhammed**  
- 🎓 Computer Science & AI student at Cairo University  
- 💻 Passionate about **AI, NLP, and Machine Learning**  
- 🔗 [LinkedIn](https://www.linkedin.com/in/hassan-muhammed-1947a12a4/)  
- 📧 hassanmuhammedd14@gmail.com
