# 🎬 Subtitle Retriever Box
![GitHub last commit](https://img.shields.io/github/last-commit/owner/repo)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-success)
![Gemini](https://img.shields.io/badge/Gemini-Google-4285F4)
![License](https://img.shields.io/github/license/owner/repo)

---

An intelligent subtitle search & context generator that:
- Transcribes spoken audio to text with `OpenAI Whisper`
- Searches across thousands of movie subtitles for relevant content
- Generates contextual insights via `Google Gemini Pro`
- Delivers an interactive web UI with `Streamlit`

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/owner/repo.git
cd subtitle-retriever-box
```

### 2. Create & Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

**requirements.txt**
```text
streamlit==1.29.0
torch==2.1.0
torchaudio==2.1.0
openai-whisper==20231117
langchain==0.1.0
langchain-google-genai==0.0.7
langchain-huggingface==0.0.3
sentence-transformers==2.2.2
chromadb==0.4.15
pandas==2.0.3
sqlite3
```

### 4. Configure API Keys
Create `gemini.txt` in the repo root:
```text
<your-google-gemini-api-key>
```

---

## 📁 Folder Structure
```
subtitle-retriever-box/
├── decoding_zip_files.py  # Exports subtitles DB → .srt files
├── model (1).py          # Main Streamlit app
├── dataset/              # Generated .srt files (run decoding first)
├── data/                 # Place additional subtitles here
├── eng_subtitles_database.db
├── gemini.txt
├── venv/
└── README.md
```

---

## 🔧 Usage

1. **Generate the subtitle dataset** (once):
   ```bash
   python decoding_zip_files.py
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run "model (1).py"
   ```

3. **Interact with the UI**:
   - Upload any `.mp3`, `.wav`, or `.m4a` audio file.
   - View transcription, matched subtitles, and LLM-generated context.

---

## 🧪 How It Works

| Steps | Technology |
|-------|------------|
| Audio → Text | OpenAI Whisper (`base`) |
| Subtitle Storage | Chroma vector DB |
| Vector Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Context Generation | Google Gemini Pro |
| Web Interface | Streamlit |

---

## 🔒 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | *(Future)* For Whisper cloud |
| `GOOGLE_API_KEY` | Injected from `gemini.txt` |

---

## 🤝 Contribution Guide

### Issues & Feature Requests
Start with an issue so we can discuss before you write code.

### Pull Request Flow
```bash
git checkout -b feature/your-feature
# Implement changes
git add .
git commit -m "feat: short descriptive message"
git push origin feature/your-feature
```
Open a Pull Request on GitHub.

### Coding Standards
- Python code follows `black --line-length 88`
- Type hints encouraged
- Docstrings for every function
- Unit tests for core logic (`pytest`)

### Local Testing
```bash
pytest tests/
streamlit run "model (1).py"
```

---

## 📜 License
MIT © Subtitle Retriever Box Contributors
