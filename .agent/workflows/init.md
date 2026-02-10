---
description: Initialize the project by setting up a virtual environment and installing dependencies.
---

This workflow initializes the ILIN project by creating a Python virtual environment, installing necessary dependencies, and downloading the required embedding model.

// turbo-all
1. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

2. Upgrade pip and install dependencies:
   ```powershell
   .\venv\Scripts\python.exe -m pip install --upgrade pip
   .\venv\Scripts\pip.exe install -r requirements.txt
   ```

3. Download the embedding model (first run only):
   ```powershell
   .\venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```
