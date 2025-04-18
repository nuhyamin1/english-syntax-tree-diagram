# Syntax Tree Diagram App

This is a Python-based application for visualizing syntax trees, offering both a web interface (using Flask) and a standalone desktop application (using PyQt6). It is designed to help users parse and display syntactic structures, typically for natural language processing, linguistics or educational purposes.

## Features
- Parse input text and generate both constituency and dependency syntax trees.
- Visual representation of tree structures (using D3.js for constituency, displaCy for dependency).
- Explanations (legends) for tags used in the parses.
- User-friendly interface for both web and desktop versions.

## Requirements
- Python 3.x
- Dependencies listed in `requirements.txt` (includes Flask, spaCy, benepar, PyQt6, PyQt6-WebEngine, etc.)
- Required spaCy and benepar models (will be downloaded automatically if not found):
  - `en_core_web_sm`
  - `benepar_en3`

## Getting Started
1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd Syntax_Tree_Diagram
   ```
2. **Set up the virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
    ```
4. **Run the Application**

   *   **Web Version (Flask):**
       ```bash
       python app.py
       ```
       Then, open your browser and go to `http://127.0.0.1:5000/`

   *   **Desktop Version (PyQt6):**
       ```bash
       python gui.py
       ```
       This will launch the standalone desktop application window.

## Project Structure
```
Syntax_Tree_Diagram
├── .gitignore
├── LICENSE
├── README.md
├── app.py             # Flask web application
├── gui.py             # PyQt6 desktop application
├── requirements.txt
├── templates
│   └── index.html     # HTML template for Flask app
```

## License
This project is open-source and available under the MIT License.

---
*Last updated: 2025-04-18*
