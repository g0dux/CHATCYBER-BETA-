```markdown
# Cyber Assistant v4.0

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-v2.0+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Cyber Assistant v4.0 is an interactive web application that combines an intelligent chat assistant with advanced forensic investigation capabilities and image metadata extraction. This project utilizes a neural model (Mistral-7B-Instruct) to generate responses in multiple languages and modes (Chat, Investigation, and Metadata), integrating various libraries for text processing, image analysis, and forensic data extraction.

## Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Operation Modes](#operation-modes)
  - [Chat Mode](#chat-mode)
  - [Investigation Mode](#investigation-mode)
  - [Metadata Mode](#metadata-mode)
- [Advanced Forensic Analysis](#advanced-forensic-analysis)
- [Contribution](#contribution)
- [License](#license)

## Features

- **Intelligent Chat:** Interact with the assistant in various languages and styles (Technical or Freeform).
- **Forensic Investigation:** Conduct online investigations based on a target query, retrieving data from multiple websites and performing detailed forensic extraction.
- **Image Metadata Extraction:** Analyze the EXIF metadata of an image provided via URL, including GPS data conversion for Google Maps integration.
- **Neural Model Integration:** Uses the Mistral-7B-Instruct model via `llama_cpp` to generate responses and perform language translation when needed.
- **Query Caching:** Implements caching to improve performance on repeated queries.
- **Interactive Web Interface:** A responsive, configurable front-end built with HTML, CSS, and JavaScript.

## Project Architecture

The project consists of:

- **Backend (Flask):** Handles user requests, generates responses using the neural model, and executes forensic analysis and image metadata extraction functions.
- **Neural Model:** Loaded using the `llama_cpp` library and downloaded from the Hugging Face Hub if not available locally.
- **Front-end:** An interactive web interface that allows users to interact with the assistant in different operational modes.
- **Supporting Libraries:** Includes `nltk`, `langdetect`, `cachetools`, `duckduckgo_search`, `Pillow`, among others, for language processing, image analysis, and data extraction.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- [Git]((https://github.com/g0dux/CHATCYBER-BETA-.git))
- Virtualenv (recommended)

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/cyber-assistant-v4.git
   cd cyber-assistant-v4
   ```

2. **Create a Virtual Environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/MacOS
   venv\Scripts\activate      # Windows
   ```

3. **Install the Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - Flask
   - nltk
   - langdetect
   - cachetools
   - requests
   - psutil
   - llama_cpp
   - huggingface_hub
   - duckduckgo_search
   - Pillow

4. **Initial NLTK Setup:**

   On first run, the project will download the necessary NLTK data (such as `punkt` and `vader_lexicon`).

## How to Run

After installing the dependencies, start the application with:

```bash
python app.py
```

The Flask server will start at `http://0.0.0.0:5000/`. Open your web browser and navigate to this URL to use Cyber Assistant v4.0.

## Operation Modes

The application supports three operation modes, configurable via the web interface:

### Chat Mode

- **Description:** Interact with the assistant in a conversational chat format.
- **Features:** Choose the language (Português, English, Español, Français, Deutsch) and response style (Technical or Freeform).

### Investigation Mode

- **Description:** Conduct online investigations based on a provided target query, collecting data from multiple websites.
- **Features:** Adjust the number of websites to search, set a specific focus, and enable options for retrieving news or leaked data.
- **Output:** Displays a detailed report with extracted information and forensic analysis.

### Metadata Mode

- **Description:** Extract and display the metadata of an image using its URL.
- **Features:** Includes EXIF data extraction and GPS coordinate conversion for a Google Maps link, if available.

## Advanced Forensic Analysis

The advanced forensic analysis function has been enhanced to extract a wide range of relevant information from text, including:

- **IPv4 and IPv6 Addresses**
- **Emails and Phone Numbers**
- **URLs and MAC Addresses**
- **Hashes:** MD5, SHA1, and SHA256
- **Vulnerability IDs:** CVE (Common Vulnerabilities and Exposures)

This functionality helps in identifying and correlating critical data useful for digital forensic investigations and security analysis.

## Contribution

Contributions are welcome! If you wish to help improve Cyber Assistant, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or fix: `git checkout -b my-new-feature`
3. Make your changes and commit them: `git commit -m 'Add new feature'`
4. Push your branch: `git push origin my-new-feature`
5. Open a Pull Request on GitHub.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to open issues and suggest improvements. Enjoy exploring the various features of Cyber Assistant v4.0!
