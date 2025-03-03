# Cyber Assistant v4.0

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)  
[![Flask](https://img.shields.io/badge/Flask-v2.0+-blue)](https://palletsprojects.com/p/flask/)  
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)  
[![GitHub Issues](https://img.shields.io/github/issues/g0dux/CHATCYBER-BETA-)](https://github.com/g0dux/CHATCYBER-BETA-/issues)  
[![GitHub Stars](https://img.shields.io/github/stars/g0dux/CHATCYBER-BETA-)](https://github.com/g0dux/CHATCYBER-BETA-/stargazers)

> **Cyber Assistant v4.0** is an interactive web application that blends an intelligent chat assistant with advanced forensic investigation and image metadata extraction capabilities. Leveraging the powerful Mistral-7B-Instruct neural model, it integrates various libraries for text processing, image analysis, and forensic data extraction.

---

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

---

## Features ✨

- **Intelligent Chat:**  
  Engage in conversations with the assistant in multiple languages, with options for **Technical** or **Freeform** (creative) responses.

- **Forensic Investigation:**  
  Perform online investigations using target queries to retrieve data from multiple websites. Enjoy detailed forensic analysis with actionable insights.

- **Image Metadata Extraction:**  
  Analyze EXIF metadata from an image URL, including automatic GPS coordinate conversion that integrates with Google Maps.

- **Neural Model Integration:**  
  Powered by the Mistral-7B-Instruct model (via `llama_cpp`), the assistant generates context-aware responses and performs real-time language translations when needed.

- **Local Caching:**  
  Utilizes an in-memory caching system for faster response times on repeated queries, optimized for Windows environments.

- **Interactive Web Interface:**  
  A responsive, modern front end built with HTML, CSS, and JavaScript ensures a smooth user experience.

- **Performance Monitoring:**  
  Integrated with Prometheus, the tool tracks request counts and response latencies for robust performance analysis.

---

## Project Architecture 🏗️

Cyber Assistant v4.0 is designed with a modular architecture:

- **Backend (Flask):**  
  Manages user requests, communicates with the neural model, and handles forensic analysis and image metadata extraction.

- **Neural Model:**  
  Loaded using the `llama_cpp` library; if not available locally, it’s automatically downloaded from the Hugging Face Hub.

- **Front-end:**  
  An interactive web interface allowing mode selection and configuration, enabling dynamic user interactions.

- **Supporting Libraries:**  
  Utilizes powerful libraries like `nltk`, `langdetect`, `duckduckgo_search`, and `Pillow` to support language processing, image analysis, and data extraction.

---

## Installation and Setup 🔧

### Prerequisites

- **Python 3.8 or higher**  
- **Git** – [Clone the repository](https://github.com/g0dux/CHATCYBER-BETA-.git)
- **Virtualenv** (Recommended for environment isolation)

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/g0dux/CHATCYBER-BETA-.git
   cd cyber-assistant-v4
   ```

2. **Create a Virtual Environment (Recommended):**

   - **Linux/MacOS:**
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   _Main dependencies include:_  
   **Flask, nltk, langdetect, cachetools, requests, psutil, llama_cpp, huggingface_hub, duckduckgo_search, Pillow**

4. **Initial NLTK Setup:**

   On first run, the project downloads required NLTK data (e.g., *punkt* and *vader_lexicon*).

---

## How to Run ▶️

Start the application by running:

```bash
python app.py
```

The Flask server will be available at [http://0.0.0.0:5000/](http://0.0.0.0:5000/). Open your web browser, navigate to the URL, and start interacting with Cyber Assistant v4.0.

---

## Operation Modes 🎮

The application supports three main modes, all configurable via the intuitive web interface:

### Chat Mode 💬

- **Description:**  
  Engage in a conversational exchange with the assistant.

- **Features:**  
  - Select your preferred language (Português, English, Español, Français, Deutsch).
  - Choose between **Technical** or **Freeform** response styles.

---

### Investigation Mode 🔍

- **Description:**  
  Conduct online investigations based on a target query.

- **Features:**  
  - Specify the number of websites to search.
  - Define a specific investigation focus.
  - Enable options to include news or leaked data in the search.

- **Output:**  
  Generates a detailed report complete with extracted links, forensic analysis, and additional findings.

---

### Metadata Mode 🖼️

- **Description:**  
  Extract and display EXIF metadata from an image provided via URL.

- **Features:**  
  - Retrieves comprehensive image metadata.
  - Converts GPS coordinates to a clickable Google Maps link when available.

---

## Advanced Forensic Analysis 🕵️‍♂️

The advanced forensic analysis feature extracts and correlates critical data useful for digital investigations, including:

- **IP Addresses:** IPv4 and IPv6  
- **Contact Information:** Emails and Phone Numbers  
- **URLs & MAC Addresses**  
- **Hashes:** MD5, SHA1, and SHA256  
- **Vulnerability IDs:** Such as CVE (Common Vulnerabilities and Exposures)

These capabilities help identify patterns and collect evidence, making Cyber Assistant an effective tool for cybersecurity and digital forensic investigations.

---

## Contribution 🤝

Contributions are highly welcome! Follow these steps to contribute:

1. **Fork the Repository**
2. **Create a New Branch:**
   ```bash
   git checkout -b my-new-feature
   ```
3. **Make Your Changes and Commit:**
   ```bash
   git commit -m "Add new feature"
   ```
4. **Push Your Branch:**
   ```bash
   git push origin my-new-feature
   ```
5. **Open a Pull Request** on GitHub detailing your changes.

Your contributions help make Cyber Assistant even better!

---

## License 📄

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute this project under the terms of the license.

---

Enjoy exploring the features of **Cyber Assistant v4.0** and happy investigating! 🚀

Feel free to open issues and suggest improvements. Your feedback is always appreciated!
