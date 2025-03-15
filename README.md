Below is a complete and detailed README in English, made interactive and comprehensive based on your requirements:

---

# Cyber Assistant v4.0

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)  
[![Flask](https://img.shields.io/badge/Flask-v2.0+-blue)](https://palletsprojects.com/p/flask/)  
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)  
[![GitHub Issues](https://img.shields.io/github/issues/g0dux/CHATCYBER-BETA-)](https://github.com/g0dux/CHATCYBER-BETA-/issues)  
[![GitHub Stars](https://img.shields.io/github/stars/g0dux/CHATCYBER-BETA-)](https://github.com/g0dux/CHATCYBER-BETA-/stargazers)

> **Cyber Assistant v4.0** is an interactive web application that combines an intelligent chat assistant with advanced forensic investigation and image metadata extraction capabilities. Powered by the robust Mistral-7B-Instruct neural model, it integrates numerous libraries for natural language processing, image analysis, and forensic data extraction.

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
- [Additional Features](#additional-features)
- [Running on Google Colab](#running-on-google-colab)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features ✨

- **Intelligent Chat:**  
  Engage in conversations with the assistant in multiple languages. Choose between **Technical** (detailed) or **Freeform** (creative) responses.

- **Forensic Investigation:**  
  Perform online investigations using target queries to retrieve data from various websites. Configure options such as number of sites, investigation focus, and inclusion of news or leaked data.

- **Image Metadata Extraction:**  
  Analyze EXIF metadata from an image URL—including automatic GPS coordinate conversion that links to Google Maps.

- **Neural Model Integration:**  
  Powered by the Mistral-7B-Instruct model (via `llama_cpp`), the assistant generates context-aware responses and supports real-time language translation if needed.

- **Local Caching:**  
  An in-memory caching system optimizes performance by reducing redundant processing on repeated queries.

- **Interactive Web Interface:**  
  A responsive and modern front end built with HTML, CSS, and JavaScript provides a smooth user experience with customizable configurations.

- **Performance Monitoring:**  
  Prometheus integration tracks request counts and response latencies for robust performance insights.

---

## Project Architecture 🏗️

Cyber Assistant v4.0 is built with a modular architecture:

- **Backend (Flask):**  
  Handles user requests, interacts with the neural model, and manages forensic and metadata extraction functionalities.

- **Neural Model:**  
  Loaded using the `llama_cpp` library; if not available locally, it is automatically downloaded from the Hugging Face Hub.

- **Front-end:**  
  An interactive web interface that allows users to select modes, adjust parameters (such as temperature and response speed), and customize the appearance.

- **Supporting Libraries:**  
  Utilizes libraries such as `nltk`, `langdetect`, `duckduckgo_search`, and `Pillow` for text processing, image analysis, and data extraction.

---

## Installation and Setup 🔧

### Prerequisites

- **Python 3.8 or higher**
- **Git** – To clone the repository
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

   On the first run, the project will download the required NLTK data (such as *punkt* and *vader_lexicon*).

---

## How to Run ▶️

Start the application by executing:

```bash
python app.py
```

The Flask server will start at [http://0.0.0.0:5000/](http://0.0.0.0:5000/). Open your browser, navigate to the URL, and interact with Cyber Assistant v4.0.

---

## Operation Modes 🎮

Cyber Assistant v4.0 offers three primary operation modes:

### Chat Mode 💬

- **Description:**  
  Engage in a natural, conversational interaction with the AI assistant.
- **Features:**  
  - Choose your preferred language (Português, English, Español, Français, Deutsch).
  - Select response style: **Technical** (detailed) or **Freeform** (creative).

---

### Investigation Mode 🔍

- **Description:**  
  Conduct online investigations based on a target query.
- **Features:**  
  - Define the number of websites to search.
  - Specify a focused investigation area.
  - Enable options to include news and leaked data.
  - Adjust response speed: choose between “Detailed” (more comprehensive but slower) or “Fast” (quicker with fewer details).
- **Output:**  
  Produces a detailed report with links, forensic analysis, and actionable insights.

---

### Metadata Mode 🖼️

- **Description:**  
  Extract and display image metadata from a provided URL.
- **Features:**  
  - Retrieve complete EXIF metadata.
  - Automatically convert GPS coordinates into a clickable Google Maps link (if available).

---

## Advanced Forensic Analysis 🕵️‍♂️

Cyber Assistant v4.0 includes an advanced forensic analysis module that extracts and correlates critical data for digital investigations, such as:

- **IP Addresses:** IPv4 and IPv6  
- **Contact Information:** Emails and Phone Numbers  
- **URLs & MAC Addresses**  
- **Hashes:** MD5, SHA1, SHA256  
- **Vulnerability IDs:** e.g., CVE identifiers

These capabilities aid in identifying patterns and gathering digital evidence for cybersecurity and forensic investigations.

---

## Additional Features 🚀

- **Autocorrection:**  
  The input text is automatically corrected for spelling and grammar to improve the coherence of responses.

- **Speed vs. Detail Configuration:**  
  Users can choose between a "Fast" mode (lower token count, less detailed, quicker responses) or a "Detailed" mode (higher token count, more comprehensive responses).

- **Interactive Configuration:**  
  The web interface includes a modal that lets users customize appearance settings (font sizes, colors, dimensions for chat and notes areas).

- **Performance Monitoring:**  
  Integrated with Prometheus to monitor key metrics such as request count and latency.

---

## Running on Google Colab ☁️

Cyber Assistant v4.0 is designed to work seamlessly on Google Colab. Follow these steps to run the project on Colab:

1. **Open a New Notebook on Google Colab.**

2. **Clone the Repository and Set Up the Environment:**

   Insert and execute the following cell:

   ```python
   %cd /content
   !git clone https://github.com/g0dux/CHATCYBER-BETA-.git
   %cd /content/CHATCYBER-BETA-
   !pip install -r requirements.txt
   !python app.py
   ```

3. **Access the Application:**

   The application will start automatically. You can access the interactive interface via the URL provided by Gradio, or use the Flask endpoints as needed.

---

## Contribution 🤝

Contributions are very welcome! To contribute:

1. **Fork the Repository.**
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

5. **Open a Pull Request** on GitHub with a detailed description of your changes.

Your contributions help make Cyber Assistant even better!

---

## License 📄

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the LICENSE file for more details.

---

## Acknowledgments 🙏

- Thanks to the developers and maintainers of Flask, nltk, llama_cpp, and all the other libraries that make this project possible.
- Special thanks to the open-source community for continuous support and contributions.

---

Enjoy using **Cyber Assistant v4.0** – your interactive tool for intelligent chat, forensic investigation, and image metadata extraction. Happy exploring!

---

This README is designed to be detailed and interactive, providing all the necessary information for users to understand, install, and run the project locally or on Google Colab.
