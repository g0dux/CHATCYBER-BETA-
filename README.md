```markdown
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
- [Flask Interface](#flask-interface)
- [Gradio Interface](#gradio-interface)
- [Advanced Forensic Analysis](#advanced-forensic-analysis)
- [Running on Google Colab](#running-on-google-colab)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Screenshots](#screenshots)

---

## Features ✨

- **Intelligent Chat:**  
  Engage in conversations with the assistant in multiple languages. Choose between **Technical** (detailed) or **Freeform** (creative) responses.

- **Forensic Investigation:**  
  Perform online investigations using target queries to retrieve data from various websites. Configure options such as the number of sites, investigation focus, and inclusion of news or leaked data.

- **Image Metadata Extraction:**  
  Analyze EXIF metadata from an image URL—including automatic GPS coordinate conversion that links directly to Google Maps.

- **Neural Model Integration:**  
  Powered by the Mistral-7B-Instruct model (via `llama_cpp`), the assistant generates context-aware responses and supports real-time language translation if necessary.

- **Local Caching:**  
  An in-memory caching system minimizes redundant processing for repeated queries.

- **Customizable Interfaces:**  
  Both the Flask and Gradio interfaces are highly configurable with options to adjust appearance, response speed, and detail level.

- **Performance Monitoring:**  
  Integrated with Prometheus to track request counts and response latencies.

---

## Project Architecture 🏗️

Cyber Assistant v4.0 is built using a modular architecture:

- **Backend (Flask):**  
  Handles HTTP requests, integrates with the neural model, and manages forensic analysis and metadata extraction.

- **Neural Model:**  
  Loaded using the `llama_cpp` library; automatically downloads the model from the Hugging Face Hub if it is not found locally.

- **Front-end:**  
  Two distinct interfaces are provided:
  - **Flask Interface:** A classic web interface built with HTML, CSS, and JavaScript.
  - **Gradio Interface:** A modern, interactive UI powered by Gradio that offers a user-friendly experience with real-time feedback.

- **Supporting Libraries:**  
  Leverages libraries such as `nltk`, `langdetect`, `duckduckgo_search`, and `Pillow` for text processing, language detection, web search, and image analysis.

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

   On the first run, the project will automatically download necessary NLTK data (e.g., *punkt* and *vader_lexicon*).

---

## How to Run ▶️

Start the application by running:

```bash
python app.py
```

The Flask server will start at [http://0.0.0.0:5000/](http://0.0.0.0:5000/). Open your browser, navigate to the URL, and interact with Cyber Assistant v4.0.

---

## Operation Modes 🎮

Cyber Assistant v4.0 offers three primary modes:

### Chat Mode 💬

- **Description:**  
  Engage in natural, interactive conversation with the AI assistant.

- **Configuration:**  
  - Choose your preferred language and response style.
  - Adjust the AI's temperature and response speed to balance creativity and performance.

---

### Investigation Mode 🔍

- **Description:**  
  Conduct online investigations based on target queries.

- **Configuration:**  
  - Define the number of websites to search.
  - Specify a focus for the investigation (e.g., phishing, malware).
  - Enable options to include news or leaked data.
  - Choose between "Detailed" (more comprehensive but slower) and "Fast" (quicker with fewer details) response modes.

- **Output:**  
  Generates a report with forensic analysis, relevant links, and actionable insights.

---

### Metadata Mode 🖼️

- **Description:**  
  Extract and display image metadata from a provided URL.

- **Features:**  
  - Retrieve complete EXIF data.
  - Automatically convert GPS coordinates to a clickable Google Maps link.

---

## Flask Interface

The Flask interface is the classic web interface powered by Flask. It handles HTTP requests and renders pages using server-side templates. Key aspects include:

- **Endpoints:**  
  Multiple endpoints handle chat, investigation, metadata extraction, email analysis, and more.
  
- **Performance Monitoring:**  
  Integrated with Prometheus for tracking metrics.
  
- **Customization:**  
  Uses HTML, CSS, and JavaScript for a responsive design with a configuration modal to adjust appearance settings.

---

## Gradio Interface

The Gradio interface provides an interactive, modern user interface that runs alongside Flask. Key features include:

- **Real-Time Interaction:**  
  Users receive immediate feedback as they interact with the assistant.
  
- **Intuitive Controls:**  
  Sliders and radio buttons allow users to configure parameters such as temperature, response speed, and more.
  
- **Seamless Integration:**  
  The Gradio interface mirrors the functionalities available in the Flask interface, providing a user-friendly experience for both chat and investigation modes.

---

## Advanced Forensic Analysis 🕵️‍♂️

The advanced forensic analysis module extracts critical forensic data from the results, including:

- **IP Addresses:** IPv4 and IPv6  
- **Contact Information:** Emails and phone numbers  
- **URLs, MAC Addresses, and Hashes:** MD5, SHA1, and SHA256  
- **Vulnerability IDs:** Such as CVE identifiers

This module aids in identifying patterns and gathering digital evidence for cybersecurity investigations.

---

## Running on Google Colab ☁️

Cyber Assistant v4.0 is designed to run seamlessly on Google Colab. Follow these steps to launch the project in Colab:

1. **Open a New Notebook in Google Colab.**

2. **Clone the Repository and Install Dependencies:**

   Paste and run the following cell:

   ```python
   %cd /content
   !git clone https://github.com/g0dux/CHATCYBER-BETA-.git
   %cd /content/CHATCYBER-BETA-
   !pip install -r requirements.txt
   !python app.py
   ```

3. **Access the Application:**

   The application will start automatically. Use the URL provided by Gradio or access the Flask endpoints as required.

---

## Contribution 🤝

Contributions are highly welcome! To contribute:

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

---

## License 📄

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the LICENSE file for details.

---

## Acknowledgments 🙏

- Thanks to the developers and maintainers of Flask, nltk, llama_cpp, and all other libraries that make this project possible.
- Special thanks to the open-source community for continuous support and contributions.

---

## Screenshots 📸

Below are areas reserved for screenshots of the application interfaces. Replace the image URLs with your own screenshots.

### Flask Interface

![Flask Interface](https://via.placeholder.com/800x450?text=Flask+Interface+Screenshot)

### Gradio Interface

![Gradio Interface](https://via.placeholder.com/800x450?text=Gradio+Interface+Screenshot)

---

Enjoy using **Cyber Assistant v4.0** – your interactive tool for intelligent chat, forensic investigation, and image metadata extraction. Happy exploring and investigating!
```

This README is written in a detailed and interactive manner, providing clear sections for the project's features, architecture, installation, operation modes, both interfaces, and even dedicated areas for screenshots. Simply update the placeholder image URLs with actual screenshots of your Flask and Gradio interfaces.
