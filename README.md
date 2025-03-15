## Cyber Assistant v4.0

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)  
[![Flask](https://img.shields.io/badge/Flask-v2.0+-blue)](https://palletsprojects.com/p/flask/)  
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)  
[![GitHub Issues](https://img.shields.io/github/issues/g0dux/CHATCYBER-BETA-)](https://github.com/g0dux/CHATCYBER-BETA-/issues)  
[![GitHub Stars](https://img.shields.io/github/stars/g0dux/CHATCYBER-BETA-)](https://github.com/g0dux/CHATCYBER-BETA-/stargazers)

> **Cyber Assistant v4.0** is an interactive web application that combines an intelligent chat assistant with advanced forensic investigation and image metadata extraction capabilities. Powered by the robust Mistral-7B-Instruct neural model, it integrates various libraries for natural language processing, image analysis, and forensic data extraction.

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

## Features

- **Intelligent Chat:**  
  Converse with the assistant in multiple languages. Choose between **Technical** (detailed) or **Freeform** (creative) responses.

- **Forensic Investigation:**  
  Perform online investigations using target queries to retrieve data from various websites. Configure options such as the number of sites, investigation focus, and inclusion of news or leaked data.

- **Image Metadata Extraction:**  
  Analyze EXIF metadata from an image URL—including automatic GPS coordinate conversion that links directly to Google Maps.

- **Neural Model Integration:**  
  Powered by the Mistral-7B-Instruct model (via `llama_cpp`), the assistant generates context-aware responses and supports real-time language translation if necessary.

- **Local Caching:**  
  An in-memory caching system reduces redundant processing for repeated queries.

- **Customizable Interfaces:**  
  Both Flask and Gradio interfaces are highly configurable, allowing you to adjust appearance, response speed, and detail level.

- **Performance Monitoring:**  
  Integrated with Prometheus to track request counts and response latencies.

---

## Project Architecture

Cyber Assistant v4.0 is built on a modular architecture:

- **Backend (Flask):**  
  Manages HTTP requests, integrates with the neural model, and handles forensic analysis and metadata extraction.

- **Neural Model:**  
  Loaded via `llama_cpp`; automatically downloads from the Hugging Face Hub if not found locally.

- **Front-end:**  
  - **Flask Interface:** A classic web interface with HTML, CSS, and JavaScript.  
  - **Gradio Interface:** A modern, interactive UI powered by Gradio for real-time feedback and ease of use.

- **Supporting Libraries:**  
  Leverages libraries like `nltk`, `langdetect`, `duckduckgo_search`, and `Pillow` for text processing, language detection, web search, and image analysis.

---

## Installation and Setup

### Prerequisites

- **Python 3.8 or higher**
- **Git** (to clone the repository)
- **Virtualenv** (recommended for environment isolation)

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

   On first run, the project automatically downloads necessary NLTK data (e.g., *punkt* and *vader_lexicon*).

---

## How to Run

Start the application by running:

```bash
python app.py
```

The Flask server will be available at [http://0.0.0.0:5000/](http://0.0.0.0:5000/). Open your browser, navigate to the URL, and begin interacting with Cyber Assistant v4.0.

---

## Operation Modes

### Chat Mode

- **Description:**  
  Engage in a natural, conversational interaction with the AI assistant.

- **Configuration:**  
  - Select language and response style.
  - Adjust the AI’s temperature and response speed (fast vs. detailed).

---

### Investigation Mode

- **Description:**  
  Conduct online investigations based on a target query.

- **Configuration:**  
  - Set the number of websites to search.
  - Specify an investigation focus (e.g., phishing, malware).
  - Enable options to include news or leaked data.
  - Choose between “Detailed” (more comprehensive, slower) or “Fast” (quicker, fewer details).

- **Output:**  
  Generates a report with forensic analysis, links, and actionable insights.

---

### Metadata Mode

- **Description:**  
  Extract and display image metadata from a provided URL.

- **Features:**  
  - Retrieve complete EXIF data.
  - Automatically convert GPS coordinates into a clickable Google Maps link.

---

## Flask Interface

The Flask interface is a classic server-rendered web interface. Key highlights include:

- **Endpoints:**  
  Multiple endpoints handle chat, investigation, metadata extraction, email analysis, and more.

- **Performance Monitoring:**  
  Prometheus integration tracks request counts and latencies.

- **Customization:**  
  Uses HTML, CSS, and JavaScript for responsive design, plus a config modal to adjust appearance.

---

## Gradio Interface

The Gradio interface offers a modern, interactive UI:

- **Real-Time Interaction:**  
  Users receive instant feedback as they chat or investigate.

- **Intuitive Controls:**  
  Sliders and radio buttons allow quick adjustments of parameters like temperature and speed.

- **Seamless Integration:**  
  Mirrors Flask functionalities, ensuring consistency between both interfaces.

---

## Advanced Forensic Analysis

This module extracts critical forensic data from investigation results, such as:

- **IP Addresses:** IPv4 and IPv6  
- **Contact Information:** Emails and phone numbers  
- **URLs, MAC Addresses, and Hashes:** MD5, SHA1, SHA256  
- **Vulnerability IDs:** e.g., CVE identifiers

These insights aid in identifying patterns and collecting digital evidence for cybersecurity investigations.

---

## Running on Google Colab

Cyber Assistant v4.0 can also run on Google Colab. Follow these steps:

1. **Open a New Notebook** in Google Colab.
2. **Clone the Repository & Install Dependencies:**

   ```python
   %cd /content
   !git clone https://github.com/g0dux/CHATCYBER-BETA-.git
   %cd /content/CHATCYBER-BETA-
   !pip install -r requirements.txt
   !python app.py
   ```

3. **Access the Application:**

   - Gradio will provide a URL.
   - Or use Flask endpoints at the provided address.

---

## Contribution

Contributions are welcome! To contribute:

1. **Fork** the repository.
2. **Create a New Branch:**

   ```bash
   git checkout -b my-new-feature
   ```
3. **Commit Your Changes:**

   ```bash
   git commit -m "Add new feature"
   ```
4. **Push Your Branch:**

   ```bash
   git push origin my-new-feature
   ```
5. **Open a Pull Request** describing your changes in detail.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for details.

---

## Acknowledgments

- Thanks to the developers and maintainers of Flask, nltk, llama_cpp, and other libraries that power this project.
- Special thanks to the open-source community for ongoing support and contributions.

---

## Screenshots

Below are areas for adding screenshots. Make sure to copy your images into an `images` folder in the repository and update the paths accordingly.

---
### Flask Interface

![Flask Interface](https://github.com/g0dux/CHATCYBER-BETA-/blob/main/screenshot/Captura%20de%20tela%20chat%20flask.png)

---
### Gradio Interface

![Gradio Interface](https://github.com/g0dux/CHATCYBER-BETA-/blob/main/screenshot/Captura%20de%20tela%20de%20chat%20gradio.png)

---

**Enjoy using Cyber Assistant v4.0** — your interactive tool for intelligent chat, forensic investigation, and image metadata extraction. Happy exploring and investigating!
