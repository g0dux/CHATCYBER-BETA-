Cyber Assistant v4.0




Cyber Assistant v4.0 is an interactive web application that combines an intelligent chat assistant with advanced forensic investigation capabilities and image metadata extraction. The project leverages the Mistral-7B-Instruct neural model and integrates multiple libraries for text processing, image analysis, and forensic data extraction.

Table of Contents
Features
Project Architecture
Installation and Setup
How to Run
Operation Modes
Chat Mode
Investigation Mode
Metadata Mode
Advanced Forensic Analysis
Contribution
License
Features
Intelligent Chat: Interact with the assistant in multiple languages with options for Technical or Freeform (creative) responses.
Forensic Investigation: Perform online investigations using a target query, retrieving data from multiple websites and conducting detailed forensic analysis.
Image Metadata Extraction: Analyze EXIF metadata from an image provided via URL, including GPS coordinate conversion for Google Maps integration.
Neural Model Integration: Uses the Mistral-7B-Instruct model via llama_cpp to generate responses and perform language translations as needed.
Local Caching: Implements an in-memory caching system to improve performance for repeated queries (optimized for Windows environments).
Interactive Web Interface: A responsive front end built with HTML, CSS, and JavaScript for user-friendly interaction.
Performance Monitoring: Integrated metrics via Prometheus to monitor request counts and response latencies.
Project Architecture
Cyber Assistant v4.0 is structured as follows:

Backend (Flask): Handles user requests, interacts with the neural model for generating responses, and executes forensic analysis and image metadata extraction.
Neural Model: Loaded using the llama_cpp library and automatically downloaded from the Hugging Face Hub if not available locally.
Front-end: A dynamic web interface for configuring modes and interacting with the assistant.
Supporting Libraries: Utilizes libraries such as nltk, langdetect, duckduckgo_search, Pillow, among others, for language processing, image analysis, and forensic data extraction.
Installation and Setup
Prerequisites
Python 3.8 or higher
Git (for cloning the repository)
Virtualenv (recommended for environment isolation)
Installation Steps
Clone the Repository:

bash
Copiar
Editar
git clone https://github.com/g0dux/CHATCYBER-BETA-.git
cd cyber-assistant-v4
Create a Virtual Environment (Recommended):

Linux/MacOS:
bash
Copiar
Editar
python -m venv venv
source venv/bin/activate
Windows:
bash
Copiar
Editar
python -m venv venv
venv\Scripts\activate
Install Dependencies:

bash
Copiar
Editar
pip install -r requirements.txt
Main dependencies include:

Flask
nltk
langdetect
cachetools
requests
psutil
llama_cpp
huggingface_hub
duckduckgo_search
Pillow
Initial NLTK Setup:

On first run, the project will download the necessary NLTK data (e.g., punkt and vader_lexicon).

How to Run
After installing the dependencies, start the application by running:

bash
Copiar
Editar
python app.py
The Flask server will start at http://0.0.0.0:5000/. Open your web browser and navigate to this URL to use Cyber Assistant v4.0.

Operation Modes
The application supports three main modes, configurable via the web interface:

Chat Mode
Description: Engage in a conversational exchange with the assistant.
Features:
Choose your preferred language (Português, English, Español, Français, Deutsch).
Select the response style: Technical or Freeform.
Investigation Mode
Description: Conduct online investigations based on a target query.
Features:
Define the number of websites to search.
Specify an investigation focus.
Enable options to include news or leaked data in the search.
Output: Generates a detailed report with extracted links, forensic analysis, and additional findings.
Metadata Mode
Description: Extract and display EXIF metadata from an image via its URL.
Features:
Provides detailed image metadata.
Converts GPS coordinates to a Google Maps link when available.
Advanced Forensic Analysis
The enhanced forensic analysis function extracts and correlates critical data useful for digital investigations, including:

IPv4 and IPv6 Addresses
Email Addresses and Phone Numbers
URLs and MAC Addresses
Hashes: MD5, SHA1, and SHA256
Vulnerability IDs (e.g., CVE)
This functionality is designed to help identify patterns and gather evidence for cybersecurity and digital forensic investigations.

Contribution
Contributions are welcome! To contribute:

Fork the Repository
Create a New Branch:
bash
Copiar
Editar
git checkout -b my-new-feature
Make Your Changes and Commit:
bash
Copiar
Editar
git commit -m "Add new feature"
Push Your Branch:
bash
Copiar
Editar
git push origin my-new-feature
Open a Pull Request on GitHub detailing your changes.
License
This project is licensed under the MIT License.

Feel free to open issues and suggest improvements. Enjoy exploring the various features of Cyber Assistant v4.0 and contributing to its development!
