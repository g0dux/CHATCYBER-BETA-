# Cyber Assistant v4.0

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)  
[![Flask](https://img.shields.io/badge/Flask-v2.0+-blue)](https://palletsprojects.com/p/flask/)  
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)  
[![GitHub Issues](https://img.shields.io/github/issues/g0dux/CHATCYBER-BETA-)](https://github.com/g0dux/CHATCYBER-BETA-/issues)  
[![GitHub Stars](https://img.shields.io/github/stars/g0dux/CHATCYBER-BETA-)](https://github.com/g0dux/CHATCYBER-BETA-/stargazers)

> **Cyber Assistant v4.0** is an interactive web application that combines an intelligent chat assistant with advanced forensic investigation and image metadata extraction. Enjoy a highly responsive and dynamic UI enhanced with animations and interactive elements for a seamless user experience.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Operation Modes](#operation-modes)
  - [Chat Mode](#chat-mode)
  - [Investigation Mode](#investigation-mode)
  - [Metadata Mode](#metadata-mode)
- [User Interfaces](#user-interfaces)
  - [Flask Interface](#flask-interface)
  - [Gradio Interface](#gradio-interface)
- [Interactive Animations & UI Enhancements](#interactive-animations--ui-enhancements)
- [Advanced Forensic Analysis](#advanced-forensic-analysis)
- [Running on Google Colab](#running-on-google-colab)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Screenshots](#screenshots)

---

## Overview

Cyber Assistant v4.0 offers a modern, dynamic approach to digital investigations and AI-powered chat. With its clean design, responsive animations, and interactive controls, users can explore technical details or creative insights in a highly engaging environment.

---

## Features

- **Intelligent Chat:**  
  Engage with the assistant in multiple languages, switching between **Technical** (detailed) or **Freeform** (creative) responses.

- **Forensic Investigation:**  
  Search and analyze online data with configurable options like the number of websites, investigation focus (e.g., phishing, malware), and inclusion of news or leaked data.

- **Image Metadata Extraction:**  
  Extract EXIF metadata—including automatic GPS coordinate conversion into clickable Google Maps links.

- **Neural Model Integration:**  
  Powered by the Mistral-7B-Instruct model (via `llama_cpp`), it offers context-aware responses and supports real-time language translation.

- **Local Caching:**  
  An in-memory cache minimizes redundant processing for repeated queries.

- **Customizable Interfaces:**  
  Choose between a classic Flask interface or a modern Gradio interface, each fully configurable to suit your needs.

- **Performance Monitoring:**  
  Integrated Prometheus metrics track request counts and response latencies.

---

## Project Architecture

Cyber Assistant v4.0 is built on a modular architecture:

- **Backend (Flask):**  
  Handles HTTP requests, integrates the neural model, and processes forensic and metadata extraction tasks.

- **Neural Model:**  
  Loaded via `llama_cpp` and automatically downloaded from the Hugging Face Hub if not available locally.

- **Front-end Interfaces:**  
  - **Flask Interface:** A traditional web UI with HTML, CSS, JavaScript, and interactive animations.  
  - **Gradio Interface:** A modern UI delivering real-time feedback and dynamic animations.

- **Supporting Libraries:**  
  Uses libraries like `nltk`, `langdetect`, `duckduckgo_search`, and `Pillow` for language processing, search, and image analysis.

---

## Installation and Setup

### Prerequisites

- **Python 3.8 or higher**
- **Git** (to clone the repository)
- **Virtualenv** (recommended for isolating dependencies)

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/g0dux/CHATCYBER-BETA-.git
   cd cyber-assistant-v4
