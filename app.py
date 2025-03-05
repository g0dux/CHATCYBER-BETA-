import sys
# Patch para contornar a ausência do módulo 'distutils'
try:
    import distutils
except ImportError:
    try:
        import setuptools._distutils as distutils
        sys.modules['distutils'] = distutils
    except ImportError:
        pass

import os
import time
import re
import logging
import requests
import io
import psutil
import threading
import subprocess
import nltk
import concurrent.futures
import email
from email import policy
from email.parser import BytesParser
import numpy as np
from sklearn.ensemble import IsolationForest
from flask import Flask, request, jsonify, render_template_string, Response
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
import emoji
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from duckduckgo_search import DDGS
from PIL import Image, ExifTags
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import tempfile
import multiprocessing
import socket  # Necessário para descoberta de IP

# Tenta importar pyshark para análise de tráfego de rede
try:
    import pyshark
except ImportError:
    pyshark = None
    logging.warning("pyshark não está instalado. A funcionalidade de análise de rede não estará disponível.")

# Configurações iniciais do NLTK (necessário apenas na primeira execução)
nltk.download('punkt')
nltk.download('vader_lexicon')

# Configurações gerais e logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sentiment_analyzer = SentimentIntensityAnalyzer()

LANGUAGE_MAP = {
    'Português': {'code': 'pt-BR', 'instruction': 'Responda em português brasileiro'},
    'English': {'code': 'en-US', 'instruction': 'Respond in English'},
    'Español': {'code': 'es-ES', 'instruction': 'Responde en español'},
    'Français': {'code': 'fr-FR', 'instruction': 'Réponds en français'},
    'Deutsch': {'code': 'de-DE', 'instruction': 'Antworte auf Deutsch'}
}

DEFAULT_MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
DEFAULT_MODEL_FILE = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
DEFAULT_LOCAL_MODEL_DIR = "models"

app = Flask(__name__)

# ===== Endpoint da Página Principal =====
index_html = """ 
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Cyber Assistant v4.0</title>
  <style>
    /* Reset e estilo base */
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background-color: #1e1e1e;
      color: #d1d5db;
      font-family: 'Inter', sans-serif;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    header {
      background-color: #111827;
      color: #fff;
      padding: 20px;
      text-align: center;
      font-size: 24px;
      font-weight: 600;
      border-bottom: 1px solid #374151;
    }
    .nav-tabs {
      display: flex;
      background-color: #1e293b;
      border-bottom: 1px solid #374151;
    }
    .nav-tabs button {
      flex: 1;
      padding: 15px 20px;
      background: none;
      border: none;
      color: #9ca3af;
      font-size: 16px;
      cursor: pointer;
      transition: background-color 0.2s;
    }
    .nav-tabs button:hover {
      background-color: #27303f;
    }
    .nav-tabs button.active {
      background-color: #374151;
      color: #fff;
      border-bottom: 2px solid #3b82f6;
    }
    .container {
      flex: 1;
      max-width: 800px;
      margin: 20px auto;
      padding: 10px;
    }
    .chat-window {
      background-color: #1f2937;
      border-radius: 8px;
      padding: 20px;
      height: 500px;
      overflow-y: auto;
      margin-bottom: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .message {
      margin-bottom: 15px;
      display: flex;
    }
    .message.user {
      justify-content: flex-end;
    }
    .message.ai {
      justify-content: flex-start;
    }
    .bubble {
      padding: 10px 15px;
      border-radius: 12px;
      max-width: 70%;
      word-wrap: break-word;
      font-size: 16px;
      line-height: 1.5;
    }
    .message.user .bubble {
      background-color: #3b82f6;
      color: #fff;
      border-radius: 12px 12px 0 12px;
    }
    .message.ai .bubble {
      background-color: #374151;
      color: #d1d5db;
      border-radius: 12px 12px 12px 0;
    }
    .input-area {
      display: flex;
      gap: 10px;
    }
    .input-area input[type="text"] {
      flex: 1;
      padding: 12px;
      border: 1px solid #374151;
      border-radius: 5px;
      background-color: #1f2937;
      color: #d1d5db;
      font-size: 16px;
    }
    .input-area button {
      padding: 12px 20px;
      border: none;
      background-color: #3b82f6;
      color: #fff;
      border-radius: 5px;
      cursor: pointer;
      font-size: 16px;
      transition: background-color 0.2s;
    }
    .input-area button:hover {
      background-color: #2563eb;
    }
    .form-options {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      font-size: 14px;
      color: #9ca3af;
    }
    .form-options label {
      margin-right: 10px;
    }
    /* Opções de investigação (ocultas por padrão) */
    #investigationOptions { display: none; }
    /* Spinner de carregamento */
    #loadingSpinner {
      display: none;
      margin: 10px auto;
      border: 6px solid #f3f3f3;
      border-top: 6px solid #3b82f6;
      border-radius: 50%;
      width: 30px;
      height: 30px;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    footer {
      background-color: #111827;
      color: #9ca3af;
      text-align: center;
      padding: 10px;
      font-size: 12px;
      border-top: 1px solid #374151;
    }
    /* Estilos para o modal de configurações */
    #configModal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.6);
      justify-content: center;
      align-items: center;
      z-index: 1000;
    }
    #configModal .modal-content {
      background: #1e1e1e;
      padding: 20px;
      border-radius: 8px;
      max-width: 500px;
      width: 90%;
      color: #d1d5db;
    }
    #configModal .modal-content h2 {
      margin-bottom: 15px;
    }
    #configModal .modal-content form > div {
      margin-bottom: 10px;
    }
    #configModal label {
      display: block;
      margin-bottom: 4px;
    }
    #configModal input[type="number"],
    #configModal input[type="color"] {
      width: 100%;
      padding: 6px;
      border: 1px solid #374151;
      border-radius: 4px;
      background-color: #1f2937;
      color: #d1d5db;
    }
    #configModal button {
      padding: 8px 12px;
      border: none;
      background-color: #3b82f6;
      color: #fff;
      border-radius: 4px;
      cursor: pointer;
      margin-right: 10px;
    }
    /* Estilo para o botão de configurações flutuante */
    #configToggleButton {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background-color: #3b82f6;
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 50px;
      height: 50px;
      cursor: pointer;
      font-size: 24px;
      z-index: 1001;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    /* Botão de limpar chat */
    #clearChat {
      margin-top: 10px;
      padding: 10px 20px;
      background-color: #ef4444;
      color: #fff;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      font-size: 14px;
      transition: background-color 0.2s;
    }
    #clearChat:hover {
      background-color: #dc2626;
    }
  </style>
  <!-- Estilo personalizado atualizado via configurações -->
  <style id="customStyles"></style>
</head>
<body>
  <header>Cyber Assistant v4.0</header>
  <div class="nav-tabs">
    <button class="tab-button active" data-tab="chatTab">Chat</button>
  </div>
  <div class="container">
    <div id="chatTab" class="tab-content">
      <div class="chat-window" id="chatWindow"></div>
      <!-- Spinner de carregamento -->
      <div id="loadingSpinner"></div>
      <form id="chatForm">
        <div class="input-area">
          <input type="text" id="chatInput" name="user_input" placeholder="Digite sua mensagem ou URL...">
          <button type="submit">Enviar</button>
        </div>
        <div class="form-options">
          <label for="mode">Modo:</label>
          <select id="mode" name="mode">
            <option value="Chat">Chat</option>
            <option value="Investigação">Investigação</option>
            <option value="Metadados">Metadados</option>
          </select>
          <label for="language">Idioma:</label>
          <select id="language" name="language">
            <option value="Português">Português</option>
            <option value="English">English</option>
            <option value="Español">Español</option>
            <option value="Français">Français</option>
            <option value="Deutsch">Deutsch</option>
          </select>
          <label for="style">Estilo:</label>
          <select id="style" name="style">
            <option value="Técnico">Técnico</option>
            <option value="Livre">Livre</option>
          </select>
          <label>
            <input type="checkbox" id="streaming" name="streaming"> Ativar Streaming
          </label>
        </div>
        <!-- Opções adicionais para investigação -->
        <div class="form-options" id="investigationOptions">
          <label for="sites_meta">Meta de sites:</label>
          <input type="number" id="sites_meta" name="sites_meta" value="5" style="width: 60px;">
          <label for="investigation_focus">Foco (opcional):</label>
          <input type="text" id="investigation_focus" name="investigation_focus" placeholder="Ex: phishing, malware...">
          <label>
            <input type="checkbox" id="search_news" name="search_news"> Ativar Notícias
          </label>
          <label>
            <input type="checkbox" id="search_leaked_data" name="search_leaked_data"> Ativar Dados Vazados
          </label>
        </div>
      </form>
      <!-- Botão para limpar o chat -->
      <button id="clearChat">Limpar Chat</button>
    </div>
  </div>
  <footer>© 2025 Cyber Assistant</footer>
  
  <!-- Botão flutuante para abrir as configurações -->
  <button id="configToggleButton">⚙️</button>
  
  <!-- Modal de configurações -->
  <div id="configModal">
    <div class="modal-content">
      <h2>Configurações</h2>
      <form id="configForm">
        <div>
          <label for="inputFontSize">Tamanho da fonte da entrada de texto (px):</label>
          <input type="number" id="inputFontSize" name="inputFontSize" value="16" min="10" max="30">
        </div>
        <div>
          <label for="inputWidth">Largura da entrada de texto (px):</label>
          <input type="number" id="inputWidth" name="inputWidth" value="300" min="100" max="800">
        </div>
        <div>
          <label for="chatWindowHeight">Altura da janela de chat (px):</label>
          <input type="number" id="chatWindowHeight" name="chatWindowHeight" value="500" min="300" max="800">
        </div>
        <div>
          <label for="chatWindowWidth">Largura da janela de chat (px):</label>
          <input type="number" id="chatWindowWidth" name="chatWindowWidth" value="800" min="400" max="1200">
        </div>
        <div>
          <label for="bodyBgColor">Cor de fundo do Body:</label>
          <input type="color" id="bodyBgColor" name="bodyBgColor" value="#1e1e1e">
        </div>
        <div>
          <label for="chatBgColor">Cor de fundo da janela de chat:</label>
          <input type="color" id="chatBgColor" name="chatBgColor" value="#1f2937">
        </div>
        <div>
          <label for="chatTextColor">Cor do texto da janela de chat:</label>
          <input type="color" id="chatTextColor" name="chatTextColor" value="#d1d5db">
        </div>
        <div>
          <label for="userBubbleColor">Cor da bolha do usuário:</label>
          <input type="color" id="userBubbleColor" name="userBubbleColor" value="#3b82f6">
        </div>
        <div>
          <label for="aiBubbleColor">Cor da bolha da IA:</label>
          <input type="color" id="aiBubbleColor" name="aiBubbleColor" value="#374151">
        </div>
        <div style="margin-top: 10px;">
          <button type="button" id="saveConfig">Salvar</button>
          <button type="button" id="cancelConfig">Cancelar</button>
        </div>
      </form>
    </div>
  </div>
  
  <script>
    // Alterna exibição das opções de investigação conforme o modo selecionado
    document.getElementById("mode").addEventListener("change", function() {
      const mode = this.value;
      const invOptions = document.getElementById("investigationOptions");
      invOptions.style.display = mode === "Investigação" ? "flex" : "none";
    });
    
    // Função para adicionar mensagem à janela de chat
    function appendMessage(windowId, sender, message) {
      const windowElement = document.getElementById(windowId);
      const messageDiv = document.createElement('div');
      messageDiv.className = `message ${sender}`;
      const bubbleDiv = document.createElement('div');
      bubbleDiv.className = 'bubble';
      bubbleDiv.innerHTML = message;
      messageDiv.appendChild(bubbleDiv);
      windowElement.appendChild(messageDiv);
      windowElement.scrollTop = windowElement.scrollHeight;
    }
    
    // Função para adicionar mensagem de tempo de resposta
    function appendTimer(time) {
      appendMessage("chatWindow", "ai", `<em>Tempo de resposta: ${time.toFixed(2)} segundos</em>`);
    }
    
    // Função para limpar o chat
    document.getElementById("clearChat").addEventListener("click", function() {
      document.getElementById("chatWindow").innerHTML = "";
    });
    
    // Exibe o spinner de carregamento
    function showSpinner() {
      document.getElementById("loadingSpinner").style.display = "block";
    }
    
    // Oculta o spinner de carregamento
    function hideSpinner() {
      document.getElementById("loadingSpinner").style.display = "none";
    }
    
    // Envio do formulário de chat
    document.getElementById("chatForm").addEventListener("submit", function(e) {
      e.preventDefault();
      const inputField = document.getElementById("chatInput");
      const message = inputField.value.trim();
      if (!message) return;
      appendMessage("chatWindow", "user", message);
      inputField.value = "";
      
      const formData = new FormData();
      formData.append('user_input', message);
      formData.append('mode', document.getElementById("mode").value);
      formData.append('language', document.getElementById("language").value);
      formData.append('style', document.getElementById("style").value);
      formData.append('sites_meta', document.getElementById("sites_meta").value);
      formData.append('investigation_focus', document.getElementById("investigation_focus").value);
      formData.append('search_news', document.getElementById("search_news").checked);
      formData.append('search_leaked_data', document.getElementById("search_leaked_data").checked);
      
      const streaming = document.getElementById("streaming").checked;
      showSpinner();
      
      // Registra o tempo de início
      const startTime = Date.now();
      
      // Define a URL com parâmetro 'stream' se streaming estiver ativado
      const url = streaming ? '/ask?stream=true' : '/ask';
      
      fetch(url, { method: 'POST', body: formData })
      .then(response => {
        if (streaming) {
          // Cria uma única mensagem para a resposta da IA e atualiza o conteúdo aos poucos
          const chatWindow = document.getElementById("chatWindow");
          const aiMessageDiv = document.createElement('div');
          aiMessageDiv.className = 'message ai';
          const bubbleDiv = document.createElement('div');
          bubbleDiv.className = 'bubble';
          aiMessageDiv.appendChild(bubbleDiv);
          chatWindow.appendChild(aiMessageDiv);
          chatWindow.scrollTop = chatWindow.scrollHeight;
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          function readStream() {
            return reader.read().then(({ done, value }) => {
              if (done) {
                hideSpinner();
                const elapsed = (Date.now() - startTime) / 1000;
                appendTimer(elapsed);
                return;
              }
              const chunk = decoder.decode(value);
              bubbleDiv.innerHTML += chunk;
              chatWindow.scrollTop = chatWindow.scrollHeight;
              return readStream();
            });
          }
          return readStream();
        } else {
          return response.json();
        }
      })
      .then(data => {
        if (!streaming) {
          hideSpinner();
          const elapsed = (Date.now() - startTime) / 1000;
          appendMessage("chatWindow", "ai", data.response || "Erro: " + data.error);
          appendTimer(elapsed);
        }
      })
      .catch(err => {
        hideSpinner();
        appendMessage("chatWindow", "ai", "Erro na requisição: " + err);
      });
    });
    
    // Configurações: exibição do modal
    const configModal = document.getElementById("configModal");
    const configToggleButton = document.getElementById("configToggleButton");
    
    configToggleButton.addEventListener("click", () => {
      configModal.style.display = "flex";
    });
    
    // Botões do modal de configurações
    document.getElementById("cancelConfig").addEventListener("click", () => {
      configModal.style.display = "none";
    });
    
    document.getElementById("saveConfig").addEventListener("click", () => {
      // Obter valores do formulário de configurações
      const inputFontSize = document.getElementById("inputFontSize").value;
      const inputWidth = document.getElementById("inputWidth").value;
      const chatWindowHeight = document.getElementById("chatWindowHeight").value;
      const chatWindowWidth = document.getElementById("chatWindowWidth").value;
      const bodyBgColor = document.getElementById("bodyBgColor").value;
      const chatBgColor = document.getElementById("chatBgColor").value;
      const chatTextColor = document.getElementById("chatTextColor").value;
      const userBubbleColor = document.getElementById("userBubbleColor").value;
      const aiBubbleColor = document.getElementById("aiBubbleColor").value;
      
      // Atualizar estilos personalizados via elemento de estilo
      const customStyles = document.getElementById("customStyles");
      customStyles.innerHTML = `
        #chatInput { font-size: ${inputFontSize}px; width: ${inputWidth}px; }
        #chatWindow { height: ${chatWindowHeight}px; width: ${chatWindowWidth}px; background-color: ${chatBgColor}; color: ${chatTextColor}; }
        .message.user .bubble { background-color: ${userBubbleColor}; }
        .message.ai .bubble { background-color: ${aiBubbleColor}; }
        body { background-color: ${bodyBgColor}; }
      `;
      
      // Fechar o modal de configurações
      configModal.style.display = "none";
    });
  </script>
</body>
</html>
"""
@app.route('/')
def index():
    return render_template_string(index_html)

# ===== Cache simples em memória =====
cache = {}

def get_cached_response(query: str, lang: str, style: str) -> str:
    key = f"response:{query}:{lang}:{style}"
    return cache.get(key)

def set_cached_response(query: str, lang: str, style: str, response_text: str, ttl: int = 3600) -> None:
    key = f"response:{query}:{lang}:{style}"
    cache[key] = response_text  # TTL não implementado nesta versão

# ===== Monitoramento com Prometheus =====
REQUEST_COUNT = Counter('flask_request_count', 'Total de requisições', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('flask_request_latency_seconds', 'Tempo de resposta', ['endpoint'])

@app.before_request
def before_request():
    request.start_time = time.time()
    REQUEST_COUNT.labels(request.path, request.method).inc()

@app.after_request
def after_request(response):
    resp_time = time.time() - request.start_time
    REQUEST_LATENCY.labels(request.path).observe(resp_time)
    return response

@app.route('/metrics')
def metrics():
    data = generate_latest()
    return Response(data, mimetype=CONTENT_TYPE_LATEST)

# ===== Pré-compilação dos padrões de regex para análise forense =====
COMPILED_REGEX_PATTERNS = {
    # Padrões originais
    'ip': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    'ipv6': re.compile(r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b'),
    'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    'phone': re.compile(r'\+?\d[\d\s()-]{7,}\d'),
    'url': re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'),
    'mac': re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b'),
    'md5': re.compile(r'\b[a-fA-F0-9]{32}\b'),
    'sha1': re.compile(r'\b[a-fA-F0-9]{40}\b'),
    'sha256': re.compile(r'\b[a-fA-F0-9]{64}\b'),
    'cve': re.compile(r'\bCVE-\d{4}-\d{4,7}\b'),
    'imei': re.compile(r'\b\d{15}\b'),
    'cpf': re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b'),
    'cnpj': re.compile(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'uuid': re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'),
    'cc': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    'btc': re.compile(r'\b(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b'),
    # Novos padrões para investigação e IA
    'ethereum': re.compile(r'\b0x[a-fA-F0-9]{40}\b'),
    'jwt': re.compile(r'\beyJ[a-zA-Z0-9-_]+?\.[a-zA-Z0-9-_]+?\.[a-zA-Z0-9-_]+?\b'),
    'cidr': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}/\d{1,2}\b'),
    'iso8601': re.compile(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}'),
    'sha512': re.compile(r'\b[a-fA-F0-9]{128}\b'),
    'base64': re.compile(r'\b(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b'),
    'google_api_key': re.compile(r'\bAIza[0-9A-Za-z-_]{35}\b'),
    'iban': re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b'),
    'us_phone': re.compile(r'\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    'twitter_handle': re.compile(r'@\w{1,15}\b'),
    'date_mmddyyyy': re.compile(r'\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/\d{4}\b'),
    'win_path': re.compile(r'\b[a-zA-Z]:\\(?:[^\\\/:*?"<>|\r\n]+\\)*[^\\\/:*?"<>|\r\n]+\b'),
    'ipv4_port': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}:\d+\b'),
    'http_status': re.compile(r'HTTP/\d\.\d"\s(\d{3})\b'),
    'version': re.compile(r'\b\d+\.\d+(\.\d+)?\b')
}

# ===== Funções de Download e Carregamento do Modelo =====
def download_model() -> None:
    try:
        logger.info("⏬ Baixando Modelo...")
        hf_hub_download(
            repo_id=DEFAULT_MODEL_NAME,
            filename=DEFAULT_MODEL_FILE,
            local_dir=DEFAULT_LOCAL_MODEL_DIR,
            resume_download=True
        )
    except Exception as e:
        logger.error(f"❌ Falha no Download: {e}")
        raise e

def load_model() -> Llama:
    model_path = os.path.join(DEFAULT_LOCAL_MODEL_DIR, DEFAULT_MODEL_FILE)
    if not os.path.exists(model_path):
        download_model()
    try:
        n_gpu_layers = 15
        n_batch = 512
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                n_gpu_layers = -1
                n_batch = 1024
                logger.info("GPU detectada. Utilizando todas as camadas na GPU (n_gpu_layers=-1) e n_batch=1024.")
            else:
                logger.info("Nenhuma GPU detectada. Usando configuração otimizada para CPU.")
        except Exception as gpu_error:
            logger.warning(f"Erro na detecção da GPU: {gpu_error}. Configuração para CPU será utilizada.")
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=psutil.cpu_count(logical=True),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch
        )
        logger.info(f"🤖 Modelo Neural Carregado com n_gpu_layers={n_gpu_layers}, n_batch={n_batch} e n_threads={psutil.cpu_count(logical=True)}")
        return model
    except Exception as e:
        logger.error(f"❌ Erro na Inicialização do Modelo: {e}")
        raise e

model = load_model()

# ===== Funções para Geração de Resposta e Validação de Idioma =====
def build_messages(query: str, lang_config: dict, style: str) -> tuple[list, float]:
    if style == "Técnico":
        system_instruction = f"{lang_config['instruction']}. Seja detalhado e técnico."
        temperature = 0.7
    else:
        system_instruction = f"{lang_config['instruction']}. Responda de forma livre e criativa."
        temperature = 0.9
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": query}
    ]
    return messages, temperature

def generate_response(query: str, lang: str, style: str) -> str:
    start_time = time.time()
    cached_text = get_cached_response(query, lang, style)
    if cached_text:
        logger.info(f"✅ Resposta obtida do cache em {time.time() - start_time:.2f}s")
        return cached_text
    lang_config = LANGUAGE_MAP.get(lang, LANGUAGE_MAP['Português'])
    messages, temperature = build_messages(query, lang_config, style)
    try:
        response = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=800,
            stop=["</s>"]
        )
        raw_response = response['choices'][0]['message']['content']
        final_response = validate_language(raw_response, lang_config)
        logger.info(f"✅ Resposta gerada em {time.time() - start_time:.2f}s")
        set_cached_response(query, lang, style, final_response)
        return final_response
    except Exception as e:
        logger.error(f"❌ Erro ao gerar resposta: {e}")
        return f"Erro ao gerar resposta: {e}"

def validate_language(text: str, lang_config: dict) -> str:
    try:
        detected_lang = detect(text)
        expected_lang = lang_config['code'].split('-')[0]
        if detected_lang != expected_lang:
            logger.info(f"Idioma detectado ({detected_lang}) difere do esperado ({expected_lang}). Corrigindo...")
            return correct_language(text, lang_config)
        return text
    except Exception as e:
        logger.warning(f"⚠️ Falha na detecção de idioma: {e}. Retornando texto original.")
        return text

def correct_language(text: str, lang_config: dict) -> str:
    try:
        correction_prompt = f"Traduza para {lang_config['instruction']}:\n{text}"
        corrected = model.create_chat_completion(
            messages=[{"role": "user", "content": correction_prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        corrected_text = corrected['choices'][0]['message']['content']
        return f"[Traduzido]\n{corrected_text}"
    except Exception as e:
        logger.error(f"❌ Erro na correção de idioma: {e}")
        return text

# ===== Novo Modo: Descoberta de IP =====
def discover_ip(target: str) -> dict:
    """
    Tenta descobrir o(s) IP(s) do alvo informado.
    Se o alvo já for um IP válido (IPv4 ou IPv6), retorna-o diretamente.
    Caso contrário, realiza resolução DNS para obter hostname, aliases e IPs.
    """
    try:
        # Verifica se o target já é um IP válido
        ipv4_pattern = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')
        ipv6_pattern = re.compile(r'^[A-Fa-f0-9:]+$')
        if ipv4_pattern.match(target) or ipv6_pattern.match(target):
            return {'target': target, 'ip': target, 'method': 'Já é um IP válido'}
        # Caso contrário, realiza resolução DNS
        hostname, aliases, ip_addresses = socket.gethostbyname_ex(target)
        return {
            'target': target,
            'hostname': hostname,
            'aliases': aliases,
            'ip_addresses': ip_addresses,
            'method': 'Resolução DNS'
        }
    except Exception as e:
        logger.error(f"Erro na descoberta de IP para {target}: {e}")
        return {'error': str(e)}

@app.route('/ip_discovery', methods=['POST'])
def ip_discovery():
    target = request.form.get('target', '').strip()
    if not target:
        return jsonify({'error': 'Erro: Por favor, insira um alvo para descoberta de IP.'}), 400
    result = discover_ip(target)
    return jsonify(result)

# ===== Endpoint /ask para chat com a IA =====
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('user_input', '')
    mode = request.form.get('mode', 'Chat')
    lang = request.form.get('language', 'Português')
    style = request.form.get('style', 'Técnico')
    
    if mode == "Investigação":
        if not user_input.strip():
            return jsonify({'response': "Erro: Por favor, insira um alvo para investigação."})
        try:
            sites_meta = int(request.form.get('sites_meta', 5))
            investigation_focus = request.form.get('investigation_focus', '')
            search_news = request.form.get('search_news', 'false').lower() == 'true'
            search_leaked_data = request.form.get('search_leaked_data', 'false').lower() == 'true'
            response_text = process_investigation(user_input, sites_meta, investigation_focus, search_news, search_leaked_data)
            return jsonify({'response': response_text})
        except Exception as e:
            logger.error(f"Erro no modo Investigação: {e}")
            return jsonify({'error': str(e)}), 500
    elif mode == "Metadados":
        if not user_input.strip():
            return jsonify({'response': "Erro: Por favor, insira um link de imagem."})
        try:
            meta = analyze_image_metadata(user_input)
            formatted_meta = "<br>".join(f"{k}: {v}" for k, v in meta.items())
            return jsonify({'response': formatted_meta})
        except Exception as e:
            logger.error(f"Erro no modo Metadados: {e}")
            return jsonify({'error': str(e)}), 500
    else:  # Modo Chat
        try:
            response_text = generate_response(user_input, lang, style)
            return jsonify({'response': response_text})
        except Exception as e:
            logger.error(f"Erro no modo Chat: {e}")
            return jsonify({'error': str(e)}), 500

# ===== Função para Streaming de Respostas (caso necessário) =====
def streaming_response(text: str, chunk_size: int = 200):
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]
        time.sleep(0.1)

# ===== Análise Forense e Processamento de Texto =====
def advanced_forensic_analysis(text: str) -> dict:
    forensic_info = {}
    try:
        for key, pattern in COMPILED_REGEX_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                label = {
                    'ip': 'Endereços IPv4',
                    'ipv6': 'Endereços IPv6',
                    'email': 'E-mails',
                    'phone': 'Telefones',
                    'url': 'URLs',
                    'mac': 'Endereços MAC',
                    'md5': 'Hashes MD5',
                    'sha1': 'Hashes SHA1',
                    'sha256': 'Hashes SHA256',
                    'cve': 'IDs CVE',
                    'imei': 'IMEI',
                    'cpf': 'CPF',
                    'cnpj': 'CNPJ',
                    'ssn': 'SSN',
                    'uuid': 'UUID',
                    'cc': 'Cartões de Crédito',
                    'btc': 'Endereço Bitcoin',
                    'ethereum': 'Ethereum Address',
                    'jwt': 'Token JWT',
                    'cidr': 'CIDR Notation',
                    'iso8601': 'Timestamp ISO8601',
                    'sha512': 'Hashes SHA-512',
                    'base64': 'String Base64',
                    'google_api_key': 'Chave API Google',
                    'iban': 'Número IBAN',
                    'us_phone': 'Telefone (EUA)',
                    'twitter_handle': 'Twitter Handle',
                    'date_mmddyyyy': 'Data (MM/DD/YYYY)',
                    'win_path': 'Caminho Windows',
                    'ipv4_port': 'IPv4 com Porta',
                    'http_status': 'Código HTTP',
                    'version': 'Número de Versão'
                }.get(key, key)
                forensic_info[label] = list(set(matches))
    except Exception as e:
        logger.error(f"❌ Erro durante a análise forense: {e}")
    return forensic_info

def convert_to_degrees(value) -> float:
    try:
        d, m, s = value
        degrees = d[0] / d[1]
        minutes = m[0] / m[1] / 60
        seconds = s[0] / s[1] / 3600
        return degrees + minutes + seconds
    except Exception as e:
        logger.error(f"❌ Erro na conversão de coordenadas: {e}")
        raise e

def analyze_image_metadata(url: str) -> dict:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_data = response.content
        image = Image.open(io.BytesIO(image_data))
        exif = image._getexif()
        meta = {}
        if exif:
            for tag_id, value in exif.items():
                tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                meta[tag] = value
            if "GPSInfo" in meta:
                gps_info = meta["GPSInfo"]
                try:
                    lat = convert_to_degrees(gps_info.get(2))
                    if gps_info.get(1) != "N":
                        lat = -lat
                    lon = convert_to_degrees(gps_info.get(4))
                    if gps_info.get(3) != "E":
                        lon = -lon
                    meta["GPS Coordinates"] = f"{lat}, {lon} (Google Maps: https://maps.google.com/?q={lat},{lon})"
                except Exception as e:
                    meta["GPS Extraction Error"] = str(e)
        else:
            meta["info"] = "Nenhum metadado EXIF encontrado."
        return meta
    except Exception as e:
        logger.error(f"❌ Erro ao analisar metadados da imagem: {e}")
        return {"error": str(e)}

# ===== Funcionalidades de Investigação Online =====
def perform_search(query: str, search_type: str, max_results: int) -> list:
    try:
        with DDGS() as ddgs:
            if search_type == 'web':
                return list(ddgs.text(keywords=query, max_results=max_results))
            elif search_type == 'news':
                return list(ddgs.news(keywords=query, max_results=max_results))
            elif search_type == 'leaked':
                return list(ddgs.text(keywords=f"{query} leaked", max_results=max_results))
            else:
                return []
    except Exception as e:
        logger.error(f"Erro na busca ({search_type}): {e}")
        return []

def format_search_results(results: list, section_title: str) -> tuple:
    count = len(results)
    info_message = f"Apenas {count} resultados encontrados para '{section_title}'.<br>" if count < 1 else ""
    formatted_text = "<br>".join(
        f"• {res.get('title', 'Sem título')}<br>&nbsp;&nbsp;{res.get('href', 'Sem link')}<br>&nbsp;&nbsp;{res.get('body', '')}"
        for res in results
    )
    links_table = (
        f"<h3>{section_title}</h3>"
        "<table border='1' style='width:100%; border-collapse: collapse; text-align: left;'>"
        "<thead><tr><th>Título</th><th>Link</th></tr></thead><tbody>"
    )
    for res in results:
        title = res.get('title', 'Sem título')
        href = res.get('href', 'Sem link')
        links_table += f"<tr><td>{title}</td><td><a href='{href}' target='_blank'>{href}</a></td></tr>"
    links_table += "</tbody></table>"
    return formatted_text, links_table, info_message

def process_investigation(target: str, sites_meta: int = 5, investigation_focus: str = "",
                          search_news: bool = False, search_leaked_data: bool = False) -> str:
    logger.info(f"🔍 Iniciando investigação para: {repr(target)}")
    if not target.strip():
        return "Erro: Por favor, insira um alvo para investigação."
    
    search_tasks = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        search_tasks['web'] = executor.submit(perform_search, target, 'web', sites_meta)
        if search_news:
            search_tasks['news'] = executor.submit(perform_search, target, 'news', sites_meta)
        if search_leaked_data:
            search_tasks['leaked'] = executor.submit(perform_search, target, 'leaked', sites_meta)
    
    results_web = search_tasks['web'].result() if 'web' in search_tasks else []
    results_news = search_tasks['news'].result() if 'news' in search_tasks else []
    results_leaked = search_tasks['leaked'].result() if 'leaked' in search_tasks else []
    
    formatted_web, links_web, info_web = format_search_results(results_web, "Sites")
    formatted_news, links_news, info_news = ("", "", "") if not results_news else format_search_results(results_news, "Notícias")
    formatted_leaked, links_leaked, info_leaked = ("", "", "") if not results_leaked else format_search_results(results_leaked, "Dados Vazados")
    
    combined_results_text = ""
    if formatted_web:
        combined_results_text += "<br><br>Resultados de Sites:<br>" + formatted_web
    if formatted_news:
        combined_results_text += "<br><br>Notícias:<br>" + formatted_news
    if formatted_leaked:
        combined_results_text += "<br><br>Dados Vazados:<br>" + formatted_leaked
    
    forensic_analysis = advanced_forensic_analysis(combined_results_text)
    forensic_details = "<br>".join(f"{k}: {v}" for k, v in forensic_analysis.items() if v)
    
    investigation_prompt = f"Analise os dados obtidos sobre '{target}'"
    if investigation_focus:
        investigation_prompt += f", focando em '{investigation_focus}'"
    investigation_prompt += "<br>" + combined_results_text
    if forensic_details:
        investigation_prompt += "<br><br>Análise Forense Extraída:<br>" + forensic_details
    investigation_prompt += "<br><br>Elabore um relatório detalhado com ligações, riscos e informações relevantes."
    
    try:
        investigation_response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "Você é um perito policial e forense digital, experiente em métodos policiais de investigação. Utilize técnicas de análise de evidências, protocolos forenses e investigação digital para identificar padrões, rastrear conexões e coletar evidências relevantes. Seja minucioso, preciso e detalhado."},
                {"role": "user", "content": investigation_prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            stop=["</s>"]
        )
        report = investigation_response['choices'][0]['message']['content']
        links_combined = links_web + (links_news if links_news else "") + (links_leaked if links_leaked else "")
        final_report = report + "<br><br>Links encontrados:<br>" + links_combined
        return final_report
    except Exception as e:
        logger.error(f"❌ Erro na investigação: {e}")
        return f"Erro na investigação: {e}"

# ===== Função para Análise de E-mails =====
def analyze_email_forensics(raw_email: bytes) -> dict:
    result = {}
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_email)
        result['From'] = msg.get('From')
        result['To'] = msg.get('To')
        result['Subject'] = msg.get('Subject')
        result['Date'] = msg.get('Date')
        attachments = []
        for part in msg.walk():
            content_disposition = part.get("Content-Disposition", "")
            if "attachment" in content_disposition:
                attachments.append({
                    "filename": part.get_filename(),
                    "content_type": part.get_content_type(),
                    "size": len(part.get_payload(decode=True))
                })
        result['Attachments'] = attachments
    except Exception as e:
        result['error'] = str(e)
    return result

@app.route('/email_forensics', methods=['POST'])
def email_forensics():
    if 'email_file' not in request.files:
        return jsonify({'error': 'Arquivo de e-mail não fornecido'}), 400
    email_file = request.files['email_file']
    raw_email = email_file.read()
    analysis = analyze_email_forensics(raw_email)
    return jsonify(analysis)

# ===== Função para Análise de Comportamento de Usuário (UBA) =====
def analyze_user_behavior(user_data: list) -> dict:
    result = {}
    try:
        if not user_data:
            return {"error": "Nenhum dado de usuário fornecido"}
        keys = list(user_data[0].keys())
        X = np.array([[record[k] for k in keys] for record in user_data])
        model_uba = IsolationForest(contamination=0.1, random_state=42)
        model_uba.fit(X)
        scores = model_uba.decision_function(X)
        anomalies = model_uba.predict(X)
        analysis = []
        for i, record in enumerate(user_data):
            record_analysis = record.copy()
            record_analysis['anomaly_score'] = scores[i]
            record_analysis['is_anomaly'] = anomalies[i] == -1
            analysis.append(record_analysis)
        result['analysis'] = analysis
    except Exception as e:
        result['error'] = str(e)
    return result

@app.route('/user_behavior', methods=['POST'])
def user_behavior():
    try:
        user_data = request.get_json()
        if not user_data:
            return jsonify({'error': 'Nenhum dado de usuário fornecido'}), 400
        analysis = analyze_user_behavior(user_data)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== Função para Análise de Logs e Integração com SIEM =====
def analyze_logs_for_siem(logs: str) -> dict:
    result = {}
    try:
        error_pattern = re.compile(r'ERROR|error')
        warning_pattern = re.compile(r'WARNING|warning')
        errors = error_pattern.findall(logs)
        warnings = warning_pattern.findall(logs)
        result['error_count'] = len(errors)
        result['warning_count'] = len(warnings)
        result['total_lines'] = len(logs.splitlines())
        result['sample_lines'] = logs.splitlines()[:5]
    except Exception as e:
        result['error'] = str(e)
    return result

@app.route('/log_analysis', methods=['POST'])
def log_analysis():
    logs = request.form.get('logs', '')
    if not logs:
        return jsonify({'error': 'Nenhum log fornecido'}), 400
    analysis = analyze_logs_for_siem(logs)
    return jsonify(analysis)

# ===== Atualização na Análise de Tráfego de Rede (PCAP) com Regex =====
PCAP_UPLOAD_FOLDER = "/tmp"
NETWORK_REGEX_PATTERNS = {
    "IP_SUSPEITO": re.compile(r"\b(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3})\b"),
    "DOMINIO_SUSPEITO": re.compile(r"\b[a-z0-9.-]+\.(ru|cn|tk|xyz|info)\b"),
    "USER_AGENT_SUSPEITO": re.compile(r"curl|python-requests|wget", re.IGNORECASE)
}

def process_pcap(pcap_path):
    try:
        # Usando only_summaries para otimizar o processamento
        cap = pyshark.FileCapture(pcap_path, only_summaries=True)
        protocol_count = {}
        ip_count = {}
        port_count = {}
        alerts = []
        total_packets = 0

        for packet in cap:
            total_packets += 1

            if hasattr(packet, 'protocol'):
                protocol = packet.protocol
                protocol_count[protocol] = protocol_count.get(protocol, 0) + 1

            if hasattr(packet, 'source'):
                ip = packet.source
                ip_count[ip] = ip_count.get(ip, 0) + 1
                if NETWORK_REGEX_PATTERNS["IP_SUSPEITO"].search(ip):
                    alerts.append(f"⚠️ IP suspeito detectado: {ip}")

            if hasattr(packet, 'destination'):
                ip = packet.destination
                ip_count[ip] = ip_count.get(ip, 0) + 1

            if hasattr(packet, 'info') and packet.info:
                ports = re.findall(r":(\d+)", packet.info)
                for port in ports:
                    port_count[port] = port_count.get(port, 0) + 1

            if hasattr(packet, 'info') and packet.info:
                if NETWORK_REGEX_PATTERNS["USER_AGENT_SUSPEITO"].search(packet.info):
                    alerts.append(f"⚠️ User-Agent suspeito detectado: {packet.info}")

        cap.close()
        result = {
            "total_packets": total_packets,
            "protocol_count": dict(sorted(protocol_count.items(), key=lambda item: item[1], reverse=True)),
            "top_ip_addresses": dict(sorted(ip_count.items(), key=lambda item: item[1], reverse=True)[:5]),
            "top_ports": dict(sorted(port_count.items(), key=lambda item: item[1], reverse=True)[:5]),
            "alerts": alerts
        }
        return result

    except Exception as e:
        return {"error": str(e)}

@app.route('/network_analysis', methods=['POST'])
def network_analysis():
    if 'pcap_file' not in request.files:
        return jsonify({"error": "Arquivo PCAP não fornecido"}), 400
    file = request.files['pcap_file']
    pcap_path = os.path.join(PCAP_UPLOAD_FOLDER, file.filename)
    file.save(pcap_path)
    with multiprocessing.Pool(processes=1) as pool:
        result = pool.apply(process_pcap, (pcap_path,))
    return jsonify(result)

# ===== Funções para LocalTunnel (opcional) =====
def ensure_localtunnel_installed():
    try:
        result = subprocess.run("which lt", shell=True, capture_output=True, text=True)
        if not result.stdout.strip():
            print("LocalTunnel não encontrado. Verificando se npm está disponível...")
            npm_result = subprocess.run("which npm", shell=True, capture_output=True, text=True)
            if not npm_result.stdout.strip():
                print("npm não está disponível. Instale Node.js e npm manualmente.")
                return
            else:
                print("Instalando LocalTunnel via npm...")
                subprocess.run("npm install -g localtunnel", shell=True, check=True)
                print("LocalTunnel instalado com sucesso!")
        else:
            print("LocalTunnel já está instalado.")
    except Exception as e:
        print("Erro ao verificar ou instalar LocalTunnel:", e)

def get_tunnel_password():
    try:
        time.sleep(3)
        result = subprocess.run("curl https://loca.lt/mytunnelpassword", shell=True, capture_output=True, text=True)
        password = result.stdout.strip()
        if password:
            print("Tunnel Password:", password)
        else:
            print("Não foi possível obter o tunnel password.")
    except Exception as e:
        print("Erro ao obter tunnel password:", e)

# ===== Execução da Aplicação =====
if __name__ == '__main__':
    ensure_localtunnel_installed()

    def read_lt_output(process):
        for line in process.stdout:
            if "your url is:" in line.lower():
                print("LocalTunnel URL:", line.strip())
            else:
                print("LocalTunnel:", line.strip())

    lt_command = "lt --port 5000"
    print("Iniciando LocalTunnel para expor a aplicação na porta 5000...")
    lt_process = subprocess.Popen(lt_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lt_thread = threading.Thread(target=read_lt_output, args=(lt_process,), daemon=True)
    lt_thread.start()
    
    get_tunnel_password()
    
    print("Executando a aplicação Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
