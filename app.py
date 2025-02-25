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
from flask import Flask, request, jsonify, render_template_string
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
from cachetools import TTLCache, cached
import emoji
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from duckduckgo_search import DDGS
from PIL import Image, ExifTags

# Configurações iniciais do NLTK (necessário apenas na primeira execução)
nltk.download('punkt')
nltk.download('vader_lexicon')

# Configurações gerais e logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sentiment_analyzer = SentimentIntensityAnalyzer()
cache = TTLCache(maxsize=500, ttl=3600)

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

def download_model() -> None:
    """
    Faz o download do modelo caso não esteja disponível localmente.
    """
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
    """
    Carrega o modelo neural. Se não existir localmente, faz o download.
    """
    model_path = os.path.join(DEFAULT_LOCAL_MODEL_DIR, DEFAULT_MODEL_FILE)
    if not os.path.exists(model_path):
        download_model()
    try:
        n_gpu_layers = 33 if psutil.virtual_memory().available > 4 * 1024 ** 3 else 15
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=psutil.cpu_count(logical=True),
            n_gpu_layers=n_gpu_layers
        )
        logger.info("🤖 Modelo Neural Carregado")
        return model
    except Exception as e:
        logger.error(f"❌ Erro na Inicialização do Modelo: {e}")
        raise e

model = load_model()

def build_messages(query: str, lang_config: dict, style: str) -> tuple[list, float]:
    """
    Constrói a lista de mensagens e define a temperatura de acordo com o estilo.
    """
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

@cached(cache)
def generate_response(query: str, lang: str, style: str) -> str:
    """
    Gera uma resposta a partir da consulta, utilizando o modelo de linguagem.
    """
    start_time = time.time()
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
        return final_response
    except Exception as e:
        logger.error(f"❌ Erro ao gerar resposta: {e}")
        return f"Erro ao gerar resposta: {e}"

def validate_language(text: str, lang_config: dict) -> str:
    """
    Verifica se o texto está no idioma esperado; se não, corrige a linguagem.
    """
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
    """
    Corrige o idioma do texto utilizando o modelo para tradução.
    """
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

def advanced_forensic_analysis(text: str) -> dict:
    """
    Realiza uma análise forense aprimorada no texto fornecido, extraindo informações relevantes.
    """
    forensic_info = {}
    try:
        ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        ipv6_pattern = re.compile(r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b')
        email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        phone_pattern = re.compile(r'\+?\d[\d\s()-]{7,}\d')
        url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
        mac_pattern = re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b')
        md5_pattern = re.compile(r'\b[a-fA-F0-9]{32}\b')
        sha1_pattern = re.compile(r'\b[a-fA-F0-9]{40}\b')
        sha256_pattern = re.compile(r'\b[a-fA-F0-9]{64}\b')
        cve_pattern = re.compile(r'\bCVE-\d{4}-\d{4,7}\b')
        
        ip_addresses = ip_pattern.findall(text)
        if ip_addresses:
            forensic_info['Endereços IPv4'] = list(set(ip_addresses))
        
        ipv6_addresses = ipv6_pattern.findall(text)
        if ipv6_addresses:
            forensic_info['Endereços IPv6'] = list(set(ipv6_addresses))
        
        emails = email_pattern.findall(text)
        if emails:
            forensic_info['E-mails'] = list(set(emails))
        
        phones = phone_pattern.findall(text)
        if phones:
            forensic_info['Telefones'] = list(set(phones))
        
        urls = url_pattern.findall(text)
        if urls:
            forensic_info['URLs'] = list(set(urls))
        
        macs = mac_pattern.findall(text)
        if macs:
            forensic_info['Endereços MAC'] = list(set(macs))
        
        md5_hashes = md5_pattern.findall(text)
        if md5_hashes:
            forensic_info['Hashes MD5'] = list(set(md5_hashes))
        
        sha1_hashes = sha1_pattern.findall(text)
        if sha1_hashes:
            forensic_info['Hashes SHA1'] = list(set(sha1_hashes))
        
        sha256_hashes = sha256_pattern.findall(text)
        if sha256_hashes:
            forensic_info['Hashes SHA256'] = list(set(sha256_hashes))
        
        cve_ids = cve_pattern.findall(text)
        if cve_ids:
            forensic_info['IDs CVE'] = list(set(cve_ids))
        
    except Exception as e:
        logger.error(f"❌ Erro durante a análise forense: {e}")
    
    return forensic_info

def convert_to_degrees(value) -> float:
    """
    Converte coordenadas GPS no formato de frações para graus decimais.
    """
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
    """
    Obtém e analisa os metadados EXIF de uma imagem a partir de sua URL.
    """
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
                    meta["GPS Coordinates"] = (
                        f"{lat}, {lon} (Google Maps: https://maps.google.com/?q={lat},{lon})"
                    )
                except Exception as e:
                    meta["GPS Extraction Error"] = str(e)
        else:
            meta["info"] = "Nenhum metadado EXIF encontrado."
        return meta
    except Exception as e:
        logger.error(f"❌ Erro ao analisar metadados da imagem: {e}")
        return {"error": str(e)}

def process_investigation(target: str, sites_meta: int = 5, investigation_focus: str = "",
                          search_news: bool = False, search_leaked_data: bool = False) -> str:
    """
    Processa uma investigação online com base no alvo informado.
    """
    logger.info(f"🔍 Iniciando investigação para: {repr(target)}")
    if not target.strip():
        return "Erro: Por favor, insira um alvo para investigação."
    
    try:
        with DDGS() as ddgs:
            results = ddgs.text(keywords=target, max_results=sites_meta)
    except Exception as e:
        logger.error(f"❌ Erro na pesquisa com DDGS: {e}")
        return f"Erro na pesquisa: {e}"
    
    info_msg = f"Apenas {len(results)} sites encontrados para '{target}'.<br>" if len(results) < sites_meta else ""
    formatted_results = "<br>".join(
        f"• {res.get('title', 'Sem título')}<br>&nbsp;&nbsp;{res.get('href', 'Sem link')}<br>&nbsp;&nbsp;{res.get('body', '')}"
        for res in results
    )
    links_table = (
        "<table border='1' style='width:100%; border-collapse: collapse; text-align: left;'>"
        "<thead><tr><th>Título</th><th>Link</th></tr></thead><tbody>"
    )
    for res in results:
        title = res.get('title', 'Sem título')
        href = res.get('href', 'Sem link')
        links_table += f"<tr><td>{title}</td><td><a href='{href}' target='_blank'>{href}</a></td></tr>"
    links_table += "</tbody></table>"
    
    forensic_analysis = advanced_forensic_analysis(formatted_results)
    forensic_details = "<br>".join(f"{k}: {v}" for k, v in forensic_analysis.items() if v)
    
    investigation_prompt = f"Analise os dados obtidos sobre '{target}'"
    if investigation_focus:
        investigation_prompt += f", focando em '{investigation_focus}'"
    investigation_prompt += "<br><br>Resultados de sites:<br>" + formatted_results
    if forensic_details:
        investigation_prompt += "<br><br>Análise Forense Extraída:<br>" + forensic_details
    investigation_prompt += "<br><br>Elabore um relatório detalhado com ligações, riscos e informações relevantes."
    
    try:
        investigation_response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "Você é um perito em investigação online e análise forense digital. Seja minucioso e detalhado."},
                {"role": "user", "content": investigation_prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            stop=["</s>"]
        )
        report = investigation_response['choices'][0]['message']['content']
        final_report = info_msg + "<br>" + report + "<br><br>Links encontrados:<br>" + links_table
        return final_report
    except Exception as e:
        logger.error(f"❌ Erro na investigação: {e}")
        return f"Erro na investigação: {e}"

# Definindo o template HTML diretamente como string
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
      
      showSpinner();
      fetch('/ask', { method: 'POST', body: formData })
      .then(res => res.json())
      .then(data => {
        hideSpinner();
        appendMessage("chatWindow", "ai", data.response || "Erro: " + data.error);
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
    """
    Renderiza a página principal usando o template embutido.
    """
    return render_template_string(index_html)

@app.route('/ask', methods=['POST'])
def ask():
    """
    Endpoint que processa as requisições em três modos:
      - Chat
      - Investigação
      - Metadados
    """
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
            logger.error(f"❌ Erro no modo Investigação: {e}")
            return jsonify({'error': str(e)}), 500
    elif mode == "Metadados":
        if not user_input.strip():
            return jsonify({'response': "Erro: Por favor, insira um link de imagem."})
        try:
            meta = analyze_image_metadata(user_input)
            formatted_meta = "<br>".join(f"{k}: {v}" for k, v in meta.items())
            return jsonify({'response': formatted_meta})
        except Exception as e:
            logger.error(f"❌ Erro no modo Metadados: {e}")
            return jsonify({'error': str(e)}), 500
    else:  # Modo Chat
        try:
            response_text = generate_response(user_input, lang, style)
            return jsonify({'response': response_text})
        except Exception as e:
            logger.error(f"❌ Erro no modo Chat: {e}")
            return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

def ensure_localtunnel_installed():
    """
    Verifica se o LocalTunnel está instalado; caso não esteja, tenta instalá-lo via npm.
    Se o npm não estiver disponível, informa ao usuário para que instale manualmente.
    """
    try:
        result = subprocess.run("which lt", shell=True, capture_output=True, text=True)
        if not result.stdout.strip():
            print("LocalTunnel não encontrado. Verificando se npm está disponível...")
            npm_result = subprocess.run("which npm", shell=True, capture_output=True, text=True)
            if not npm_result.stdout.strip():
                print("npm não está disponível. Por favor, instale Node.js e npm manualmente.")
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
    """
    Obtém e exibe o tunnel password (IP público) utilizando o comando curl.
    """
    try:
        # Aguarda alguns segundos para garantir que o LocalTunnel já esteja ativo
        time.sleep(3)
        result = subprocess.run("curl https://loca.lt/mytunnelpassword", shell=True, capture_output=True, text=True)
        password = result.stdout.strip()
        if password:
            print("Tunnel Password:", password)
        else:
            print("Não foi possível obter o tunnel password.")
    except Exception as e:
        print("Erro ao obter tunnel password:", e)

if __name__ == '__main__':
    # Verifica e instala o LocalTunnel, se necessário
    ensure_localtunnel_installed()

    # Inicia o LocalTunnel para expor a porta 5000.
    def read_lt_output(process):
        for line in process.stdout:
            # Exibe a URL pública gerada pelo LocalTunnel
            if "your url is:" in line.lower():
                print("LocalTunnel URL:", line.strip())
            else:
                print("LocalTunnel:", line.strip())

    lt_command = "lt --port 5000"
    print("Iniciando LocalTunnel para expor a aplicação na porta 5000...")
    lt_process = subprocess.Popen(lt_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Inicia uma thread para exibir a saída do LocalTunnel
    lt_thread = threading.Thread(target=read_lt_output, args=(lt_process,), daemon=True)
    lt_thread.start()
    
    # Obtém e exibe o tunnel password
    get_tunnel_password()
    
    print("Executando a aplicação Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
