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
import importlib.util
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
import gradio as gr  # Integração com Gradio.live

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
  <!-- Importação da fonte Inter do Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  <style>
    /* Reset e estilo base */
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: radial-gradient(circle, #0f0f0f, #1c1c1c);
      color: #e0e0e0;
      font-family: 'Inter', sans-serif;
      display: flex;
      min-height: 100vh;
      transition: background 0.5s, color 0.5s;
    }
    header {
      background: linear-gradient(90deg, #141414, #202020);
      color: #fff;
      padding: 25px;
      text-align: center;
      font-size: 28px;
      font-weight: 600;
      border-bottom: 2px solid #333;
      box-shadow: 0 4px 8px rgba(0,0,0,0.5);
      transition: background 0.3s;
    }
    .nav-tabs {
      display: flex;
      background-color: #151515;
      border-bottom: 2px solid #333;
    }
    .nav-tabs button {
      flex: 1;
      padding: 18px 20px;
      background: none;
      border: none;
      color: #a0a0a0;
      font-size: 17px;
      cursor: pointer;
      transition: background 0.3s, color 0.3s;
    }
    .nav-tabs button:hover {
      background-color: #1d1d1d;
    }
    .nav-tabs button.active {
      background-color: #252525;
      color: #00ffe0;
      border-bottom: 3px solid #00ffe0;
    }
    .main-container {
      display: flex;
      flex: 1;
      max-width: 1200px;
      margin: 30px auto;
      padding: 20px;
      gap: 20px;
    }
    .chat-container {
      flex: 3;
      display: flex;
      flex-direction: column;
    }
    .chat-window {
      background-color: #1a1a1a;
      border-radius: 10px;
      padding: 25px;
      height: 500px;
      overflow-y: auto;
      margin-bottom: 20px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.7);
      position: relative;
      transition: background 0.3s, box-shadow 0.3s;
    }
    .message {
      margin-bottom: 20px;
      display: flex;
      animation: slideIn 0.5s ease-out;
    }
    .message.user {
      justify-content: flex-end;
    }
    .message.ai {
      justify-content: flex-start;
    }
    .bubble {
      padding: 12px 18px;
      border-radius: 15px;
      max-width: 75%;
      word-wrap: break-word;
      font-size: 17px;
      line-height: 1.6;
      transition: background 0.3s, transform 0.2s;
    }
    .message.user .bubble {
      background: linear-gradient(90deg, #00ffe0, #00aaff);
      color: #0a0a0a;
      border-radius: 15px 15px 0 15px;
      transform: translateX(10px);
    }
    .message.ai .bubble {
      background-color: #2a2a2a;
      color: #e0e0e0;
      border-radius: 15px 15px 15px 0;
      transform: translateX(-10px);
    }
    .input-area {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .input-area input[type="text"] {
      flex: 1;
      padding: 14px;
      border: 1px solid #333;
      border-radius: 6px;
      background-color: #141414;
      color: #e0e0e0;
      font-size: 16px;
      transition: border-color 0.3s, box-shadow 0.3s;
    }
    .input-area input[type="text"]:focus {
      border-color: #00ffe0;
      outline: none;
      box-shadow: 0 0 8px rgba(0,255,224,0.5);
    }
    .input-area button {
      padding: 14px 24px;
      border: none;
      background-color: #00aaff;
      color: #fff;
      border-radius: 6px;
      cursor: pointer;
      font-size: 16px;
      transition: background 0.3s, transform 0.2s;
    }
    .input-area button:hover {
      background-color: #0088cc;
      transform: scale(1.05);
    }
    .form-options {
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      font-size: 14px;
      color: #b0b0b0;
    }
    .form-options label {
      margin-right: 8px;
    }
    /* Campo para Temperatura da IA */
    .form-options input[type="number"] {
      width: 80px;
      padding: 4px;
      border: 1px solid #333;
      border-radius: 4px;
      background-color: #141414;
      color: #e0e0e0;
      font-size: 14px;
      transition: border-color 0.3s;
    }
    /* Novas opções para investigação */
    #investigationOptions { display: none; }
    /* Sidebar para anotações */
    .sidebar {
      flex: 1;
      background-color: #1a1a1a;
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.7);
      height: 550px;
      overflow-y: auto;
      position: relative;
      transition: background 0.3s, box-shadow 0.3s;
    }
    .sidebar h3 {
      margin-bottom: 12px;
      font-size: 20px;
      color: #00ffe0;
      text-align: center;
    }
    .sidebar textarea {
      width: 100%;
      height: calc(100% - 50px);
      background-color: #141414;
      border: 1px solid #333;
      border-radius: 6px;
      color: #e0e0e0;
      padding: 10px;
      font-size: 15px;
      resize: none;
      transition: border-color 0.3s;
    }
    .sidebar textarea:focus {
      border-color: #00ffe0;
      outline: none;
    }
    /* Spinner de carregamento */
    #loadingSpinner {
      display: none;
      margin: 15px auto;
      border: 6px solid #333;
      border-top: 6px solid #00ffe0;
      border-radius: 50%;
      width: 36px;
      height: 36px;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(15px); }
      to { opacity: 1; transform: translateY(0); }
    }
    footer {
      background-color: #111827;
      color: #a0a0a0;
      text-align: center;
      padding: 15px;
      font-size: 12px;
      border-top: 1px solid #333;
      transition: background 0.3s;
    }
    /* Modal de configurações */
    #configModal {
      display: none;
      position: fixed;
      top: 0; left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.85);
      justify-content: center;
      align-items: center;
      z-index: 1000;
      animation: fadeInModal 0.5s ease;
    }
    @keyframes fadeInModal {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    #configModal .modal-content {
      background: #121212;
      padding: 25px;
      border-radius: 8px;
      max-width: 500px;
      width: 90%;
      color: #e0e0e0;
      box-shadow: 0 6px 18px rgba(0,0,0,0.7);
      animation: slideDown 0.5s ease-out;
    }
    @keyframes slideDown {
      from { transform: translateY(-20px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    #configModal .modal-content h2 {
      margin-bottom: 18px;
      text-align: center;
      font-size: 22px;
    }
    #configModal .modal-content form > div {
      margin-bottom: 12px;
    }
    #configModal label {
      display: block;
      margin-bottom: 4px;
    }
    #configModal input[type="number"],
    #configModal input[type="color"] {
      width: 100%;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background-color: #1a1a1a;
      color: #e0e0e0;
    }
    /* Novas configurações para área de anotações */
    #configModal .modal-content form > div.notes-config {
      margin-bottom: 12px;
    }
    #configModal button {
      padding: 10px 16px;
      border: none;
      background-color: #00aaff;
      color: #fff;
      border-radius: 4px;
      cursor: pointer;
      margin-right: 10px;
      transition: background 0.3s, transform 0.2s;
    }
    #configModal button:hover {
      background-color: #0088cc;
      transform: scale(1.05);
    }
    /* Botão de configurações flutuante */
    #configToggleButton {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background-color: #00aaff;
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 55px;
      height: 55px;
      cursor: pointer;
      font-size: 26px;
      z-index: 1001;
      box-shadow: 0 4px 12px rgba(0,0,0,0.6);
      transition: transform 0.3s;
    }
    #configToggleButton:hover {
      transform: rotate(45deg) scale(1.1);
    }
    /* Botão de limpar chat */
    #clearChat {
      margin-top: 12px;
      padding: 12px 22px;
      background-color: #ef4444;
      color: #fff;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 15px;
      transition: background 0.3s, transform 0.2s;
    }
    #clearChat:hover {
      background-color: #dc2626;
      transform: scale(1.05);
    }
  </style>
  
  <!-- Estilo personalizado atualizado via configurações -->
  <style id="customStyles"></style>
  
  <!-- Estilos extras de interatividade e animação -->
  <style id="extraAnimations">
    /* Particle background canvas */
    #particleCanvas {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: -2;
    }
    /* Animação dinâmica de fundo */
    @keyframes backgroundShift {
      0% { background: radial-gradient(circle, #0f0f0f, #1c1c1c); }
      50% { background: radial-gradient(circle, #1c1c1c, #0f0f0f); }
      100% { background: radial-gradient(circle, #0f0f0f, #1c1c1c); }
    }
    body {
      animation: backgroundShift 30s infinite alternate;
    }
    /* Efeito de ripple para botões */
    .button-ripple {
      position: relative;
      overflow: hidden;
    }
    .button-ripple::after {
      content: "";
      position: absolute;
      background: rgba(255,255,255,0.4);
      border-radius: 50%;
      transform: scale(0);
      animation: rippleAnimation 0.6s linear;
      pointer-events: none;
    }
    @keyframes rippleAnimation {
      to {
        transform: scale(4);
        opacity: 0;
      }
    }
    /* Animação de pulsar ao hover nas bolhas */
    .message.user .bubble:hover, .message.ai .bubble:hover {
      animation: pulse 1s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.05); }
      100% { transform: scale(1); }
    }
    /* Animação de tremor (shake) ao clicar nas mensagens */
    @keyframes shake {
      0% { transform: translate(0); }
      25% { transform: translate(5px, 0); }
      50% { transform: translate(-5px, 0); }
      75% { transform: translate(5px, 0); }
      100% { transform: translate(0); }
    }
    .message:hover {
      cursor: pointer;
    }
    /* Estilo para Toast Notification */
    #toast {
      position: fixed;
      bottom: 30px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0,0,0,0.8);
      color: #fff;
      padding: 12px 24px;
      border-radius: 5px;
      opacity: 0;
      transition: opacity 0.5s ease;
      z-index: 1100;
    }
    #toast.show {
      opacity: 1;
    }
  </style>
</head>
<body>
  <!-- Canvas para partículas animadas -->
  <canvas id="particleCanvas"></canvas>
  
  <header>Cyber Assistant v4.0</header>
  <div class="nav-tabs">
    <button class="tab-button active" data-tab="chatTab">Chat</button>
  </div>
  <div class="main-container">
    <div class="chat-container">
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
            <label for="temperature">Temperatura da IA:</label>
            <input type="number" id="temperature" name="temperature" min="0" max="1" step="0.1" value="0.7">
            <!-- Nova opção para Velocidade -->
            <label for="velocidade">Velocidade:</label>
            <select id="velocidade" name="velocidade">
              <option value="Detalhada">Detalhada</option>
              <option value="Rápida">Rápida</option>
            </select>
            <label>
              <input type="checkbox" id="streaming" name="streaming"> Ativar Streaming
            </label>
            <!-- Novos campos para configuração de GPU/CPU -->
            <label for="gpu_layers">Camadas GPU:</label>
            <input type="number" id="gpu_layers" name="gpu_layers" placeholder="Automático">
            <label for="n_batch">Tamanho do Lote:</label>
            <input type="number" id="n_batch" name="n_batch" placeholder="Automático">
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
    <!-- Sidebar para anotações com handle para arrastar -->
    <div class="sidebar" id="sidebar">
      <div id="sidebarHandle">Arraste aqui</div>
      <h3>Anotações</h3>
      <textarea id="notesArea" placeholder="Anote informações importantes aqui..."></textarea>
    </div>
  </div>
  <footer>© 2025 Cyber Assistant</footer>
  
  <!-- Botão flutuante para abrir as configurações -->
  <button id="configToggleButton" class="button-ripple">⚙️</button>
  
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
        <!-- Novas configurações para a área de anotações -->
        <div class="notes-config">
          <label for="notesWidth">Largura da área de anotações (px):</label>
          <input type="number" id="notesWidth" name="notesWidth" value="300" min="100" max="800">
        </div>
        <div class="notes-config">
          <label for="notesHeight">Altura da área de anotações (px):</label>
          <input type="number" id="notesHeight" name="notesHeight" value="550" min="200" max="800">
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
  
  <!-- Modal de confirmação para limpar o chat -->
  <div id="clearChatModal">
    <div class="modal-content">
      <h3>Tem certeza que deseja limpar o chat?</h3>
      <button id="confirmClear">Sim</button>
      <button id="cancelClear">Não</button>
    </div>
  </div>
  
  <!-- Toast Notification -->
  <div id="toast"></div>
  
  <script>
    // Alterna exibição das opções de investigação conforme o modo selecionado
    document.getElementById("mode").addEventListener("change", function() {
      const mode = this.value;
      const invOptions = document.getElementById("investigationOptions");
      invOptions.style.display = mode === "Investigação" ? "flex" : "none";
      showToast("Modo alterado para: " + mode);
      // Animação extra na janela de chat
      const chatWindow = document.getElementById("chatWindow");
      chatWindow.style.boxShadow = "0 0 20px #00ffe0";
      setTimeout(() => chatWindow.style.boxShadow = "0 6px 20px rgba(0,0,0,0.7)", 500);
    });
    
    // Função para adicionar mensagem à janela de chat com interatividade
    function appendMessage(windowId, sender, message) {
      const windowElement = document.getElementById(windowId);
      const messageDiv = document.createElement('div');
      messageDiv.className = `message ${sender}`;
      const bubbleDiv = document.createElement('div');
      bubbleDiv.className = 'bubble';
      bubbleDiv.innerHTML = message;
      messageDiv.appendChild(bubbleDiv);
      // Efeito de shake ao clicar na mensagem
      messageDiv.addEventListener('click', () => {
        messageDiv.style.animation = 'shake 0.5s';
        setTimeout(() => messageDiv.style.animation = '', 500);
      });
      windowElement.appendChild(messageDiv);
      windowElement.scrollTop = windowElement.scrollHeight;
    }
    
    // Função para adicionar mensagem de tempo de resposta
    function appendTimer(time) {
      appendMessage("chatWindow", "ai", `<em>Tempo de resposta: ${time.toFixed(2)} segundos</em>`);
    }
    
    // Função para exibir o Toast Notification
    function showToast(message) {
      const toast = document.getElementById("toast");
      toast.innerText = message;
      toast.classList.add("show");
      setTimeout(() => {
        toast.classList.remove("show");
      }, 3000);
    }
    
    // Função para limpar o chat com animação de fade-out
    function clearChat() {
      const chatWindow = document.getElementById("chatWindow");
      chatWindow.style.transition = "opacity 0.5s";
      chatWindow.style.opacity = "0";
      setTimeout(() => {
        chatWindow.innerHTML = "";
        chatWindow.style.opacity = "1";
      }, 500);
    }
    
    // Exibe o spinner de carregamento
    function showSpinner() {
      document.getElementById("loadingSpinner").style.display = "block";
    }
    
    // Oculta o spinner de carregamento
    function hideSpinner() {
      document.getElementById("loadingSpinner").style.display = "none";
    }
    
    // Envio do formulário de chat com suporte a streaming e novos parâmetros de GPU/CPU
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
      formData.append('temperature', document.getElementById("temperature").value);
      formData.append('velocidade', document.getElementById("velocidade").value);
      formData.append('sites_meta', document.getElementById("sites_meta").value);
      formData.append('investigation_focus', document.getElementById("investigation_focus").value);
      formData.append('search_news', document.getElementById("search_news").checked);
      formData.append('search_leaked_data', document.getElementById("search_leaked_data").checked);
      formData.append('gpu_layers', document.getElementById("gpu_layers").value);
      formData.append('n_batch', document.getElementById("n_batch").value);
      
      const streaming = document.getElementById("streaming").checked;
      showSpinner();
      
      const startTime = Date.now();
      const url = streaming ? '/ask?stream=true' : '/ask';
      
      fetch(url, { method: 'POST', body: formData })
      .then(response => {
        if (streaming) {
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
    
    // Configurações: exibição do modal de configurações
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
      const inputFontSize = document.getElementById("inputFontSize").value;
      const inputWidth = document.getElementById("inputWidth").value;
      const chatWindowHeight = document.getElementById("chatWindowHeight").value;
      const chatWindowWidth = document.getElementById("chatWindowWidth").value;
      const notesWidth = document.getElementById("notesWidth").value;
      const notesHeight = document.getElementById("notesHeight").value;
      const bodyBgColor = document.getElementById("bodyBgColor").value;
      const chatBgColor = document.getElementById("chatBgColor").value;
      const chatTextColor = document.getElementById("chatTextColor").value;
      const userBubbleColor = document.getElementById("userBubbleColor").value;
      const aiBubbleColor = document.getElementById("aiBubbleColor").value;
      
      const customStyles = document.getElementById("customStyles");
      customStyles.innerHTML = `
        #chatInput { font-size: ${inputFontSize}px; width: ${inputWidth}px; }
        #chatWindow { height: ${chatWindowHeight}px; width: ${chatWindowWidth}px; background-color: ${chatBgColor}; color: ${chatTextColor}; }
        .sidebar { width: ${notesWidth}px; height: ${notesHeight}px; }
        .message.user .bubble { background-color: ${userBubbleColor}; }
        .message.ai .bubble { background-color: ${aiBubbleColor}; }
        body { background-color: ${bodyBgColor}; }
      `;
      configModal.style.display = "none";
      showToast("Configurações salvas!");
    });
    
    // Modal de confirmação para limpar o chat
    const clearChatModal = document.getElementById("clearChatModal");
    document.getElementById("clearChat").addEventListener("click", () => {
      clearChatModal.style.display = "flex";
    });
    document.getElementById("cancelClear").addEventListener("click", () => {
      clearChatModal.style.display = "none";
    });
    document.getElementById("confirmClear").addEventListener("click", () => {
      clearChat();
      clearChatModal.style.display = "none";
      showToast("Chat limpo!");
    });
    
    // Ativa efeito ripple em todos os botões
    document.querySelectorAll('button').forEach(btn => {
      btn.classList.add('button-ripple');
    });
    
    // Função para tornar a sidebar arrastável
    (function() {
      const sidebar = document.getElementById("sidebar");
      const handle = document.getElementById("sidebarHandle");
      let isDragging = false, offsetX = 0, offsetY = 0;
      
      handle.addEventListener("mousedown", (e) => {
        isDragging = true;
        sidebar.style.position = "absolute";
        offsetX = e.clientX - sidebar.getBoundingClientRect().left;
        offsetY = e.clientY - sidebar.getBoundingClientRect().top;
      });
      
      document.addEventListener("mousemove", (e) => {
        if (isDragging) {
          sidebar.style.left = (e.clientX - offsetX) + "px";
          sidebar.style.top = (e.clientY - offsetY) + "px";
        }
      });
      
      document.addEventListener("mouseup", () => {
        isDragging = false;
      });
    })();
    
    // Duplo clique na janela de chat para alternar para modo full-screen
    document.getElementById("chatWindow").addEventListener("dblclick", function() {
      this.classList.toggle("fullscreen");
      showToast(this.classList.contains("fullscreen") ? "Chat em tela cheia" : "Chat normalizado");
    });
    
    // Script para partículas animadas no fundo
    (function(){
      const canvas = document.getElementById('particleCanvas');
      const ctx = canvas.getContext('2d');
      let particles = [];
      const particleCount = 100;
      
      function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }
      
      window.addEventListener('resize', resizeCanvas);
      resizeCanvas();
      
      class Particle {
        constructor(){
          this.x = Math.random() * canvas.width;
          this.y = Math.random() * canvas.height;
          this.radius = Math.random() * 2 + 1;
          this.speedX = (Math.random() - 0.5) * 1;
          this.speedY = (Math.random() - 0.5) * 1;
          this.alpha = Math.random() * 0.5 + 0.5;
        }
        
        update(){
          this.x += this.speedX;
          this.y += this.speedY;
          if(this.x < 0 || this.x > canvas.width) this.speedX = -this.speedX;
          if(this.y < 0 || this.y > canvas.height) this.speedY = -this.speedY;
        }
        
        draw(){
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,255,255,${this.alpha})`;
          ctx.fill();
        }
      }
      
      function initParticles(){
        particles = [];
        for(let i = 0; i < particleCount; i++){
          particles.push(new Particle());
        }
      }
      
      function animateParticles(){
        ctx.clearRect(0,0,canvas.width, canvas.height);
        particles.forEach(p => {
          p.update();
          p.draw();
        });
        requestAnimationFrame(animateParticles);
      }
      
      initParticles();
      animateParticles();
    })();
    
    /* -- NOVA INTERATIVIDADE: COELHINHO COM VIDA PRÓPRIA -- */
    (function() {
      // Cria UM coelhinho e posiciona aleatoriamente
      const rabbit = document.createElement("div");
      rabbit.className = "rabbit";
      rabbit.innerText = "🐰";
      rabbit.style.left = Math.random() * (window.innerWidth - 50) + "px";
      rabbit.style.top = Math.random() * (window.innerHeight - 50) + "px";
      document.body.appendChild(rabbit);
      
      let carrot = null; // referência à cenoura
      
      // Função para criar uma cenoura em local aleatório
      function spawnCarrot() {
        if (carrot) {
          carrot.remove();
        }
        carrot = document.createElement("div");
        carrot.className = "carrot";
        carrot.innerText = "🥕";
        carrot.style.left = Math.random() * (window.innerWidth - 50) + "px";
        carrot.style.top = Math.random() * (window.innerHeight - 50) + "px";
        document.body.appendChild(carrot);
      }
      
      // Função para mover o coelhinho em direção à cenoura
      function moveRabbitToCarrot() {
        if (!carrot) return;
        const rabbitRect = rabbit.getBoundingClientRect();
        const carrotRect = carrot.getBoundingClientRect();
        const rx = rabbitRect.left;
        const ry = rabbitRect.top;
        const cx = carrotRect.left;
        const cy = carrotRect.top;
        const dx = cx - rx;
        const dy = cy - ry;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 10) {
          // Coelhinho alcançou a cenoura; "comer" (remover) a cenoura
          carrot.remove();
          carrot = null;
          showToast("O coelhinho comeu a cenoura!");
          return;
        }
        // Move o coelhinho em direção à cenoura (velocidade de 2px por frame)
        const speed = 2;
        rabbit.style.left = (rx + (dx / distance) * speed) + "px";
        rabbit.style.top = (ry + (dy / distance) * speed) + "px";
      }
      
      // Função para fazer o coelhinho vaguear aleatoriamente se não houver cenoura
      function wanderRabbit() {
        if (carrot) return; // se a cenoura existe, o coelhinho vai até ela
        const rect = rabbit.getBoundingClientRect();
        let newX = rect.left + (Math.random() - 0.5) * 10;
        let newY = rect.top + (Math.random() - 0.5) * 10;
        newX = Math.max(0, Math.min(window.innerWidth - 50, newX));
        newY = Math.max(0, Math.min(window.innerHeight - 50, newY));
        rabbit.style.left = newX + "px";
        rabbit.style.top = newY + "px";
      }
      
      // Intervalo para spawn de cenoura a cada 10 minutos (600000 ms)
      setInterval(spawnCarrot, 600000);
      
      // Loop de animação para o coelhinho
      function animateRabbit() {
        if (carrot) {
          moveRabbitToCarrot();
        } else {
          wanderRabbit();
        }
        requestAnimationFrame(animateRabbit);
      }
      animateRabbit();
      
      // Permite arrastar o coelhinho manualmente
      rabbit.addEventListener("mousedown", function(e) {
        rabbit.classList.add("dragging");
        const offsetX = e.clientX - rabbit.getBoundingClientRect().left;
        const offsetY = e.clientY - rabbit.getBoundingClientRect().top;
        function onMouseMove(e) {
          rabbit.style.left = (e.clientX - offsetX) + "px";
          rabbit.style.top = (e.clientY - offsetY) + "px";
        }
        function onMouseUp() {
          rabbit.classList.remove("dragging");
          document.removeEventListener("mousemove", onMouseMove);
          document.removeEventListener("mouseup", onMouseUp);
        }
        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("mouseup", onMouseUp);
      });
    })();
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

# ===== COMPILED_REGEX_PATTERNS =====
COMPILED_REGEX_PATTERNS = {
    'ip': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    'ipv6': re.compile(r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b'),
    'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    'phone': re.compile(r'\+?\d[\d\s()-]{7,}\d'),
    'url': re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'),
    'mac': re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b'),
    # ... outros padrões conforme necessário ...
    'domain': re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b')
}

# ===== Carregamento de Plugins (Módulos Externos) =====
def load_plugins(plugin_dir="plugins"):
    plugins = {}
    if os.path.isdir(plugin_dir):
        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py"):
                module_name = filename[:-3]
                file_path = os.path.join(plugin_dir, filename)
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    plugins[module_name] = module
                    logger.info(f"Plugin carregado: {module_name}")
                except Exception as e:
                    logger.error(f"Erro ao carregar plugin {module_name}: {e}")
    return plugins

plugins = load_plugins()

# ===== Gerenciamento de GPU/CPU e Paralelização =====
model_lock = threading.Lock()  # Para acesso thread-safe ao modelo

def load_model(custom_gpu_layers=None, custom_n_batch=None) -> Llama:
    model_path = os.path.join(DEFAULT_LOCAL_MODEL_DIR, DEFAULT_MODEL_FILE)
    if not os.path.exists(model_path):
        hf_hub_download(
            repo_id=DEFAULT_MODEL_NAME,
            filename=DEFAULT_MODEL_FILE,
            local_dir=DEFAULT_LOCAL_MODEL_DIR,
            resume_download=True
        )
    try:
        # Detecta GPU ou utiliza valores customizados, se fornecidos
        n_gpu_layers = custom_gpu_layers if custom_gpu_layers is not None else 15
        n_batch = custom_n_batch if custom_n_batch is not None else 512
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus and custom_gpu_layers is None and custom_n_batch is None:
                n_gpu_layers = -1
                n_batch = 1024
                logger.info("GPU detectada. Utilizando todas as camadas na GPU (n_gpu_layers=-1) e n_batch=1024.")
            else:
                logger.info("Nenhuma GPU detectada ou parâmetros customizados fornecidos. Usando configuração otimizada para CPU ou customizada.")
        except Exception as gpu_error:
            logger.warning(f"Erro na detecção da GPU: {gpu_error}. Configuração para CPU será utilizada.")
        model_instance = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=psutil.cpu_count(logical=True),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch
        )
        logger.info(f"🤖 Modelo Neural Carregado com n_gpu_layers={n_gpu_layers}, n_batch={n_batch} e n_threads={psutil.cpu_count(logical=True)}")
        return model_instance
    except Exception as e:
        logger.error(f"❌ Erro na Inicialização do Modelo: {e}")
        raise e

# Inicializa o modelo globalmente
model = load_model()

def update_model_config(custom_gpu_layers, custom_n_batch):
    global model
    with model_lock:
        logger.info("Atualizando configuração do modelo...")
        model = load_model(custom_gpu_layers, custom_n_batch)
        logger.info("Configuração do modelo atualizada.")

# ===== Função Autocorretora =====
def autocorrect_text(text: str, lang: str) -> str:
    prompt = f"Corrija os erros de digitação e melhore a gramática do seguinte texto, mantendo o mesmo significado:\n\n{text}"
    try:
        with model_lock:
            correction_response = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                stop=["</s>"]
            )
        corrected_text = correction_response['choices'][0]['message']['content'].strip()
        return corrected_text
    except Exception as e:
        logger.error(f"Erro na autocorreção: {e}")
        return text

# ===== Funções para Geração de Resposta e Validação de Idioma =====
def build_messages(query: str, lang_config: dict, style: str, custom_temperature: float = None) -> tuple[list, float]:
    if style == "Técnico":
        system_instruction = f"{lang_config['instruction']}. Seja detalhado e técnico."
        default_temp = 0.7
    else:
        system_instruction = f"{lang_config['instruction']}. Responda de forma livre e criativa."
        default_temp = 0.9
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": query}
    ]
    temperature = custom_temperature if custom_temperature is not None else default_temp
    return messages, temperature

def generate_response(query: str, lang: str, style: str, custom_temperature: float = None, fast_mode: bool = False) -> str:
    corrected_query = autocorrect_text(query, lang)
    start_time = time.time()
    cached_text = get_cached_response(corrected_query, lang, style)
    if cached_text:
        logger.info(f"✅ Resposta obtida do cache em {time.time() - start_time:.2f}s")
        return cached_text
    lang_config = LANGUAGE_MAP.get(lang, LANGUAGE_MAP['Português'])
    messages, temperature = build_messages(corrected_query, lang_config, style, custom_temperature)
    max_tokens = 400 if fast_mode else 800
    try:
        with model_lock:
            response = model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["</s>"]
            )
        raw_response = response['choices'][0]['message']['content']
        final_response = validate_language(raw_response, lang_config)
        logger.info(f"✅ Resposta gerada em {time.time() - start_time:.2f}s")
        set_cached_response(corrected_query, lang, style, final_response)
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
        with model_lock:
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
    try:
        ipv4_pattern = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')
        ipv6_pattern = re.compile(r'^[A-Fa-f0-9:]+$')
        if ipv4_pattern.match(target) or ipv6_pattern.match(target):
            return {'target': target, 'ip': target, 'method': 'Já é um IP válido'}
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
    temp_input = request.form.get('temperature', None)
    custom_temperature = float(temp_input) if temp_input is not None and temp_input != "" else None
    fast_mode = request.form.get('fast_mode', 'false').lower() == 'true'
    
    if mode == "Investigação":
        if not user_input.strip():
            return jsonify({'response': "Erro: Por favor, insira um alvo para investigação."})
        try:
            sites_meta = int(request.form.get('sites_meta', 5))
            investigation_focus = request.form.get('investigation_focus', '')
            search_news = request.form.get('search_news', 'false').lower() == 'true'
            search_leaked_data = request.form.get('search_leaked_data', 'false').lower() == 'true'
            report, links_table = process_investigation(user_input, sites_meta, investigation_focus, search_news, search_leaked_data, custom_temperature, lang, fast_mode)
            final_report = report + "<br><br>Links encontrados:<br>" + links_table
            return jsonify({'response': final_report})
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
            response_text = generate_response(user_input, lang, style, custom_temperature, fast_mode)
            return jsonify({'response': response_text})
        except Exception as e:
            logger.error(f"Erro no modo Chat: {e}")
            return jsonify({'error': str(e)}), 500

# ===== Função para Streaming de Respostas (para feedback em tempo real) =====
def streaming_response(text: str, chunk_size: int = 200):
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]
        time.sleep(0.1)

# ===== Funções para Análise Forense e Processamento de Texto =====
def advanced_forensic_analysis(text: str) -> dict:
    forensic_info = {}
    try:
        for key, pattern in COMPILED_REGEX_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                forensic_info[key] = list(set(matches))
    except Exception as e:
        logger.error(f"❌ Erro durante a análise forense: {e}")
    return forensic_info

# --- SISTEMA DE METADADOS ---
def get_decimal_from_dms(dms, ref):
    try:
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except Exception as e:
        raise ValueError("Erro ao converter coordenadas DMS para decimal: " + str(e))

def analyze_image_metadata(url: str) -> dict:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_data = response.content
        image = Image.open(io.BytesIO(image_data))
        meta = {}
        exif_data = image.getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                meta[tag] = value
            if "GPSInfo" in meta:
                gps_info = meta["GPSInfo"]
                try:
                    lat = get_decimal_from_dms(gps_info.get(2), gps_info.get(1))
                    lon = get_decimal_from_dms(gps_info.get(4), gps_info.get(3))
                    meta["GPS Coordinates"] = f"{lat}, {lon} (Google Maps: https://maps.google.com/?q={lat},{lon})"
                except Exception as e:
                    meta["GPS Extraction Error"] = str(e)
        else:
            meta["info"] = "Nenhum metadado EXIF encontrado."
        return meta
    except Exception as e:
        logger.error(f"❌ Erro ao analisar metadados da imagem: {e}")
        return {"error": str(e)}
# --- Fim do sistema de metadados ---

# ===== Funcionalidades de Investigação Online =====
def perform_search(query: str, search_type: str, max_results: int) -> list:
    try:
        ddgs = DDGS()  # Cria uma nova instância a cada chamada
        if search_type == 'web':
            results = list(ddgs.text(keywords=query, max_results=max_results))
        elif search_type == 'news':
            results = list(ddgs.news(keywords=query, max_results=max_results))
        elif search_type == 'leaked':
            results = list(ddgs.text(keywords=f"{query} leaked", max_results=max_results))
        else:
            results = []
        return results
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
        "<thead><tr><th>Nº</th><th>Título</th><th>Link</th></tr></thead><tbody>"
    )
    for i, res in enumerate(results, 1):
        title = res.get('title', 'Sem título')
        href = res.get('href', 'Sem link')
        links_table += f"<tr><td>{i}</td><td>{title}</td><td><a href='{href}' target='_blank'>{href}</a></td></tr>"
    links_table += "</tbody></table>"
    return formatted_text, links_table, info_message

def process_investigation(target: str, sites_meta: int = 5, investigation_focus: str = "",
                          search_news: bool = False, search_leaked_data: bool = False, custom_temperature: float = None,
                          lang: str = "Português", fast_mode: bool = False) -> tuple:
    logger.info(f"🔍 Iniciando investigação para: {repr(target)}")
    if not target.strip():
        return "Erro: Por favor, insira um alvo para investigação.", ""
    
    corrected_target = autocorrect_text(target, lang)
    
    search_tasks = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        search_tasks['web'] = executor.submit(perform_search, corrected_target, 'web', sites_meta)
        if search_news:
            search_tasks['news'] = executor.submit(perform_search, corrected_target, 'news', sites_meta)
        if search_leaked_data:
            search_tasks['leaked'] = executor.submit(perform_search, corrected_target, 'leaked', sites_meta)
    
    results_web = search_tasks['web'].result() if 'web' in search_tasks else []
    results_news = search_tasks['news'].result() if 'news' in search_tasks else []
    results_leaked = search_tasks['leaked'].result() if 'leaked' in search_tasks else []
    
    formatted_web, links_web, _ = format_search_results(results_web, "Sites")
    formatted_news, links_news, _ = ( "", "", "" ) if not results_news else format_search_results(results_news, "Notícias")
    formatted_leaked, links_leaked, _ = ( "", "", "" ) if not results_leaked else format_search_results(results_leaked, "Dados Vazados")
    
    combined_results_text = ""
    if formatted_web:
        combined_results_text += "<br><br>Resultados de Sites:<br>" + formatted_web
    if formatted_news:
        combined_results_text += "<br><br>Notícias:<br>" + formatted_news
    if formatted_leaked:
        combined_results_text += "<br><br>Dados Vazados:<br>" + formatted_leaked
    
    forensic_analysis = advanced_forensic_analysis(combined_results_text)
    forensic_details = "<br>".join(f"{k}: {v}" for k, v in forensic_analysis.items() if v)
    
    investigation_prompt = f"Analise os dados obtidos sobre '{corrected_target}'"
    if investigation_focus:
        investigation_prompt += f", focando em '{investigation_focus}'"
    investigation_prompt += "<br>" + combined_results_text
    if forensic_details:
        investigation_prompt += "<br><br>Análise Forense Extraída:<br>" + forensic_details
    investigation_prompt += "<br><br>Elabore um relatório detalhado com ligações, riscos e informações relevantes."
    
    temp = custom_temperature if custom_temperature is not None else 0.7
    max_tokens = 500 if fast_mode else 1000
    logger.info(f"Utilizando temperatura {temp} na investigação e max_tokens={max_tokens}.")
    
    try:
        with model_lock:
            investigation_response = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Você é um perito policial e forense digital, experiente em métodos policiais de investigação. Utilize técnicas de análise de evidências, protocolos forenses e investigação digital para identificar padrões, rastrear conexões e coletar evidências relevantes. Seja minucioso, preciso e detalhado."},
                    {"role": "user", "content": investigation_prompt}
                ],
                temperature=temp,
                max_tokens=max_tokens,
                stop=["</s>"]
            )
        report = investigation_response['choices'][0]['message']['content']
        links_combined = links_web + (links_news if links_news else "") + (links_leaked if links_leaked else "")
        return report, links_combined
    except Exception as e:
        logger.error(f"❌ Erro na investigação: {e}")
        return f"Erro na investigação: {e}", ""

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

# ===== Integração com Gradio.live (Interface Aprimorada e Reativa) =====
def gradio_interface(query, mode, language, style, investigation_focus, num_sites, search_news, search_leaked_data, temperature, velocidade, gpu_layers, n_batch):
    # Atualiza configurações de GPU/CPU se fornecidas
    if gpu_layers != "" and n_batch != "":
        try:
            update_model_config(int(gpu_layers), int(n_batch))
        except Exception as e:
            logger.error(f"Erro ao atualizar configuração do modelo: {e}")

    custom_temperature = float(temperature) if temperature != "" else None
    fast_mode = True if velocidade == "Rápida" else False
    
    # Se o modo for "Investigação", retornamos feedback em tempo real via generator
    if mode == "Investigação":
        yield "⏳ Iniciando investigação...", ""
        sites_meta = int(num_sites)
        report, links_table = process_investigation(query, sites_meta, investigation_focus, search_news, search_leaked_data, custom_temperature, language, fast_mode)
        yield report, links_table
    elif mode == "Chat":
        result = generate_response(query, language, style, custom_temperature, fast_mode)
        yield result, ""
    elif mode == "Metadados":
        meta = analyze_image_metadata(query)
        formatted_meta = "<br>".join(f"{k}: {v}" for k, v in meta.items())
        yield formatted_meta, ""
    else:
        yield "Modo não suportado.", ""

def build_gradio_interface():
    with gr.Blocks(title="Interface de IA - Chat & Investigação") as demo:
        gr.Markdown("# Interface de IA - Chat & Investigação")
        gr.Markdown("### Insira os parâmetros para interagir com o sistema")
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(label="Pergunta/Alvo", placeholder="Digite sua pergunta (Chat), alvo (Investigação) ou URL de imagem (Metadados)...", lines=3)
                mode_input = gr.Radio(["Chat", "Investigação", "Metadados"], label="Modo", value="Chat", interactive=True)
                language_input = gr.Radio(["Português", "English", "Español", "Français", "Deutsch"], label="Idioma", value="Português", interactive=True)
                style_input = gr.Radio(["Técnico", "Livre"], label="Estilo", value="Técnico", interactive=True)
                investigation_focus = gr.Textbox(label="Foco da Investigação (opcional)", placeholder="Ex: vulnerabilidades, evidências, etc.", lines=1)
                num_sites = gr.Number(label="Número de Sites", value=5, precision=0)
                search_news = gr.Checkbox(label="Pesquisar Notícias", value=False)
                search_leaked_data = gr.Checkbox(label="Pesquisar Dados Vazados", value=False)
                temperature_input = gr.Slider(label="Temperatura da IA", minimum=0.0, maximum=1.0, step=0.1, value=0.7)
                velocidade_input = gr.Radio(["Rápida", "Detalhada"], label="Velocidade (menos detalhes / mais detalhes)", value="Detalhada", interactive=True)
                # Novos parâmetros para configuração de GPU/CPU
                gpu_layers_input = gr.Textbox(label="Camadas GPU (n_gpu_layers)", placeholder="Deixe em branco para detecção automática")
                n_batch_input = gr.Textbox(label="Tamanho do Lote (n_batch)", placeholder="Deixe em branco para detecção automática")
                submit_btn = gr.Button("Enviar")
            with gr.Column(scale=1):
                report_output = gr.HTML(label="Relatório")
                links_output = gr.HTML(label="Links Encontrados")
        # Define a interface como geradora para feedback em tempo real
        submit_btn.click(fn=gradio_interface, 
                 inputs=[query_input, mode_input, language_input, style_input,
                         investigation_focus, num_sites, search_news, search_leaked_data, temperature_input, velocidade_input, gpu_layers_input, n_batch_input],
                 outputs=[report_output, links_output])
    return demo

demo = build_gradio_interface()

def launch_gradio():
    logger.info("Iniciando Gradio (o sistema escolherá uma porta livre).")
    demo.launch(share=True)

# ===== Execução da Aplicação =====
if __name__ == '__main__':
    gradio_thread = threading.Thread(target=launch_gradio, daemon=True)
    gradio_thread.start()
    logger.info("Executando a aplicação Flask na porta 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
