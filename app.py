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
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from duckduckgo_search import DDGS
from PIL import Image, ExifTags
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
from nltk.sentiment import SentimentIntensityAnalyzer
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

# ===== Cache simples em memória =====
cache = {}

def get_cached_response(query: str, lang: str, style: str) -> str:
    key = f"response:{query}:{lang}:{style}"
    return cache.get(key)

def set_cached_response(query: str, lang: str, style: str, response_text: str, ttl: int = 3600) -> None:
    key = f"response:{query}:{lang}:{style}"
    cache[key] = response_text  # TTL não implementado nesta versão

# ===== COMPILED_REGEX_PATTERNS =====
COMPILED_REGEX_PATTERNS = {
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
    'version': re.compile(r'\b\d+\.\d+(\.\d+)?\b'),
    'bitcoin_wif': re.compile(r'\b[5KL][1-9A-HJ-NP-Za-km-z]{50,51}\b'),
    'github_repo': re.compile(r'https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+'),
    'sql_injection': re.compile(r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC)\b"),
    'unix_path': re.compile(r'(/(?:[\w._-]+/)*[\w._-]+)'),
    'base32': re.compile(r'\b(?:[A-Z2-7]{8,})\b'),
    'bitcoin_cash': re.compile(r'\b(?:q|p)[a-z0-9]{41}\b'),
    'passport': re.compile(r'\b\d{9}\b'),
    'win_registry': re.compile(r'(?:HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER|HKEY_CLASSES_ROOT|HKEY_USERS|HKEY_CURRENT_CONFIG)\\[\\\w]+'),
    'onion_v2': re.compile(r'\b[a-z2-7]{16}\.onion\b'),
    'onion_v3': re.compile(r'\b[a-z2-7]{56}\.onion\b'),
    'domain': re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'),
    'e164_phone': re.compile(r'\+\d{10,15}'),
    'geo_coordinates': re.compile(r'[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)[, ]+\s*[-+]?((1[0-7]\d)|([1-9]?\d))(\.\d+)?')
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
        from langdetect import detect
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

# ===== Funções de Análise Forense e Processamento de Texto =====
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
        "<thead><tr><th>Nº</th><th>Título</th><th>Link</th></tr></thead><tbody>"
    )
    for i, res in enumerate(results, 1):
        title = res.get('title', 'Sem título')
        href = res.get('href', 'Sem link')
        links_table += f"<tr><td>{i}</td><td>{title}</td><td><a href='{href}' target='_blank'>{href}</a></td></tr>"
    links_table += "</tbody></table>"
    return formatted_text, links_table, info_message

def process_investigation(target: str, sites_meta: int = 5, investigation_focus: str = "",
                          search_news: bool = False, search_leaked_data: bool = False) -> tuple:
    logger.info(f"🔍 Iniciando investigação para: {repr(target)}")
    if not target.strip():
        return "Erro: Por favor, insira um alvo para investigação.", ""
    
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
        return report, links_combined
    except Exception as e:
        logger.error(f"❌ Erro na investigação: {e}")
        return f"Erro na investigação: {e}", ""

# ===== Integração com Gradio.live (Interface Aprimorada) =====
def gradio_interface(query, mode, language, style, investigation_focus, num_sites, search_news, search_leaked_data):
    if mode == "Chat":
        result = generate_response(query, language, style)
        return result, ""  # Segundo campo vazio para links
    elif mode == "Investigação":
        sites_meta = int(num_sites)
        report, links_table = process_investigation(query, sites_meta, investigation_focus,
                                                     search_news, search_leaked_data)
        return report, links_table
    else:
        return "Modo não suportado.", ""

def build_gradio_interface():
    with gr.Blocks(title="Interface de IA - Chat & Investigação") as demo:
        gr.Markdown("# Interface de IA - Chat & Investigação")
        gr.Markdown("### Insira os parâmetros para interagir com a IA")
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(label="Pergunta/Alvo", placeholder="Digite sua pergunta (Chat) ou o alvo (Investigação)...", lines=3)
                mode_input = gr.Radio(["Chat", "Investigação"], label="Modo", value="Chat", interactive=True)
                language_input = gr.Radio(["Português", "English", "Español", "Français", "Deutsch"], label="Idioma", value="Português", interactive=True)
                style_input = gr.Radio(["Técnico", "Criativo"], label="Estilo", value="Técnico", interactive=True)
                investigation_focus = gr.Textbox(label="Foco da Investigação (opcional)", placeholder="Ex: vulnerabilidades, evidências, etc.", lines=1)
                num_sites = gr.Number(label="Número de Sites", value=5, precision=0)
                search_news = gr.Checkbox(label="Pesquisar Notícias", value=False)
                search_leaked_data = gr.Checkbox(label="Pesquisar Dados Vazados", value=False)
                submit_btn = gr.Button("Enviar")
            with gr.Column(scale=1):
                report_output = gr.HTML(label="Relatório")
                links_output = gr.HTML(label="Links Encontrados")
        submit_btn.click(gradio_interface, inputs=[query_input, mode_input, language_input, style_input,
                                                     investigation_focus, num_sites, search_news, search_leaked_data],
                         outputs=[report_output, links_output])
    return demo

demo = build_gradio_interface()

def launch_gradio():
    print("Iniciando Gradio (o sistema escolherá uma porta livre).")
    demo.launch(share=True)

# ===== Execução da Aplicação =====
if __name__ == '__main__':
    launch_gradio()
