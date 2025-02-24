import os
import time
import re
import logging
import requests
import io
import psutil
import threading
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
from cachetools import TTLCache, cached
import emoji
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from duckduckgo_search import DDGS
from PIL import Image, ExifTags
from pyngrok import ngrok

# Downloads iniciais do NLTK (necessário apenas na primeira execução)
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
    """Faz o download do modelo caso não esteja disponível localmente."""
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
    """Carrega o modelo neural. Se não existir localmente, faz o download."""
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

def build_messages(query: str, lang_config: dict, style: str) -> tuple:
    """
    Constrói uma lista de mensagens e define a temperatura de acordo com o estilo.
    :param query: Consulta do usuário.
    :param lang_config: Configuração de idioma a partir de LANGUAGE_MAP.
    :param style: Estilo desejado ("Técnico" ou "Livre").
    :return: Tupla contendo a lista de mensagens e a temperatura.
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
    Gera uma resposta a partir da consulta, utilizando o modelo de linguagem, com cache.
    :param query: Consulta do usuário.
    :param lang: Idioma desejado.
    :param style: Estilo de resposta.
    :return: Resposta final (possivelmente validada e corrigida quanto ao idioma).
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
    Verifica se o texto está no idioma esperado; se não, corrige-o.
    :param text: Texto a ser validado.
    :param lang_config: Configuração de idioma com código e instrução.
    :return: Texto no idioma esperado.
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
    :param text: Texto que necessita de correção de idioma.
    :param lang_config: Configuração de idioma com instrução.
    :return: Texto traduzido para o idioma desejado.
    """
    try:
        correction_prompt = f"{lang_config['instruction']}. Traduza o seguinte texto para o idioma correto:\n\n{text}"
        corrected = model.create_chat_completion(
            messages=[
                {"role": "system", "content": correction_prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            stop=["</s>"]
        )
        corrected_text = corrected['choices'][0]['message']['content']
        return f"[Traduzido]\n{corrected_text}"
    except Exception as e:
        logger.error(f"❌ Erro na correção de idioma: {e}")
        return text

def advanced_forensic_analysis(text: str) -> dict:
    """
    Realiza uma análise forense aprimorada no texto fornecido, extraindo informações relevantes.
    Extrações incluem: IPv4, IPv6, e-mails, telefones, URLs, endereços MAC, hashes MD5, SHA1, SHA256 e IDs CVE.
    :param text: Texto a ser analisado.
    :return: Dicionário com listas dos itens encontrados para cada categoria.
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

        ipv4_addresses = ip_pattern.findall(text)
        if ipv4_addresses:
            forensic_info['Endereços IPv4'] = list(set(ipv4_addresses))
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
        logger.error(f"❌ Erro durante análise forense: {e}")
    return forensic_info

def convert_to_degrees(value) -> float:
    """
    Converte coordenadas GPS em formato de frações para graus decimais.
    :param value: Valor de coordenadas GPS em formato fracionário.
    :return: Coordenada em graus decimais.
    """
    try:
        d, m, s = value
        degrees = d[0] / d[1]
        minutes = m[0] / m[1] / 60
        seconds = s[0] / s[1] / 3600
        return degrees + minutes + seconds
    except Exception as e:
        logger.error(f"❌ Erro na conversão de coordenadas GPS: {e}")
        raise e

def analyze_image_metadata(url: str) -> dict:
    """
    Obtém e analisa os metadados EXIF de uma imagem a partir de sua URL.
    Se houver dados GPS, converte para graus decimais e adiciona link para o Google Maps.
    :param url: URL da imagem.
    :return: Dicionário com os metadados extraídos ou mensagem de erro.
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
                tag = ExifTags.TAGS.get(tag_id, tag_id)
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
                    meta["Coordenadas GPS"] = f"{lat}, {lon} (Google Maps: https://maps.google.com/?q={lat},{lon})"
                except Exception as e:
                    meta["Erro de extração de GPS"] = str(e)
        else:
            meta["info"] = "Nenhum metadado EXIF encontrado."
        return meta
    except Exception as e:
        logger.error(f"❌ Erro ao analisar metadados da imagem: {e}")
        return {"error": str(e)}

def process_investigation(target: str, sites_meta: int = 5, research_focus: str = "", search_news: bool = False, search_leaked_data: bool = False) -> str:
    """
    Realiza uma investigação online com base no alvo informado.
    :param target: Alvo da investigação.
    :param sites_meta: Número máximo de sites a serem pesquisados.
    :param research_focus: Foco adicional da investigação.
    :param search_news: Indica se notícias devem ser pesquisadas.
    :param search_leaked_data: Indica se dados vazados devem ser pesquisados.
    :return: Relatório detalhado da investigação.
    """
    if not target.strip():
        return "Erro: Por favor, insira um alvo para investigação."
    try:
        with DDGS() as ddgs:
            resultados = ddgs.text(keywords=target, max_results=sites_meta)
    except Exception as e:
        logger.error(f"❌ Erro na pesquisa com DDGS: {e}")
        return f"Erro na pesquisa: {e}"
    
    info_msg = f"Apenas {len(resultados)} sites encontrados para '{target}'.<br>" if len(resultados) < sites_meta else ""
    formatted_results = "<br>".join(
        f"• {res.get('title', 'Sem título')}<br>&nbsp;&nbsp;{res.get('href', 'Sem link')}<br>&nbsp;&nbsp;{res.get('body', '')}"
        for res in resultados
    )
    links_table = (
        "<table border='1' style='width:100%; border-collapse: collapse; text-align: left;'>"
        "<thead><tr><th>Título</th><th>Link</th></tr></thead><tbody>"
    )
    for res in resultados:
        title = res.get('title', 'Sem título')
        href = res.get('href', 'Sem link')
        links_table += f"<tr><td>{title}</td><td><a href='{href}' target='_blank'>{href}</a></td></tr>"
    links_table += "</tbody></table>"
    
    forensic_analysis = advanced_forensic_analysis(formatted_results)
    forensic_details = "<br>".join(f"{k}: {v}" for k, v in forensic_analysis.items() if v)
    
    investigation_prompt = f"Analisar os dados obtidos sobre '{target}'"
    if research_focus:
        investigation_prompt += f", focando em '{research_focus}'"
    investigation_prompt += "<br><br>Resultados de sites:<br>" + formatted_results
    if forensic_details:
        investigation_prompt += "<br><br>Análise Forense Extraída:<br>" + forensic_details
    investigation_prompt += "<br><br>Elabore um relatório detalhado com ligações, riscos e informações relevantes."
    
    try:
        research_response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "Você é um perito em investigação online e análise forense digital. Seja minucioso e detalhado."},
                {"role": "user", "content": investigation_prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            stop=["</s>"]
        )
        report = research_response['choices'][0]['message']['content']
        final_report = info_msg + "<br>" + report + "<br><br>Links encontrados:<br>" + links_table
        return final_report
    except Exception as e:
        logger.error(f"❌ Erro na investigação: {e}")
        return f"Erro na investigação: {e}"

@app.route('/')
def index():
    """Renderiza a página principal."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """
    Endpoint principal que processa as requisições em três modos:
    - Chat
    - Investigação
    - Metadados
    """
    user_input = request.form.get('user_input', '')
    mode = request.form.get('mode', 'Chat')
    lang = request.form.get('idioma', 'Português')
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
    else:
        # Modo Chat
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
    return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    # Substitua "YOUR_NGROK_AUTHTOKEN" por um authtoken válido obtido em https://dashboard.ngrok.com/get-started/your-authtoken
    ngrok.set_auth_token("2tVBewEibjKz94pYVqJCr4q2HTx_4krjDHHxie5gNYJZvcRwh")
    app.run(host='0.0.0.0', port=5000, debug=True)
