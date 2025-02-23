import os, time, re, logging, requests, io, psutil, threading, nltk
from flask import Flask, render_template, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
from cachetools import TTLCache, cached
import emoji
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from duckduckgo_search import DDGS
from PIL import Image, ExifTags

# Configurações iniciais do NLTK
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

# Inicializa o Flask
app = Flask(__name__)

def download_model():
    try:
        logger.info("⏬ Baixando Modelo...")
        hf_hub_download(
            repo_id=DEFAULT_MODEL_NAME,
            filename=DEFAULT_MODEL_FILE,
            local_dir=DEFAULT_LOCAL_MODEL_DIR,
            resume_download=True
        )
    except Exception as e:
        logger.error(f"❌ Falha no Download: {str(e)}")
        raise e

def load_model():
    model_path = os.path.join(DEFAULT_LOCAL_MODEL_DIR, DEFAULT_MODEL_FILE)
    if not os.path.exists(model_path):
        download_model()
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=psutil.cpu_count(),
            n_gpu_layers=33 if psutil.virtual_memory().available > 4*1024**3 else 15
        )
        logger.info("🤖 Modelo Neural Carregado")
        return model
    except Exception as e:
        logger.error(f"❌ Erro na Inicialização: {str(e)}")
        raise e

model = load_model()

def generate_response(query, lang, style):
    start_time = time.time()
    lang_config = LANGUAGE_MAP.get(lang, LANGUAGE_MAP['Português'])
    if style == "Técnico":
        system_instruction = f"{lang_config['instruction']}. Seja detalhado e técnico."
        temperature = 0.7
    else:
        system_instruction = f"{lang_config['instruction']}. Responda de forma livre e criativa."
        temperature = 0.9

    system_msg = {"role": "system", "content": system_instruction}
    user_msg = {"role": "user", "content": query}
    
    response = model.create_chat_completion(
        messages=[system_msg, user_msg],
        temperature=temperature,
        max_tokens=800,
        stop=["</s>"]
    )
    raw_response = response['choices'][0]['message']['content']
    final_response = validate_language(raw_response, lang_config)
    logger.info(f"✅ Resposta gerada em {time.time()-start_time:.2f}s")
    return final_response

def validate_language(text, lang_config):
    try:
        if detect(text) != lang_config['code'].split('-')[0]:
            return correct_language(text, lang_config)
        return text
    except Exception:
        return text

def correct_language(text, lang_config):
    correction_prompt = f"Traduza para {lang_config['instruction']}:\n{text}"
    corrected = model.create_chat_completion(
        messages=[{"role": "user", "content": correction_prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    return f"[Traduzido]\n{corrected['choices'][0]['message']['content']}"

def process_investigation(target, sites_meta=5, investigation_focus="", search_news=False, search_leaked_data=False):
    logger.info(f"Alvo recebido para investigação: {repr(target)}")
    if not target or not target.strip():
        return "Erro: Por favor, insira um alvo para investigação."
    
    try:
        with DDGS() as ddgs:
            results = ddgs.text(keywords=target, max_results=sites_meta)
        info_msg = f"Apenas {len(results)} sites encontrados para '{target}'.<br>" if len(results) < sites_meta else ""
        
        # Formata resultados para exibição e cria tabela de links
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
        logger.error(f"❌ Erro na investigação: {str(e)}")
        return f"Erro na investigação: {str(e)}"

def advanced_forensic_analysis(text):
    forensic_info = {}
    ip_addresses = re.findall(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', text)
    if ip_addresses:
        forensic_info['Endereços IP'] = list(set(ip_addresses))
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    if emails:
        forensic_info['E-mails'] = list(set(emails))
    phones = re.findall(r'\+?\d[\d\s()-]{7,}\d', text)
    if phones:
        forensic_info['Telefones'] = list(set(phones))
    return forensic_info

def analyze_image_metadata(url):
    try:
        response = requests.get(url)
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
                    lat = convert_to_degrees(gps_info[2])
                    if gps_info[1] != "N":
                        lat = -lat
                    lon = convert_to_degrees(gps_info[4])
                    if gps_info[3] != "E":
                        lon = -lon
                    meta["GPS Coordinates"] = f"{lat}, {lon} (Google Maps: https://maps.google.com/?q={lat},{lon})"
                except Exception as e:
                    meta["GPS Extraction Error"] = str(e)
        else:
            meta["info"] = "Nenhum metadado EXIF encontrado."
        return meta
    except Exception as e:
        return {"error": str(e)}

def convert_to_degrees(value):
    d, m, s = value
    return d[0] / d[1] + (m[0] / m[1]) / 60 + (s[0] / s[1]) / 3600

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('user_input', '')
    mode = request.form.get('mode', 'Chat')
    lang = request.form.get('language', 'Português')
    style = request.form.get('style', 'Técnico')
    
    if mode == "Investigação":
        if not user_input or not user_input.strip():
            return jsonify({'response': "Erro: Por favor, insira um alvo para investigação."})
        try:
            sites_meta = int(request.form.get('sites_meta', 5))
            investigation_focus = request.form.get('investigation_focus', '')
            search_news = request.form.get('search_news', 'false').lower() == 'true'
            search_leaked_data = request.form.get('search_leaked_data', 'false').lower() == 'true'
            response_text = process_investigation(user_input, sites_meta, investigation_focus, search_news, search_leaked_data)
            return jsonify({'response': response_text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    elif mode == "Metadados":
        if not user_input or not user_input.strip():
            return jsonify({'response': "Erro: Por favor, insira um link de imagem."})
        try:
            meta = analyze_image_metadata(user_input)
            formatted_meta = "<br>".join(f"{k}: {v}" for k, v in meta.items())
            return jsonify({'response': formatted_meta})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:  # Modo Chat
        try:
            response_text = generate_response(user_input, lang, style)
            return jsonify({'response': response_text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
