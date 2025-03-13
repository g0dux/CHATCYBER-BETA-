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
from bs4 import BeautifulSoup  # Para scraping de conteúdo

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

# ===== COMPILED_REGEX_PATTERNS (Definição original) =====
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

# ===== Funções para Geração de Resposta e Multi-turn Conversation =====
def build_messages(query: str, lang_config: dict, style: str, conversation_history: list = None) -> tuple[list, float]:
    """
    Cria as mensagens do sistema com base no histórico de conversa (se houver).
    """
    if style == "Técnico":
        system_instruction = f"{lang_config['instruction']}. Seja detalhado e técnico."
        temperature = 0.7
    else:
        system_instruction = f"{lang_config['instruction']}. Responda de forma livre e criativa."
        temperature = 0.9

    messages = [{"role": "system", "content": system_instruction}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})
    return messages, temperature

def generate_response_multi_turn(query: str, conversation_history: list, lang: str, style: str) -> tuple[str, list]:
    """
    Gera resposta levando em conta o histórico de conversa.
    Retorna a resposta e o histórico atualizado.
    """
    start_time = time.time()
    lang_config = LANGUAGE_MAP.get(lang, LANGUAGE_MAP['Português'])
    messages, temperature = build_messages(query, lang_config, style, conversation_history)
    try:
        response = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=800,
            stop=["</s>"]
        )
        raw_response = response['choices'][0]['message']['content']
        final_response = validate_language(raw_response, lang_config)
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": final_response})
        logger.info(f"✅ Resposta gerada em {time.time() - start_time:.2f}s")
        return final_response, conversation_history
    except Exception as e:
        logger.error(f"❌ Erro ao gerar resposta: {e}")
        return f"Erro ao gerar resposta: {e}", conversation_history

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

# ===== Módulo de Sumarização =====
def summarize_text(text: str, lang: str, detail_level: str) -> str:
    """
    Gera um resumo do texto com o nível de detalhe desejado.
    detail_level pode ser 'Resumo', 'Intermediário' ou 'Detalhado'
    """
    lang_config = LANGUAGE_MAP.get(lang, LANGUAGE_MAP['Português'])
    prompt = f"{lang_config['instruction']}. Por favor, resuma o seguinte texto com um nível de detalhe {detail_level}:\n\n{text}"
    try:
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=600,
            stop=["</s>"]
        )
        summary = response['choices'][0]['message']['content']
        return summary
    except Exception as e:
        logger.error(f"❌ Erro ao gerar resumo: {e}")
        return f"Erro ao resumir o texto: {e}"

# ===== Mecanismo de Busca Avançado =====
def scrape_page(url: str) -> dict:
    """
    Tenta extrair informações básicas (título e meta descrição) da página.
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else "Sem título"
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_desc = meta_desc_tag["content"] if meta_desc_tag and "content" in meta_desc_tag.attrs else ""
        return {"title": title, "description": meta_desc}
    except Exception as e:
        logger.error(f"Erro ao fazer scraping da página {url}: {e}")
        return {"title": "Erro", "description": ""}

def advanced_search(query: str, max_results: int, advanced: bool = False) -> list:
    """
    Realiza uma busca utilizando DuckDuckGo e, se advanced=True,
    tenta fazer scraping adicional dos links encontrados.
    """
    results = []
    try:
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(keywords=query, max_results=max_results))
        for res in ddgs_results:
            if advanced:
                extra = scrape_page(res.get('href', ''))
                res['scraped_title'] = extra.get('title', '')
                res['scraped_description'] = extra.get('description', '')
            results.append(res)
        return results
    except Exception as e:
        logger.error(f"Erro na busca avançada: {e}")
        return results

# ===== Análise de Sentimentos e Detecção de Nuances Linguísticas =====
def detect_linguistic_nuances(text: str, lang: str) -> str:
    """
    Utiliza o modelo para identificar ironia, sarcasmo ou nuances no texto.
    Também complementa com análise de sentimento usando VADER.
    """
    prompt = f"Analise o seguinte texto e identifique se há indícios de ironia, sarcasmo ou outras nuances linguísticas:\n\n{text}"
    try:
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            stop=["</s>"]
        )
        nuances = response['choices'][0]['message']['content']
        sentiment = sentiment_analyzer.polarity_scores(text)
        nuances += f"\n\nAnálise de Sentimentos (VADER): {sentiment}"
        return nuances
    except Exception as e:
        logger.error(f"Erro na detecção de nuances: {e}")
        return f"Erro na detecção de nuances: {e}"

# ===== Funções Integradas para Chat com Resumo e Nuances =====
def chat_interface(user_message, chat_history, language, style, summary_flag, nuances_flag, detail_level):
    """
    Função para conversas multi-turn com opção de aplicar resumo e análise de nuances na resposta.
    """
    response, chat_history = generate_response_multi_turn(user_message, chat_history, language, style)
    extra_info = ""
    if summary_flag:
        summary_text = summarize_text(response, language, detail_level)
        extra_info += "\n\n<b>Resumo:</b>\n" + summary_text
    if nuances_flag:
        nuance_text = detect_linguistic_nuances(response, language)
        extra_info += "\n\n<b>Análise de Nuances:</b>\n" + nuance_text
    # Se houver informações extras, anexa-as à resposta da IA
    final_response = response + extra_info if extra_info else response
    if chat_history and chat_history[-1]["role"] == "assistant":
        chat_history[-1]["content"] = final_response
    formatted_history = []
    for i in range(0, len(chat_history), 2):
        user_msg = chat_history[i]["content"]
        ai_msg = chat_history[i+1]["content"] if i+1 < len(chat_history) else ""
        formatted_history.append((user_msg, ai_msg))
    return formatted_history, chat_history

# ===== Função Integrada para Investigação com Resumo e Nuances =====
def investigation_interface(query, investigation_focus, num_sites, search_news, search_leaked_data, advanced_search_enabled, language, summary_flag, nuances_flag, detail_level):
    """
    Realiza investigação online e, opcionalmente, aplica resumo e análise de nuances ao resultado.
    """
    results = advanced_search(query, int(num_sites), advanced=advanced_search_enabled)
    # Gera um relatório de texto simples com os resultados
    text_results = ""
    for i, res in enumerate(results, 1):
        text_results += f"{i}. {res.get('title', 'Sem título')} - {res.get('href', 'Sem link')}\n"
    links_table = "<h3>Resultados de Busca para '" + query + "'</h3><br>" + "<br>".join(
        [f"{i}. <a href='{res.get('href', 'Sem link')}' target='_blank'>{res.get('title', 'Sem título')}</a>" for i, res in enumerate(results, 1)]
    )
    extra_info = ""
    if summary_flag:
        summary_text = summarize_text(text_results, language, detail_level)
        extra_info += f"<br><br><b>Resumo:</b><br>{summary_text}"
    if nuances_flag:
        nuance_text = detect_linguistic_nuances(text_results, language)
        extra_info += f"<br><br><b>Análise de Nuances:</b><br>{nuance_text}"
    final_output = links_table + extra_info
    return final_output

def clear_history():
    """
    Função para limpar o histórico de conversa.
    """
    return [], []  # Retorna histórico vazio

# ===== Interface via Gradio =====
def build_gradio_interface():
    with gr.Blocks(title="IA Avançada - Chat & Investigação com Resumo e Nuances") as demo:
        gr.Markdown("# IA Avançada")
        with gr.Tabs():
            # Aba de Chat Multi-turn
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(label="Conversa com a IA")
                with gr.Row():
                    txt_input = gr.Textbox(label="Sua mensagem", placeholder="Digite sua mensagem aqui...", lines=2)
                    send_btn = gr.Button("Enviar")
                    clear_btn = gr.Button("Limpar Histórico")
                state = gr.State([])  # Histórico de mensagens
                with gr.Row():
                    chat_lang = gr.Radio(["Português", "English", "Español", "Français", "Deutsch"],
                                           label="Idioma", value="Português", interactive=True)
                    chat_style = gr.Radio(["Técnico", "Criativo"],
                                          label="Estilo", value="Técnico", interactive=True)
                    chat_summary_chk = gr.Checkbox(label="Aplicar Resumo", value=False)
                    chat_nuances_chk = gr.Checkbox(label="Analisar Nuances", value=False)
                    chat_detail = gr.Radio(["Resumo", "Intermediário", "Detalhado"],
                                           label="Nível de Resumo", value="Intermediário", interactive=True)
                send_btn.click(fn=chat_interface,
                               inputs=[txt_input, state, chat_lang, chat_style, chat_summary_chk, chat_nuances_chk, chat_detail],
                               outputs=[chatbot, state])
                clear_btn.click(fn=clear_history, inputs=[], outputs=[chatbot, state])
            # Aba de Investigação
            with gr.TabItem("Investigação"):
                inv_query = gr.Textbox(label="Alvo/Consulta", placeholder="Digite o que deseja investigar...", lines=2)
                inv_focus = gr.Textbox(label="Foco da Investigação (opcional)", placeholder="Ex: vulnerabilidades, evidências, etc.", lines=1)
                num_sites = gr.Number(label="Número de Sites", value=5, precision=0)
                search_news = gr.Checkbox(label="Pesquisar Notícias", value=False)
                search_leaked_data = gr.Checkbox(label="Pesquisar Dados Vazados", value=False)
                advanced_search_chk = gr.Checkbox(label="Busca Avançada (Scraping Extra)", value=False)
                inv_lang = gr.Radio(["Português", "English", "Español", "Français", "Deutsch"],
                                    label="Idioma", value="Português", interactive=True)
                inv_summary_chk = gr.Checkbox(label="Aplicar Resumo", value=False)
                inv_nuances_chk = gr.Checkbox(label="Analisar Nuances", value=False)
                inv_detail = gr.Radio(["Resumo", "Intermediário", "Detalhado"],
                                      label="Nível de Resumo", value="Intermediário", interactive=True)
                inv_output = gr.HTML(label="Resultados da Investigação")
                inv_btn = gr.Button("Pesquisar")
                inv_btn.click(fn=investigation_interface,
                              inputs=[inv_query, inv_focus, num_sites, search_news, search_leaked_data, advanced_search_chk, inv_lang, inv_summary_chk, inv_nuances_chk, inv_detail],
                              outputs=inv_output)
        gr.Markdown("### Desenvolvido com funcionalidades avançadas para uma IA interativa e customizável.")
    return demo

demo = build_gradio_interface()

def launch_gradio():
    print("Iniciando Gradio (o sistema escolherá uma porta livre).")
    demo.launch(share=True)

if __name__ == '__main__':
    launch_gradio()
