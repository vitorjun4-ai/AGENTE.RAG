import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AzureOpenAI, APIError
import streamlit as st
import pandas as pd
from pypdf import PdfReader

# Carrega variáveis locais se existirem (para execução local com .env)
load_dotenv(override=True)

# Compatibilidade entre st.secrets e os nomes de variáveis de ambiente solicitados
def get_env_variable(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

AZURE_ENDPOINT = get_env_variable("ENDPOINT")
AZURE_API_KEY = get_env_variable("API_KEY")
AZURE_API_VERSION = get_env_variable("API_VERSION", "2025-04-01-preview")
DEPLOYMENT_NAME = get_env_variable("GPT5_MODEL")

# Inicialização do Cliente Azure OpenAI
client = None
if AZURE_ENDPOINT and AZURE_API_KEY:
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )
    except Exception:
        pass

@st.cache_data(ttl=3600)
def raspar_site_dados():
    url = "https://gratuitos.netlify.app/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove elementos indesejados
        for script in soup(["script", "style"]):
            script.extract()
            
        texto_limpo = soup.get_text(separator="\n")
        linhas = [linha.strip() for linha in texto_limpo.splitlines() if linha.strip()]
        return "\n".join(linhas)
    except Exception as e:
        return f"Erro ao acessar o site: {e}"

@st.cache_data
def carregar_dados_locais():
    conteudo_csv = "dados.csv"
    conteudo_pdf = "edu.pdf"
    
    # Leitura de arquivo CSV se existir no diretório
    try:
        if os.path.exists("dados.csv"):
            df = pd.read_csv("dados.csv")
            conteudo_csv = df.to_string(index=False)
    except Exception as e:
        conteudo_csv = f"Erro ao ler CSV: {e}"

    # Leitura de arquivo PDF se existir no diretório
    try:
        if os.path.exists("documento.pdf"):
            reader = PdfReader("documento.pdf")
            texto_pdf_list = []
            for pagina in reader.pages:
                texto_pagina = pagina.extract_text()
                if texto_pagina:
                    texto_pdf_list.append(texto_pagina)
            conteudo_pdf = "\n".join(texto_pdf_list)
    except Exception as e:
        conteudo_pdf = f"Erro ao ler PDF: {e}"

    return conteudo_csv, conteudo_pdf

def validar_configuracao():
    faltando = []
    if not AZURE_ENDPOINT:
        faltando.append("ENDPOINT")
    if not AZURE_API_KEY:
        faltando.append("API_KEY")
    if not DEPLOYMENT_NAME:
        faltando.append("GPT5_MODEL")
    return faltando

def main():
    st.set_page_config(page_title="Consulta de Unidades", page_icon="🏫", layout="centered")
    
    st.title("🏫 Consulta de Unidades e Cursos")
    st.write("Digite sua dúvida abaixo para consultar as informações disponíveis no portal e nos arquivos locais.")

    faltando = validar_configuracao()
    if faltando:
        st.error(f"Erro de configuração: As seguintes variáveis não foram definidas no ambiente ou secrets: {', '.join(faltando)}")
        return

    # Executa o scraping e a leitura dos arquivos locais com cache
    dados_site = raspar_site_dados()
    dados_csv, dados_pdf = carregar_dados_locais()

    # Consolida todo o contexto do RAG
    dados_recuperados = f"""
--- DADOS DO SITE ---
{dados_site}

--- DADOS DO CSV ---
{dados_csv}

--- DADOS DO PDF ---
{dados_pdf}
"""

    # Interface de Entrada do Aluno
    with st.form(key="form_pergunta"):
        pergunta = st.text_input("Qual a sua dúvida ou unidade que deseja buscar?", placeholder="Ex: Qual o endereço da unidade mais próxima?")
        botao_enviar = st.form_submit_button("Buscar Unidade / Perguntar")

    if botao_enviar:
        if not pergunta.strip():
            st.warning("Por favor, digite uma pergunta válida.")
            return

        if not client:
            st.error("Cliente Azure OpenAI não inicializado corretamente.")
            return

        prompt = f"""Você é um assistente educacional prestativo. Use apenas os dados extraídos das fontes abaixo para responder de forma clara, objetiva e educativa:

Dados disponíveis:
{dados_recuperados}
"""

        try:
            with st.spinner("Consultando informações..."):
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": pergunta
                        }
                    ]
                )
                resposta_texto = response.choices[0].message.content
                
                st.markdown("### Resposta:")
                st.success(resposta_texto)

        except APIError as e:
            st.error(f"Erro na API do Azure: {e}")
        except Exception as e:
            st.error(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()