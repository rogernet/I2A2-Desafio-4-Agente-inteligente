import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ingestion import process_zip
from agent import build_llm, answer_question

load_dotenv()

st.set_page_config(page_title="Consulta de Notas Fiscais", page_icon="▦", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #14213d; }
    .stApp { background: #ffffff; }
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label { color: #14213d; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; color: #14213d; }
    code, pre, .stCode { font-family: 'IBM Plex Mono', monospace; }
    button[data-baseweb="tab"] { color: #14213d; }
    button[data-baseweb="tab"] p { color: #14213d; font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #0f766e; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #0f766e; }
    .stTextInput input { color: #14213d; }
    .stButton > button { background: #ffffff; color: #14213d; border: 1px solid #d1d5db; font-weight: 500; }
    .stButton > button:hover { border-color: #0f766e; color: #0f766e; background: #ffffff; }
    .stButton > button:active, .stButton > button:focus { background: #ffffff; color: #0f766e; border-color: #0f766e; box-shadow: none; }
    .brand { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; color: #14213d; }
    .brand span { color: #0f766e; }
    .subtle { color: #6b7280; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="brand">nota<span>.</span>ask</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtle">Pergunte sobre suas notas fiscais em português.</p>',
    unsafe_allow_html=True,
)

for key in ("db_path", "schema_text", "dictionary_text", "tables"):
    st.session_state.setdefault(key, None)

tab_load, tab_query = st.tabs(["Carga", "Consulta"])

with tab_load:
    st.subheader("Enviar dados")
    st.write(
        "Suba um arquivo .ZIP com um ou mais CSVs e, de preferência, um dicionário de dados."
    )
    uploaded = st.file_uploader(
        "Arquivo ZIP", type=["zip"], label_visibility="collapsed"
    )

    if uploaded is not None and st.button("Processar"):
        with st.spinner("Descompactando e carregando as tabelas..."):
            try:
                result = process_zip(uploaded.getvalue())
                st.session_state.update(result)
                st.success(
                    f"{len(result['tables'])} tabela(s) carregada(s). Vá para a aba Consulta."
                )
            except Exception as exc:
                st.error(f"Falha ao processar o arquivo: {exc}")

    if st.session_state["schema_text"]:
        st.markdown("**Estrutura carregada**")
        st.code(st.session_state["schema_text"], language="text")
        if st.session_state["dictionary_text"]:
            with st.expander("Dicionário de dados"):
                st.text(st.session_state["dictionary_text"])

with tab_query:
    st.subheader("Faça uma pergunta")

    if not st.session_state["db_path"]:
        st.info("Carregue um arquivo na aba Carga para começar.")
    else:
        question = st.text_input(
            "Pergunta",
            placeholder="Ex.: Quais foram os cinco maiores fornecedores por valor total?",
            label_visibility="collapsed",
        )
        if st.button("Perguntar") and question.strip():
            with st.spinner("Interpretando e consultando..."):
                llm = build_llm()
                outcome = answer_question(
                    llm,
                    st.session_state["db_path"],
                    st.session_state["schema_text"],
                    st.session_state["dictionary_text"] or "",
                    question.strip(),
                )

            st.markdown("**Resposta**")
            st.write(outcome["answer"])

            df = outcome["dataframe"]
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.dataframe(df, use_container_width=True)
                numeric = df.select_dtypes(include="number").columns
                if df.shape[1] == 2 and len(numeric) == 1 and 1 < len(df) <= 50:
                    label = [c for c in df.columns if c not in numeric][0]
                    st.bar_chart(df.set_index(label)[numeric[0]])

            with st.expander("SQL executado"):
                st.code(outcome["sql"], language="sql")
