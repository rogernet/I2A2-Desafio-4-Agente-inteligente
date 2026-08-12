import os
import re
import difflib

import duckdb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|attach|detach|pragma|vacuum|grant)\b",
    re.IGNORECASE,
)

SQL_SYSTEM = """Você é um analista de dados que traduz perguntas em SQL para o DuckDB.
Regras:
- Gere UMA única consulta SELECT (pode usar WITH).
- Use APENAS os nomes de tabela e de coluna que aparecem no schema abaixo, copiados exatamente, incluindo maiúsculas, acentos e underscores. Nunca invente, encurte ou adivinhe um nome de coluna.
- Se uma coluna necessária não existir no schema, escolha a coluna existente mais próxima em vez de inventar.
- Sempre coloque nomes de coluna entre aspas duplas.
- Colunas de valor e quantidade JÁ SÃO numéricas: use SUM, AVG, MIN, MAX e ORDER BY diretamente sobre elas, sem REPLACE e sem CAST. Colunas de data já são do tipo data.
- Para relacionar duas tabelas de notas fiscais, junte pela coluna de chave de acesso comum (ex.: CHAVE_ACESSO, CHAVE_NFE), nunca por CNPJ ou CPF.
- Se a pergunta puder ser respondida com uma única tabela, não faça JOIN.
- Para perguntas de "maiores", "menores", "top", "ranking" ou "mais/menos", SEMPRE use ORDER BY sobre a coluna numérica na direção certa (DESC para maiores) e LIMIT com a quantidade pedida.
- Ao agregar (soma, contagem, média) por um grupo, inclua a coluna do grupo no SELECT e no GROUP BY.
- REGRA DO GROUP BY: toda coluna do SELECT que não estiver dentro de SUM, COUNT, AVG, MIN ou MAX precisa aparecer no GROUP BY. Não selecione colunas que não sejam necessárias para responder.
- Selecione apenas as colunas pedidas pela pergunta, sem adicionar colunas extras.
- Para agrupar por mês use strftime("coluna_de_data", '%Y-%m'); por ano use strftime("coluna_de_data", '%Y'). Nunca invente funções de data.

Exemplos de forma (adapte os nomes ao schema real, não copie estes nomes):
Pergunta: quais as 3 notas de maior valor?
SQL: SELECT "VALOR_NOTA_FISCAL" FROM "cabecalho" ORDER BY "VALOR_NOTA_FISCAL" DESC LIMIT 3
Pergunta: quais os 5 fornecedores que mais receberam?
SQL: SELECT "RAZAO_SOCIAL_EMITENTE", SUM("VALOR_NOTA_FISCAL") AS total FROM "cabecalho" GROUP BY "RAZAO_SOCIAL_EMITENTE" ORDER BY total DESC LIMIT 5
Pergunta: qual o valor total por mês?
SQL: SELECT strftime("DATA_EMISSAO", '%Y-%m') AS mes, SUM("VALOR_NOTA_FISCAL") AS total FROM "cabecalho" GROUP BY mes ORDER BY mes
Pergunta: quantas notas existem?
SQL: SELECT COUNT(*) AS total FROM "cabecalho"
- Nunca escreva, altere ou apague dados.
- Responda somente com o SQL, sem explicação, sem crases, sem markdown.

Schema disponível:
{schema}

Dicionário de dados:
{dictionary}"""

REPAIR_SYSTEM = """A consulta SQL anterior falhou no DuckDB. Corrija-a.
Use somente nomes de coluna que existem no schema, copiados exatamente.
Se o erro trouxer "Candidate bindings", use um desses nomes.
Responda somente com o SQL corrigido, sem explicação, sem crases.

Schema disponível:
{schema}

SQL que falhou:
{sql}

Erro retornado:
{error}"""

ANSWER_SYSTEM = """Você responde perguntas de negócio a partir do resultado de uma consulta SQL.
- Baseie-se ESTRITAMENTE nas linhas fornecidas, na ordem exata em que aparecem.
- Não reordene, não recalcule e não invente nenhum valor que não esteja no resultado.
- Se houver muitas linhas, diga que a tabela abaixo traz o detalhe completo.
- Escreva em português, de forma direta e curta.
- Se o resultado estiver vazio, diga que não há dados para a pergunta."""


def build_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": temperature,
        }
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=temperature,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )


def _clean_sql(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^SQLQuery:\s*", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :].strip()
    if ";" in text:
        text = text.split(";")[0].strip()
    return text


def is_read_only(sql: str) -> bool:
    stripped = sql.strip().lower()
    if not (stripped.startswith("select") or stripped.startswith("with")):
        return False
    return FORBIDDEN.search(sql) is None


def generate_sql(llm, schema_text: str, dictionary_text: str, question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [("system", SQL_SYSTEM), ("human", "{question}")]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "schema": schema_text,
            "dictionary": dictionary_text or "não fornecido",
            "question": question,
        }
    )
    return _clean_sql(raw)


def repair_sql(llm, schema_text: str, failed_sql: str, error: str) -> str:
    prompt = ChatPromptTemplate.from_messages([("system", REPAIR_SYSTEM)])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"schema": schema_text, "sql": failed_sql, "error": error})
    return _clean_sql(raw)


SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "outer",
    "full",
    "cross",
    "on",
    "using",
    "group",
    "by",
    "order",
    "asc",
    "desc",
    "limit",
    "offset",
    "as",
    "and",
    "or",
    "not",
    "in",
    "is",
    "null",
    "distinct",
    "having",
    "with",
    "union",
    "all",
    "between",
    "like",
    "case",
    "when",
    "then",
    "else",
    "end",
    "sum",
    "avg",
    "count",
    "min",
    "max",
    "cast",
    "double",
    "integer",
    "bigint",
    "varchar",
    "date",
    "replace",
    "coalesce",
    "round",
    "abs",
    "extract",
    "year",
    "month",
    "day",
    "substr",
    "trim",
    "lower",
    "upper",
    "true",
    "false",
}


def schema_names(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    try:
        columns = {
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns"
            ).fetchall()
        }
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
    finally:
        con.close()
    return columns, tables


def correct_identifiers(sql: str, columns: set, tables: set) -> str:
    lower_columns = {c.lower(): c for c in columns}
    skip = {t.lower() for t in tables}

    def fix(token: str) -> str:
        low = token.lower()
        if low in SQL_KEYWORDS or low in skip:
            return token
        if low in lower_columns:
            return lower_columns[low]
        if len(token) < 4 or ("_" not in token and not token.isupper()):
            return token
        match = difflib.get_close_matches(token, list(columns), n=1, cutoff=0.82)
        return match[0] if match else token

    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", lambda m: fix(m.group()), sql)


ALIAS_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?(?:\s+AS)?\s+"?([A-Za-z_][A-Za-z0-9_]*)"?',
    re.IGNORECASE,
)
CLAUSE_WORDS = {
    "where",
    "on",
    "group",
    "order",
    "left",
    "right",
    "inner",
    "outer",
    "full",
    "cross",
    "join",
    "limit",
    "having",
    "using",
}
QUALIFIED_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\.("?)([A-Za-z_][A-Za-z0-9_]*)\2')


def columns_by_table(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT table_name, column_name FROM information_schema.columns"
        ).fetchall()
    finally:
        con.close()
    mapping = {}
    for table, column in rows:
        mapping.setdefault(table, set()).add(column)
    return mapping


def _alias_map(sql: str, tables: set) -> dict:
    amap = {}
    for table, alias in ALIAS_RE.findall(sql):
        if table not in tables:
            continue
        amap[table] = table
        if alias.lower() not in CLAUSE_WORDS and alias != table:
            amap[alias] = table
    return amap


def correct_qualified(sql: str, cols_by_table: dict, tables: set) -> str:
    amap = _alias_map(sql, tables)
    if not amap:
        return sql

    def repl(match):
        alias, quote, col = match.group(1), match.group(2), match.group(3)
        table = amap.get(alias)
        if not table:
            return match.group(0)
        table_cols = cols_by_table.get(table, set())
        if col in table_cols:
            return match.group(0)
        best = difflib.get_close_matches(col, list(table_cols), n=1, cutoff=0.6)
        if best:
            return f"{alias}.{quote}{best[0]}{quote}"
        return match.group(0)

    return QUALIFIED_RE.sub(repl, sql)


BAD_COL_RE = re.compile(
    r'(?:column named|Referenced column)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', re.IGNORECASE
)
CANDIDATES_RE = re.compile(r"Candidate bindings:\s*:?\s*(.+)", re.IGNORECASE)


def repair_from_error(sql: str, error: str):
    bad = BAD_COL_RE.search(error)
    cand = CANDIDATES_RE.search(error)
    if not bad or not cand:
        return None
    bad_col = bad.group(1)
    candidates = re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', cand.group(1))
    if not candidates:
        return None
    best = difflib.get_close_matches(bad_col, candidates, n=1, cutoff=0.0) or [
        candidates[0]
    ]
    fixed = re.sub(r"\b" + re.escape(bad_col) + r"\b", best[0], sql)
    return fixed if fixed != sql else None


def repair_cast(sql: str, error: str):
    if "No function matches" not in error and "explicit type cast" not in error.lower():
        return None

    def wrap(match):
        fn, arg = match.group(1), match.group(2).strip()
        if "cast" in arg.lower():
            return match.group(0)
        return f"{fn}(CAST(REPLACE({arg}, ',', '.') AS DOUBLE))"

    fixed = re.sub(r"\b(SUM|AVG)\s*\(\s*([^()]+?)\s*\)", wrap, sql, flags=re.IGNORECASE)
    return fixed if fixed != sql else None


def repair_numeric_cast(sql: str, error: str):
    if not re.search(
        r"replace\((?:DOUBLE|BIGINT|INTEGER|REAL|DECIMAL|HUGEINT|SMALLINT|FLOAT)",
        error,
        re.IGNORECASE,
    ):
        return None
    fixed = re.sub(
        r'CAST\(\s*REPLACE\(\s*("?[A-Za-z_][A-Za-z0-9_]*"?)\s*,[^()]*\)\s*AS\s+\w+\s*\)',
        r"\1",
        sql,
        flags=re.IGNORECASE,
    )
    return fixed if fixed != sql else None


def run_query(db_path: str, sql: str):
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def summarize(llm, question: str, result_preview: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM),
            ("human", "Pergunta: {question}\n\nResultado:\n{result}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "result": result_preview}).strip()


def answer_question(
    llm,
    db_path: str,
    schema_text: str,
    dictionary_text: str,
    question: str,
    max_attempts: int = 5,
) -> dict:
    columns, tables = schema_names(db_path)
    cols_by_table = columns_by_table(db_path)

    def fix(sql: str) -> str:
        return correct_qualified(
            correct_identifiers(sql, columns, tables), cols_by_table, tables
        )

    sql = fix(generate_sql(llm, schema_text, dictionary_text, question))
    last_error = ""

    for _ in range(max_attempts):
        if not is_read_only(sql):
            sql = fix(
                repair_sql(
                    llm,
                    schema_text,
                    sql,
                    "A resposta não era uma consulta SELECT pura. Devolva APENAS a consulta SELECT, sem nenhum texto ao redor.",
                )
            )
            if not is_read_only(sql):
                continue
        try:
            df = run_query(db_path, sql)
        except Exception as exc:
            last_error = str(exc)
            deterministic = (
                repair_from_error(sql, last_error)
                or repair_numeric_cast(sql, last_error)
                or repair_cast(sql, last_error)
            )
            if deterministic:
                sql = fix(deterministic)
            else:
                sql = fix(repair_sql(llm, schema_text, sql, last_error))
            continue

        preview = (
            df.head(30).to_string(index=False)
            if not df.empty
            else "consulta sem resultados"
        )
        answer = summarize(llm, question, preview)
        return {"sql": sql, "dataframe": df, "answer": answer, "error": None}

    final_error = last_error or "a consulta gerada não era uma leitura válida"
    return {
        "sql": sql,
        "dataframe": None,
        "answer": f"Não foi possível responder com os dados carregados após várias tentativas. Detalhe técnico: {final_error}",
        "error": "falha_execucao",
    }
