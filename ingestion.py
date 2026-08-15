import os
import re
import zipfile
import tempfile
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd

DICTIONARY_HINTS = ("dicion", "dictionary", "dados", "schema", "layout")
DICTIONARY_EXTS = (".txt", ".md", ".csv", ".json", ".xlsx", ".xls")


ID_TOKENS = {
    "cnpj",
    "cpf",
    "chave",
    "codigo",
    "cod",
    "ncm",
    "cfop",
    "cst",
    "cep",
    "id",
    "ie",
    "im",
    "inscricao",
    "numero",
    "protocolo",
    "matricula",
    "nsu",
    "serie",
}

VALUE_TOKENS = {
    "valor",
    "vlr",
    "preco",
    "montante",
    "frete",
    "desconto",
    "imposto",
    "icms",
    "ipi",
    "pis",
    "cofins",
    "tributo",
    "tributos",
    "aliquota",
    "total",
    "quantidade",
    "qtd",
    "qtde",
    "quant",
    "peso",
    "volume",
}

DATE_TOKENS = {"data", "emissao", "vencimento", "competencia", "dt", "datahora"}


def _normalize_column(name, index: int) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_")
    return text or f"col_{index}"


def _tipo_amigavel(duckdb_type: str) -> str:
    t = str(duckdb_type).upper()
    if any(x in t for x in ("CHAR", "TEXT", "STRING")):
        return "texto"
    if any(x in t for x in ("INT", "DECIMAL", "DOUBLE", "REAL", "FLOAT", "NUMERIC", "HUGEINT")):
        return "número"
    if any(x in t for x in ("DATE", "TIME", "TIMESTAMP")):
        return "data"
    return str(duckdb_type).lower()


def _slugify_table(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not stem or stem[0].isdigit():
        stem = f"t_{stem}"
    return stem


def _read_csv_resilient(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.read_csv(path, sep=";", encoding="latin-1", on_bad_lines="skip")


def _looks_like_dictionary(filename: str) -> bool:
    lower = filename.lower()
    if not lower.endswith(DICTIONARY_EXTS):
        return False
    if lower.endswith(".csv"):
        return any(hint in lower for hint in DICTIONARY_HINTS)
    return True


def _read_dictionary(path: str) -> str:
    ext = Path(path).suffix.lower()
    try:
        if ext in (".xlsx", ".xls"):
            frames = pd.read_excel(path, sheet_name=None)
            return "\n\n".join(
                f"[{sheet}]\n{df.to_string(index=False)}"
                for sheet, df in frames.items()
            )
        if ext == ".csv":
            return _read_csv_resilient(path).to_string(index=False)
        for encoding in ("utf-8", "latin-1"):
            try:
                return Path(path).read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
    except Exception:
        return ""
    return ""


def _coerce_date(series):
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    if parsed.notna().mean() >= 0.8:
        return parsed
    return series


def _coerce_value(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    raw = series.astype(str).str.strip()
    cleaned = raw.str.replace(r"[^\d,.-]", "", regex=True)
    cleaned = cleaned.str.replace(".", "", regex=False).str.replace(
        ",", ".", regex=False
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        return numeric
    return series


def process_zip(uploaded_bytes: bytes, work_dir: str | None = None) -> dict:
    work_dir = work_dir or tempfile.mkdtemp(prefix="nfquery_")
    extract_dir = os.path.join(work_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    zip_path = os.path.join(work_dir, "upload.zip")
    with open(zip_path, "wb") as handle:
        handle.write(uploaded_bytes)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    csv_files, dictionary_text = [], ""
    for root, _, files in os.walk(extract_dir):
        for filename in files:
            full = os.path.join(root, filename)
            if filename.lower().endswith(".csv") and not _looks_like_dictionary(
                filename
            ):
                csv_files.append(full)
            elif _looks_like_dictionary(filename):
                text = _read_dictionary(full)
                if text:
                    dictionary_text += f"\n### {filename}\n{text}\n"

    if not csv_files:
        raise ValueError("Nenhum arquivo CSV encontrado no ZIP.")

    db_path = os.path.join(work_dir, "warehouse.duckdb")
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    schema_parts, used_names, quality = [], set(), []
    try:
        for csv_path in csv_files:
            table = _slugify_table(csv_path)
            suffix = 2
            while table in used_names:
                table = f"{_slugify_table(csv_path)}_{suffix}"
                suffix += 1
            used_names.add(table)

            df = _read_csv_resilient(csv_path)
            df.columns = [_normalize_column(c, i) for i, c in enumerate(df.columns)]
            for col in df.columns:
                parts = [p.lower() for p in col.split("_")]
                if any(p in ID_TOKENS for p in parts):
                    df[col] = df[col].astype("string")
                elif any(p in DATE_TOKENS for p in parts):
                    df[col] = _coerce_date(df[col])
                elif any(p in VALUE_TOKENS for p in parts):
                    df[col] = _coerce_value(df[col])
            con.register("staging_df", df)
            con.execute(
                f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM staging_df'
            )
            con.unregister("staging_df")

            columns = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            rows = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            col_lines = "\n".join(f"  - {name} ({dtype})" for name, dtype in columns)
            schema_parts.append(f'Tabela "{table}" ({rows} linhas):\n{col_lines}')

            col_quality = []
            for name, dtype in columns:
                serie = df[name].astype("string")
                ausentes = int((serie.isna() | (serie.str.strip() == "")).sum())
                col_quality.append({
                    "coluna": name,
                    "tipo": _tipo_amigavel(dtype),
                    "ausentes": ausentes,
                    "pct_ausentes": round(100 * ausentes / rows, 1) if rows else 0.0,
                })
            quality.append({"tabela": table, "linhas": rows,
                            "colunas": len(columns), "detalhe": col_quality})
    finally:
        con.close()

    return {
        "db_path": db_path,
        "schema_text": "\n\n".join(schema_parts),
        "dictionary_text": dictionary_text.strip(),
        "tables": sorted(used_names),
        "quality": quality,
        "work_dir": work_dir,
    }
