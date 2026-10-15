"""
Loading and normalization of the Transparency Portal CSVs.

Handles the portal format's quirks:
  - separator ";" OR "," (auto-detected per file)
  - encoding latin-1 OR utf-8 (auto-detected per file)
  - Brazilian number format ("1.234.567,89")

The portal is inconsistent: the Expense and Revenue exports can come with
different separators and encodings (e.g. expenses as ";"/latin-1 and
revenues as ","/utf-8). So instead of assuming a single fixed format, the
loader sniffs both the encoding and the separator for each file.

Exposes two main functions:
  - load_expenses() -> normalized DataFrame
  - load_revenues() -> normalized DataFrame

Normalization renames the columns the benchmark uses to short, stable
names so the rest of the code does not depend on the original headers
(which differ between expenses and revenues).
"""

from pathlib import Path

import pandas as pd

import config
from benchmark.logging_setup import log


def _detect_encoding(path: Path) -> str:
    """
    Pick the encoding that reads the header without mojibake.

    Tries utf-8 first (strict): if the bytes are valid utf-8 it is almost
    certainly utf-8. Otherwise falls back to latin-1, which never fails
    but would turn real utf-8 bytes into garbage like 'ÃRGÃO'.
    """
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            with open(path, encoding=encoding) as fh:
                fh.readline()
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _detect_separator(header_line: str) -> str:
    """
    Choose ';' or ',' by counting occurrences outside quoted fields.

    Portal number values use a decimal comma and are usually quoted, so
    counting separators outside quotes avoids mistaking a decimal comma
    for a field separator.
    """
    outside = []
    in_quotes = False
    for ch in header_line:
        if ch == '"':
            in_quotes = not in_quotes
        elif not in_quotes:
            outside.append(ch)
    outside_text = "".join(outside)
    semicolons = outside_text.count(";")
    commas = outside_text.count(",")
    return ";" if semicolons >= commas else ","


def _parse_br_number(series: pd.Series) -> pd.Series:
    """Convert BR numeric text ('1.234,56') into float (1234.56)."""
    if series.dtype.kind in "if":  # already numeric
        return series
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)   # remove thousands separator
        .str.replace(",", ".", regex=False)  # decimal comma -> dot
        .str.replace(r"[^0-9.\-]", "", regex=True)  # leftovers (R$, spaces)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a portal CSV, auto-detecting its encoding and separator."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Download the matching CSV at "
            f"https://portaldatransparencia.gov.br/download-de-dados "
            f"and save it at that path (see config.py)."
        )

    encoding = _detect_encoding(path)
    with open(path, encoding=encoding) as fh:
        header_line = fh.readline()
    separator = _detect_separator(header_line)
    log.info("  %s -> encoding=%s, separator='%s'", path.name, encoding, separator)

    df = pd.read_csv(
        path,
        sep=separator,
        encoding=encoding,
        dtype=str,          # read everything as text; convert what we need
        keep_default_na=False,
    )
    # Strip quotes and whitespace from column names.
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def load_expenses() -> pd.DataFrame:
    """
    Load Expense Execution already normalized.

    Output columns (short names):
      year_month, top_agency, agency, uf, city, function,
      expense_group, expense_element, economic_category,
      committed, settled, paid
    """
    log.info("Loading expenses from %s ...", config.EXPENSES_FILE.name)
    df = _read_csv(config.EXPENSES_FILE)

    mapping = {
        "Ano e mês do lançamento": "year_month",
        "Nome Órgão Superior": "top_agency",
        "Nome Órgão Subordinado": "agency",
        "UF": "uf",
        "Município": "city",
        "Nome Função": "function",
        "Nome Grupo de Despesa": "expense_group",
        "Nome Elemento de Despesa": "expense_element",
        "Nome Categoria Econômica": "economic_category",
        "Valor Empenhado (R$)": "committed",
        "Valor Liquidado (R$)": "settled",
        "Valor Pago (R$)": "paid",
    }
    df = _rename_existing(df, mapping)

    for column in ("committed", "settled", "paid"):
        if column in df.columns:
            df[column] = _parse_br_number(df[column])

    df = _derive_year_month(df)
    log.info("Expenses loaded: %d rows.", len(df))
    return df


def load_revenues() -> pd.DataFrame:
    """
    Load Revenue Execution already normalized.

    Output columns (short names):
      top_agency, agency, economic_category, source, kind,
      forecast, posted, realized, realized_percent,
      posting_date, fiscal_year
    """
    log.info("Loading revenues from %s ...", config.REVENUES_FILE.name)
    df = _read_csv(config.REVENUES_FILE)

    mapping = {
        "NOME ÓRGÃO SUPERIOR": "top_agency",
        "NOME ÓRGÃO": "agency",
        "CATEGORIA ECONÔMICA": "economic_category",
        "ORIGEM RECEITA": "source",
        "ESPÉCIE RECEITA": "kind",
        "VALOR PREVISTO ATUALIZADO": "forecast",
        "VALOR LANÇADO": "posted",
        "VALOR REALIZADO": "realized",
        "PERCENTUAL REALIZADO": "realized_percent",
        "DATA LANÇAMENTO": "posting_date",
        "ANO EXERCÍCIO": "fiscal_year",
    }
    df = _rename_existing(df, mapping)

    for column in ("forecast", "posted", "realized", "realized_percent"):
        if column in df.columns:
            df[column] = _parse_br_number(df[column])

    log.info("Revenues loaded: %d rows.", len(df))
    return df


def _rename_existing(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename only the mapped columns that actually exist in the file."""
    present = {orig: new for orig, new in mapping.items() if orig in df.columns}
    missing = [orig for orig in mapping if orig not in df.columns]
    if missing:
        log.warning(
            "Expected columns missing in the CSV (ignored): %s",
            ", ".join(missing),
        )
    return df.rename(columns=present)


def _derive_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """Split 'year_month' (e.g. '2024/01') into 'year' and 'month'."""
    if "year_month" not in df.columns:
        return df
    parts = df["year_month"].astype(str).str.extract(r"(\d{4})\D*(\d{1,2})?")
    df["year"] = pd.to_numeric(parts[0], errors="coerce")
    df["month"] = pd.to_numeric(parts[1], errors="coerce")
    return df
