from __future__ import annotations
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from typing import List, Any

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _auth(service_account_json_path: str) -> gspread.Client:
    creds = Credentials.from_service_account_file(service_account_json_path, scopes=SCOPES)
    return gspread.authorize(creds)

def open_sheet(sheet_id: str, service_account_json_path: str):
    gc = _auth(service_account_json_path)
    return gc.open_by_key(sheet_id)

def ensure_worksheet(spreadsheet, title: str, headers: List[str]):
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=max(10, len(headers)))
    if ws.row_values(1) == [] and headers:
        ws.update("A1", [headers])
    return ws

def clear_and_write(ws, headers: List[str], df: pd.DataFrame):
    ws.clear()
    ws.update("A1", [headers])
    if df.empty:
        return
    ws.update("A2", df[headers].values.tolist())

def append_rows(ws, rows: List[List[Any]]):
    if rows:
        ws.append_rows(rows, value_input_option="USER_ENTERED")

def read_worksheet_df(ws) -> pd.DataFrame:
    """
    Reads entire worksheet into a DataFrame using the first row as headers.
    Returns empty DF if sheet is empty.
    """
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df

def append_df(ws, df: pd.DataFrame, headers: List[str]):
    """
    Appends a dataframe to a worksheet. If worksheet is empty, writes headers first.
    """
    if ws.row_values(1) == []:
        ws.update("A1", [headers])

    if df is None or df.empty:
        return

    # Ensure correct col order + convert NaNs to empty
    df2 = df.copy()
    for h in headers:
        if h not in df2.columns:
            df2[h] = ""
    df2 = df2[headers].fillna("")
    ws.append_rows(df2.values.tolist(), value_input_option="USER_ENTERED")

