import os
import json
import csv
import re
from typing import Any, Dict, List, Tuple
from azure.storage.blob import BlobServiceClient
from io import BytesIO, StringIO
from dotenv import load_dotenv

load_dotenv()

# ========== ENV VARIABLES ==========
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
INPUT_CONTAINER_NAME = os.getenv("INPUT_CONTAINER_NAME")
INPUT_BLOB_PATH_PREFIX = os.getenv("INPUT_BLOB_PATH_PREFIX")
OUTPUT_CONTAINER_NAME = os.getenv("OUTPUT_CONTAINER_NAME")
OUTPUT_BLOB_PATH_PREFIX = os.getenv("OUTPUT_BLOB_PATH_PREFIX", "")
NESTED_PATH = os.getenv("NESTED_PATH", "").strip()

# ========== UTILS ==========

def flatten_json(y: Any, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    out = {}

    def flatten(x: Any, name: str = ''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}{sep}")
        elif isinstance(x, list):
            if all(isinstance(i, dict) for i in x):
                return
            else:
                for i, item in enumerate(x):
                    flatten(item, f"{name}{i}{sep}")
        else:
            out[name[:-1]] = x

    flatten(y, parent_key)
    return out

def expand_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_rows = [{}]
    for key, value in data.items():
        if isinstance(value, list) and all(isinstance(i, dict) for i in value):
            new_rows = []
            for row in base_rows:
                for item in value:
                    new_row = row.copy()
                    new_row.update(flatten_json(item, f"{key}_"))
                    new_rows.append(new_row)
            base_rows = new_rows
        else:
            for row in base_rows:
                row[key] = value
    return base_rows

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def extract_nested_rows(json_data: Any, nested_path: str) -> List[Dict[str, Any]]:
    def traverse(data: Any, path_parts: List[str], parents: List[Tuple[Dict[str, Any], str]], current_path: List[str]) -> List[Dict[str, Any]]:
        if not path_parts:
            full_prefix = '_'.join(current_path)
            if isinstance(data, list):
                rows = []
                for item in data:
                    row = {}
                    for parent_obj, parent_path in parents:
                        row.update(flatten_json(parent_obj, parent_path + '_'))
                    row.update(flatten_json(item, full_prefix + '_'))
                    rows.append(row)
                return rows
            elif isinstance(data, dict):
                row = {}
                for parent_obj, parent_path in parents:
                    row.update(flatten_json(parent_obj, parent_path + '_'))
                row.update(flatten_json(data, full_prefix + '_'))
                return [row]
            else:
                return []

        key = path_parts[0]
        if isinstance(data, dict):
            value = data.get(key)
            new_parents = parents + [(data, '_'.join(current_path))]
            return traverse(value, path_parts[1:], new_parents, current_path + [key])
        elif isinstance(data, list):
            rows = []
            for item in data:
                rows.extend(traverse(item, path_parts, parents, current_path))
            return rows
        return []

    return traverse(json_data, nested_path.split('.'), [], [])

def write_csv_to_blob(blob_service: BlobServiceClient, container: str, blob_path: str, headers: List[str], rows: List[Dict[str, Any]]):
    output_stream = StringIO()

    def escape_with_backslash(value: Any) -> str:
        if value is None:
            return '""'
        val = str(value)
        val = val.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{val}"'

    output_stream.write(','.join([f'"{h}"' for h in headers]) + '\n')
    for row in rows:
        output_stream.write(','.join([escape_with_backslash(row.get(h, "")) for h in headers]) + '\n')

    output_stream.seek(0)
    byte_data = output_stream.getvalue().encode('utf-8')

    blob_client = blob_service.get_blob_client(container=container, blob=blob_path)
    blob_client.upload_blob(byte_data, overwrite=True)

    print(f"Wrote CSV: {blob_path} with {len(headers)} columns and {len(rows)} rows.")
    
def build_ordered_headers(rows: List[Dict[str, Any]], reference_order: List[str] = None) -> List[str]:
    seen = set()
    ordered_headers = []

    if reference_order:
        for key in reference_order:
            if key not in seen:
                seen.add(key)
                ordered_headers.append(key)

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered_headers.append(key)

    return ordered_headers

def main():
    if not AZURE_STORAGE_CONNECTION_STRING or not INPUT_CONTAINER_NAME or not INPUT_BLOB_PATH_PREFIX or not OUTPUT_CONTAINER_NAME:
        raise ValueError("Missing one or more required environment variables.")

    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service.get_blob_client(container=INPUT_CONTAINER_NAME, blob=INPUT_BLOB_PATH_PREFIX)

    blob_data = blob_client.download_blob().readall()
    try:
        raw_data = json.loads(blob_data)
    except Exception as e:
        print(f"❌ Failed to load JSON from {INPUT_BLOB_PATH_PREFIX}: {e}")
        return

    if isinstance(raw_data, list):
        records = raw_data
    elif isinstance(raw_data, dict):
        records = []
        for value in raw_data.values():
            if isinstance(value, list):
                records = value
                break
        if not records:
            records = [raw_data]
    else:
        print(f"Unexpected format in {INPUT_BLOB_PATH_PREFIX}")
        return

    # Processing path
    if NESTED_PATH:
        nested_rows = extract_nested_rows(raw_data, NESTED_PATH)
        if not nested_rows:
            print(f"No valid nested rows found at path: {NESTED_PATH}")
            return

        reference_keys = list(flatten_json(raw_data).keys())
        headers = build_ordered_headers(nested_rows, reference_order=reference_keys)

        base_input_name = os.path.basename(INPUT_BLOB_PATH_PREFIX)
        base_name_without_ext = os.path.splitext(base_input_name)[0]

        match = re.search(r'(.*)_((\d{4}-\d{2}-\d{2}))$', base_name_without_ext)
        if match:
            input_name_part = match.group(1)
            date_suffix = match.group(2)
        else:
            input_name_part = base_name_without_ext
            date_suffix = ""

        nested_part = sanitize_filename(NESTED_PATH)
        file_parts = [input_name_part, nested_part]
        if date_suffix:
            file_parts.append(date_suffix)
        csv_filename = "_".join(file_parts) + ".csv"

        output_path = os.path.join(OUTPUT_BLOB_PATH_PREFIX, csv_filename)
        write_csv_to_blob(blob_service, OUTPUT_CONTAINER_NAME, output_path, headers, nested_rows)

    else:
        flat_rows = []
        reference_keys = []
        for record in records:
            reference_keys = list(flatten_json(record).keys())
            break

        for record in records:
            flat = flatten_json(record)
            expanded = expand_rows(flat)
            flat_rows.extend(expanded)

        if not flat_rows:
            print(f"No valid flat rows found.")
            return

        headers = build_ordered_headers(flat_rows, reference_order=reference_keys)
        base_filename = os.path.splitext(os.path.basename(INPUT_BLOB_PATH_PREFIX))[0]
        csv_filename = base_filename + ".csv"
        output_path = os.path.join(OUTPUT_BLOB_PATH_PREFIX, csv_filename)
        write_csv_to_blob(blob_service, OUTPUT_CONTAINER_NAME, output_path, headers, flat_rows)

if __name__ == "__main__":
    main()
