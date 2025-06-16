import json
import os
from azure.storage.blob import BlobServiceClient
from io import BytesIO

def log_message(message: str):
    with open("log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

def log_error(message: str):
    with open("error.txt", "a", encoding="utf-8") as error_file:
        error_file.write(message + "\n")

def split_json_by_size_azure(
    connection_string: str,
    input_container: str,
    input_blob_path: str,
    output_container: str,
    output_blob_prefix: str,
    nested_path: str,
    max_bytes: int = 25 * 1024 * 1024  # Default: 25 MB
):
    try:
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        input_blob = blob_service.get_blob_client(container=input_container, blob=input_blob_path)

        log_message(f"Downloading blob: {input_blob_path}")
        blob_stream = input_blob.download_blob().readall()

        try:
            data = json.loads(blob_stream)
        except Exception as e:
            log_error(f"Failed to parse JSON: {str(e)}")
            return

        # Navigate to nested list
        keys = nested_path.split('.')
        d = data
        for k in keys[:-1]:
            d = d.get(k, {})
        nested_list_key = keys[-1]

        nested_list = d.pop(nested_list_key, [])
        log_message(f"Length of nested list at {nested_path}: {len(nested_list)}")

        if not isinstance(nested_list, list) or not nested_list:
            log_message(f"No valid list found at {nested_path}. Nothing to split.")
            return

        chunk = []
        current_size = 0
        file_index = 1

        for item in nested_list:
            item_str = json.dumps(item, ensure_ascii=False)
            item_size = len(item_str.encode('utf-8'))

            if current_size + item_size > max_bytes and chunk:
                out_json = data.copy()
                dd = out_json
                for k in keys[:-1]:
                    dd = dd[k]
                dd[nested_list_key] = chunk

                out_blob_name = f"{output_blob_prefix}_part_{file_index}.json"
                output_blob = blob_service.get_blob_client(container=output_container, blob=out_blob_name)
                output_blob.upload_blob(json.dumps(out_json, ensure_ascii=False, indent=2), overwrite=True)

                log_message(f"Wrote blob: {out_blob_name} ~{current_size / (1024 * 1024):.2f} MB, {len(chunk)} items")

                chunk = []
                current_size = 0
                file_index += 1

            chunk.append(item)
            current_size += item_size

        if chunk:
            out_json = data.copy()
            dd = out_json
            for k in keys[:-1]:
                dd = dd[k]
            dd[nested_list_key] = chunk

            out_blob_name = f"{output_blob_prefix}_part_{file_index}.json"
            output_blob = blob_service.get_blob_client(container=output_container, blob=out_blob_name)
            output_blob.upload_blob(json.dumps(out_json, ensure_ascii=False, indent=2), overwrite=True)

            log_message(f"Wrote blob: {out_blob_name} ~{current_size / (1024 * 1024):.2f} MB, {len(chunk)} items")

        log_message(f"Completed. Total files written: {file_index}")
    except Exception as e:
        log_error(f"Unhandled error: {str(e)}")

