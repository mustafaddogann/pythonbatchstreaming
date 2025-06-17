import os
import sys
import time

# Add packages directory to path BEFORE any other imports
# This script needs to find dependencies in the 'packages' directory
try:
    # When running from Azure Batch, find packages relative to script location
    app_path = os.path.abspath(sys.argv[0])
    app_dir = os.path.dirname(app_path)
    packages_dir = os.path.join(app_dir, 'packages')
    if os.path.isdir(packages_dir):
        sys.path.insert(0, packages_dir)
        # Also add to system PATH for DLLs
        os.environ["PATH"] = packages_dir + os.pathsep + os.environ["PATH"]
except Exception:
    # Fallback for local execution
    cwd = os.getcwd()
    packages_dir = os.path.join(cwd, 'packages')
    if os.path.isdir(packages_dir) and packages_dir not in sys.path:
        sys.path.insert(0, packages_dir)
        os.environ["PATH"] = packages_dir + os.pathsep + os.environ["PATH"]

# Check if Visual C++ runtime is installed, install if not (Windows only)
if sys.platform == "win32":
    try:
        # Try to import the C backend to see if it works
        import ijson.backends.yajl2_c
        print("Visual C++ runtime is already installed.")
    except ImportError as e:
        print(f"C backend import failed: {e}")
        print("Visual C++ runtime may be missing. Attempting to install...")
        
        # Check if we're running with admin privileges (common in Batch)
        import ctypes
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        except:
            is_admin = False
        
        if is_admin:
            try:
                import subprocess
                import urllib.request
                
                # Download VC++ runtime
                vc_redist_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
                vc_redist_path = os.path.join(os.environ.get('TEMP', '.'), 'vc_redist.x64.exe')
                
                print(f"Downloading Visual C++ runtime from {vc_redist_url}...")
                urllib.request.urlretrieve(vc_redist_url, vc_redist_path)
                
                # Install silently
                print("Installing Visual C++ runtime...")
                result = subprocess.run([vc_redist_path, '/install', '/quiet', '/norestart'], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("Visual C++ runtime installed successfully.")
                    # Clean up
                    try:
                        os.remove(vc_redist_path)
                    except:
                        pass
                else:
                    print(f"Installation failed with code {result.returncode}")
                    print(f"stdout: {result.stdout}")
                    print(f"stderr: {result.stderr}")
                    
            except Exception as install_error:
                print(f"Failed to install Visual C++ runtime: {install_error}")
        else:
            print("Not running with admin privileges, cannot install Visual C++ runtime.")

import io
import csv
import json
import logging
import argparse
import itertools
import requests
from io import BytesIO, RawIOBase
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions
from typing import Any, Dict, List, Iterator, Generator, Tuple
import ijson
import re

# Recommended: Use the faster C backend if available
try:
    import ijson.backends.yajl2_cffi as ijson_backend
    print("Using yajl2_cffi ijson backend.")
    backend_name = "yajl2_cffi"
except ImportError:
    try:
        import ijson.backends.yajl2_c as ijson_backend
        print("Using yajl2_c ijson backend.")
        backend_name = "yajl2_c"
    except ImportError as e:
        print("="*80)
        print("WARNING: ijson C backend not found.")
        print(f"ImportError: {e}")
        print("The high-performance C backend for JSON parsing (yajl) could not be loaded.")
        print("This is likely because the Visual C++ Redistributable is not installed on the Azure Batch node.")
        print("\nPERFORMANCE IMPACT:")
        print("- Your 200MB file will take ~40 minutes instead of ~3-5 minutes")
        print("- Processing rate will be 10-50x slower")
        print("\nTO FIX THIS:")
        print("Add a Start Task to your Azure Batch pool to install Visual C++ Runtime.")
        print("Go to your Batch Pool -> Start Task -> and configure:")
        print("  Command line: cmd /c \"powershell -Command \\\"Invoke-WebRequest -Uri https://aka.ms/vs/17/release/vc_redist.x64.exe -OutFile vc_redist.x64.exe; Start-Process -FilePath .\\vc_redist.x64.exe -ArgumentList '/install', '/quiet', '/norestart' -Wait; Remove-Item .\\vc_redist.x64.exe\\\"\"")
        print("  User identity: Task user (Admin)")
        print("  Wait for success: Enabled")
        print("\nFalling back to slower Python backend...")
        print("="*80)
        
        import ijson.backends.python as ijson_backend
        backend_name = "python"

# Print diagnostic info about the backend
print(f"ijson backend module: {ijson_backend}")
print(f"ijson backend name: {backend_name}")
print(f"ijson backend file: {getattr(ijson_backend, '__file__', 'Unknown')}")

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert large JSON files from Azure Blob Storage to CSV.")
    parser.add_argument("AZURE_STORAGE_CONNECTION_STRING", help="Azure Storage connection string.")
    parser.add_argument("INPUT_CONTAINER_NAME", help="Name of the input container.")
    parser.add_argument("INPUT_BLOB_PATH_PREFIX", help="Full path to the input JSON blob.")
    parser.add_argument("OUTPUT_CONTAINER_NAME", help="Name of the output container.")
    parser.add_argument("OUTPUT_BLOB_PATH_PREFIX", help="Path prefix for the output CSV file.")
    parser.add_argument("NESTED_PATH", nargs='?', default="", help="Optional dot-separated path to a nested array to extract, e.g., 'data.records'.")
    return parser.parse_args()

# ---------- UTILS ----------

def flatten_json(y: Any, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flattens a nested dictionary."""
    out = {}
    def flatten(x: Any, name: str = ''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}{sep}")
        elif isinstance(x, list):
            # To prevent memory issues, we avoid expanding lists here.
            # Lists of simple types will be converted to a string.
            # Lists of dicts should be handled by expand_rows_generator.
            out[name[:-1]] = json.dumps(x)
        else:
            out[name[:-1]] = x
    flatten(y, parent_key)
    return out

def expand_rows_generator(row: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Expands only the first list-of-dicts field in a row into multiple rows.
    All other lists (simple or nested) are serialized into strings.
    Prevents Cartesian explosion by not cross-joining multiple lists.
    """
    base_row = {}
    expandable_list_key = None
    expandable_list = []

    for k, v in row.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and expandable_list_key is None:
            expandable_list_key = k
            expandable_list = v
        else:
            base_row[k] = json.dumps(v) if isinstance(v, list) else v

    if expandable_list_key is None:
        yield base_row
    else:
        for item in expandable_list:
            new_row = base_row.copy()
            flat = flatten_json(item, parent_key=f"{expandable_list_key}_")
            new_row.update(flat)
            yield new_row


def sanitize_filename(filename: str) -> str:
    """Removes illegal characters from a filename."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def escape_csv_value(value: Any) -> str:
    """
    Escapes a value for Snowflake-compatible CSV formatting based on provided working script.
    This uses backslash as the escape character for quotes and backslashes.
    All fields are enclosed in double quotes.
    None values become empty quoted fields "".
    """
    if value is None:
        s_val = ''
    else:
        s_val = str(value)
    
    # Escape backslashes first, then double quotes with a backslash
    s_val = s_val.replace('\\', '\\\\').replace('"', '\\"')
    
    return f'"{s_val}"'

class CsvStreamer(RawIOBase):
    """
    A file-like object that generates CSV data on the fly.
    It takes an iterator of dictionaries and yields byte chunks for CSV.
    """
    def __init__(self, row_iterator: Iterator[Dict[str, Any]], headers: List[str]):
        self.row_iterator = row_iterator
        self.headers = headers
        self._buffer = BytesIO()
        self._write_header = True
        self._row_count = 0
        self._bytes_written = 0

    def readable(self):
        return True

    def _write_to_internal_buffer(self):
        """Writes the next chunk of CSV data to the internal BytesIO buffer."""
        if self._write_header:
            header_line = (','.join(map(escape_csv_value, self.headers)) + '\n').encode('utf-8')
            self._buffer.write(header_line)
            self._bytes_written += len(header_line)
            self._write_header = False

        try:
            row = next(self.row_iterator)
            line = ','.join(escape_csv_value(row.get(h)) for h in self.headers) + '\n'
            line_bytes = line.encode('utf-8')
            self._buffer.write(line_bytes)
            self._bytes_written += len(line_bytes)
            self._row_count += 1
        except StopIteration:
            pass # No more rows

    def read(self, n=-1):
        """Reads up to n bytes from the stream."""
        if self._buffer.tell() == self._buffer.getbuffer().nbytes: # If buffer is empty or fully consumed
            self._buffer = BytesIO() # Reset buffer
            self._write_to_internal_buffer()
            if self._buffer.tell() == 0: # If nothing was written to buffer, means iterator is exhausted
                return b'' # End of stream

        # Read from internal buffer
        self._buffer.seek(0)
        chunk = self._buffer.read(n if n != -1 else self._buffer.getbuffer().nbytes)
        # Shift remaining data to the beginning of the buffer for next read
        remaining = self._buffer.read()
        self._buffer = BytesIO(remaining)
        return chunk
    
    def get_row_count(self):
        return self._row_count
    
    def get_bytes_written(self):
        return self._bytes_written


class ChunkedCsvStreamer:
    """
    Manages streaming CSV data in chunks, creating new streamers when size threshold is reached.
    """
    def __init__(self, row_iterator: Iterator[Dict[str, Any]], headers: List[str], chunk_threshold_bytes: int):
        self.row_iterator = row_iterator
        self.headers = headers
        self.chunk_threshold_bytes = chunk_threshold_bytes
        self.current_streamer = None
        self.total_rows = 0
        self.chunk_number = 0
        self._exhausted = False
        self._peeked_row = None
        
    def get_next_chunk_streamer(self) -> Tuple[CsvStreamer, bool]:
        """
        Returns a tuple of (streamer, is_last_chunk).
        The streamer will stop when it reaches the chunk threshold or runs out of data.
        """
        if self._exhausted:
            return None, True
            
        self.chunk_number += 1
        
        # Create a limited iterator that stops at chunk threshold
        def limited_iterator():
            bytes_in_chunk = 0
            rows_in_chunk = 0
            
            # If we have a peeked row from previous chunk, yield it first
            if self._peeked_row is not None:
                yield self._peeked_row
                self.total_rows += 1
                rows_in_chunk += 1
                self._peeked_row = None
            
            # Process rows in batches to reduce overhead
            batch_size = 1000  # Process 1000 rows at a time before checking size
            
            for row in self.row_iterator:
                yield row
                self.total_rows += 1
                rows_in_chunk += 1
                
                # Only check size every batch_size rows to reduce overhead
                if rows_in_chunk % batch_size == 0:
                    # Estimate bytes based on average row size
                    # Assume average row is ~1KB (this is a rough estimate)
                    estimated_bytes = rows_in_chunk * 1024
                    
                    if estimated_bytes >= self.chunk_threshold_bytes:
                        print(f"  Chunk {self.chunk_number} reached size limit with {rows_in_chunk} rows")
                        return
            
            # If we get here, iterator is exhausted
            self._exhausted = True
            print(f"  Chunk {self.chunk_number} is final chunk with {rows_in_chunk} rows")
        
        streamer = CsvStreamer(limited_iterator(), self.headers)
        return streamer, self._exhausted
    
    def get_total_rows(self):
        return self.total_rows


def main():
    """
    Main function to orchestrate the JSON to CSV conversion process.
    """
    start_time = time.time()
    
    print(f"--- Running script version 1.7: Performance Optimized ---")
    print(f"Script started at: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"ijson backend: {backend_name}")
    
    args = parse_args()

    # Define chunk size for output files (e.g., 150MB)
    CHUNK_THRESHOLD_BYTES = 150 * 1024 * 1024

    try:
        blob_service = BlobServiceClient.from_connection_string(args.AZURE_STORAGE_CONNECTION_STRING)
        input_blob_client = blob_service.get_blob_client(container=args.INPUT_CONTAINER_NAME, blob=args.INPUT_BLOB_PATH_PREFIX)

        # Generate a SAS token to stream the blob directly with the `requests` library.
        # This bypasses a bug in the Azure SDK's streaming downloader and allows true streaming.
        print("Generating SAS token for direct download.")
        sas_start = time.time()
        sas_token = generate_blob_sas(
            account_name=input_blob_client.account_name,
            container_name=input_blob_client.container_name,
            blob_name=input_blob_client.blob_name,
            account_key=blob_service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        blob_url_with_sas = f"{input_blob_client.url}?{sas_token}"
        print(f"SAS token generated in {time.time() - sas_start:.2f} seconds")

        print(f"Starting blob download and processing...")
        download_start = time.time()
        
        with requests.get(blob_url_with_sas, stream=True) as r:
            r.raise_for_status()
            print(f"HTTP response received in {time.time() - download_start:.2f} seconds")

            # Get blob size from Content-Length header to determine if we need to chunk.
            input_blob_size = int(r.headers.get('Content-Length', 0))
            if input_blob_size > 0:
                print(f"Input blob size: {input_blob_size / (1024*1024):.2f} MB")
            else:
                print("Warning: Could not determine input blob size from header. Assuming small file for single CSV output.")

            ijson_path = f'{args.NESTED_PATH}.item' if args.NESTED_PATH else 'item'
            
            # Use larger buffer for better performance
            buffer_size = 4 * 1024 * 1024  # 4MB buffer
            print(f"Using ijson with {buffer_size / (1024*1024):.1f}MB buffer")
            
            json_parse_start = time.time()
            json_iterator = ijson_backend.items(r.raw, ijson_path, buf_size=buffer_size)

            # Create a processing pipeline using generators
            def expanded_generator():
                item_count = 0
                expand_count = 0
                for obj in json_iterator:
                    item_count += 1
                    if item_count % 10000 == 0:
                        elapsed = time.time() - json_parse_start
                        print(f"  Processed {item_count} JSON items in {elapsed:.1f}s ({item_count/elapsed:.0f} items/sec)")
                    
                    # In this version, we only expand top-level list-of-dicts
                    expanded_rows = [obj]
                    list_fields_to_expand = {k for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict)}

                    if list_fields_to_expand:
                        expand_count += 1
                        base_row = {k: v for k, v in obj.items() if k not in list_fields_to_expand}
                        # Assumption: expand only the first found list-of-dicts field
                        field_to_expand = list(list_fields_to_expand)[0]
                        expanded_rows = [{**base_row, **sub_dict} for sub_dict in obj[field_to_expand]]
                    
                    for row in expanded_rows:
                        yield row
                
                print(f"JSON parsing complete: {item_count} items processed, {expand_count} expanded")

            expanded_row_iterator = expanded_generator()

            try:
                first_row = next(expanded_row_iterator)
                headers = list(first_row.keys())
                print(f"CSV headers ({len(headers)} columns): {headers[:5]}..." if len(headers) > 5 else f"CSV headers: {headers}")
            except StopIteration:
                print("Warning: JSON stream was empty, no CSV file created.")
                return

            full_row_iterator = itertools.chain([first_row], expanded_row_iterator)

            # --- UPLOAD LOGIC ---
            if input_blob_size < CHUNK_THRESHOLD_BYTES:
                # Original logic: process all at once for smaller files
                print("Input blob is smaller than threshold. Processing as a single CSV.")
                output_blob_name = os.path.basename(args.INPUT_BLOB_PATH_PREFIX).replace('.json', '.csv')
                output_blob_path = os.path.join(os.path.dirname(args.INPUT_BLOB_PATH_PREFIX), output_blob_name)
                
                # Create an instance of our streaming CSV writer
                csv_streamer = CsvStreamer(full_row_iterator, headers)

                # Upload the CSV data directly from the CsvStreamer
                output_blob_client = blob_service.get_blob_client(container=args.INPUT_CONTAINER_NAME, blob=output_blob_path)
                
                print(f"Uploading processed CSV to: {output_blob_path}")
                upload_start = time.time()
                output_blob_client.upload_blob(csv_streamer, overwrite=True, content_settings=ContentSettings(content_type='text/csv'), max_concurrency=8)
                upload_end = time.time()
                print("Upload complete.")
                
                end_time = time.time()
                print(f"Upload time: {upload_end - upload_start:.2f} seconds")
                print(f"Total processing time: {end_time - start_time:.2f} seconds")
                print(f"Rows processed: {csv_streamer.get_row_count()}")
                print(f"Processing rate: {csv_streamer.get_row_count() / (end_time - start_time):.1f} rows/second")
            else:
                # New streaming chunk logic for large files
                print(f"Input blob is large. Using streaming chunked CSV output (chunk size: {CHUNK_THRESHOLD_BYTES / (1024*1024):.2f} MB).")
                output_blob_basename = os.path.basename(args.INPUT_BLOB_PATH_PREFIX).replace('.json', '')
                output_blob_dir = os.path.dirname(args.INPUT_BLOB_PATH_PREFIX)
                
                chunked_streamer = ChunkedCsvStreamer(full_row_iterator, headers, CHUNK_THRESHOLD_BYTES)
                
                while True:
                    chunk_streamer, is_last = chunked_streamer.get_next_chunk_streamer()
                    if chunk_streamer is None:
                        break
                        
                    output_blob_name = f"{output_blob_basename}_part_{chunked_streamer.chunk_number:04d}.csv"
                    output_blob_path = os.path.join(output_blob_dir, output_blob_name)
                    output_blob_client = blob_service.get_blob_client(container=args.INPUT_CONTAINER_NAME, blob=output_blob_path)
                    
                    print(f"Streaming chunk {chunked_streamer.chunk_number} to {output_blob_path}")
                    chunk_start = time.time()
                    output_blob_client.upload_blob(chunk_streamer, overwrite=True, content_settings=ContentSettings(content_type='text/csv'), max_concurrency=8)
                    chunk_end = time.time()
                    print(f"Chunk {chunked_streamer.chunk_number} upload complete in {chunk_end - chunk_start:.2f} seconds ({chunk_streamer.get_row_count()} rows, {chunk_streamer.get_bytes_written() / (1024*1024):.2f} MB)")
                    
                    if is_last:
                        break
                
                end_time = time.time()
                print(f"Total processing time: {end_time - start_time:.2f} seconds")
                print(f"Total rows processed: {chunked_streamer.get_total_rows()}")
                if (end_time - start_time) > 0:
                    print(f"Processing rate: {chunked_streamer.get_total_rows() / (end_time - start_time):.1f} rows/second")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

