import os
import sys
import time

# Add packages directory to Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
packages_dir = os.path.join(app_dir, 'packages')
if os.path.exists(packages_dir):
    sys.path.insert(0, packages_dir)
    # Also add to PATH for DLLs
    os.environ["PATH"] = packages_dir + os.pathsep + os.environ.get("PATH", "")

import re
import argparse
import json
from typing import Any, Dict, List, Iterator, Generator, Tuple
from azure.storage.blob import BlobServiceClient, ContentSettings
from io import BytesIO, BufferedReader, RawIOBase
import ijson
import itertools

# Recommended: Use the faster C backend if available
try:
    import ijson.backends.yajl2_c as ijson_backend
    print("SUCCESS: Using fast C backend (yajl2_c)")
except ImportError:
    import ijson.backends.python as ijson_backend
    print("Warning: C backend for ijson not found. Falling back to slower Python backend.")

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
            # Convert lists to JSON strings to prevent memory issues
            out[name[:-1]] = json.dumps(x)
        else:
            out[name[:-1]] = x
    flatten(y, parent_key)
    return out

def expand_rows_generator(row: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Expands only the first list-of-dicts field in a row into multiple rows.
    Optimized to reduce memory usage.
    """
    # First pass: find expandable list
    expandable_list_key = None
    expandable_list = None
    
    for k, v in row.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            expandable_list_key = k
            expandable_list = v
            break
    
    if expandable_list_key is None:
        # No expansion needed - convert any remaining lists to JSON strings
        yield {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
    else:
        # Build base row without the expandable list
        base_row = {k: json.dumps(v) if isinstance(v, list) else v 
                   for k, v in row.items() if k != expandable_list_key}
        
        # Expand the list
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
    Escapes a value for Snowflake/ADF-compatible CSV formatting, based on the provided
    working csvsplitter.py script.
    - Handles None values by creating an empty quoted field: "".
    - Escapes backslashes (\\) with a preceding backslash (\\\\).
    - Escapes double quotes (") with a preceding backslash (\\").
    - Encloses the entire field in double quotes.
    """
    if value is None:
        # For None, return an empty quoted string as per Snowflake requirements.
        return '""'
    
    # Convert the value to its string representation.
    s_val = str(value)
    
    # First, replace backslashes, then double quotes, as in the reference script.
    s_val = s_val.replace('\\', '\\\\').replace('"', '\\"')
    
    return f'"{s_val}"'

class CsvStreamer(RawIOBase):
    """
    A file-like object that generates CSV data on the fly.
    It takes an iterator of dictionaries and yields byte chunks for CSV.
    """
    def __init__(self, row_iterator: Iterator[Dict[str, Any]], headers: List[str], batch_size: int = 10000):
        self.row_iterator = row_iterator
        self.headers = headers
        self._buffer = BytesIO()
        self._write_header = True
        self._row_count = 0
        self._batch_size = batch_size
        self._exhausted = False

    def readable(self):
        return True

    def _write_batch_to_buffer(self):
        """Writes a batch of rows to the internal buffer."""
        if self._write_header:
            header_line = (','.join(map(escape_csv_value, self.headers)) + '\n').encode('utf-8')
            self._buffer.write(header_line)
            self._write_header = False

        # Process a batch of rows
        batch_count = 0
        try:
            while batch_count < self._batch_size:
                row = next(self.row_iterator)
                line = ','.join(escape_csv_value(row.get(h)) for h in self.headers) + '\n'
                self._buffer.write(line.encode('utf-8'))
                self._row_count += 1
                batch_count += 1
        except StopIteration:
            self._exhausted = True

    def read(self, n=-1):
        """Reads up to n bytes from the stream."""
        # If buffer is empty, fill it with a new batch
        if self._buffer.tell() == self._buffer.getbuffer().nbytes:
            if self._exhausted:
                return b''  # No more data
            
            self._buffer = BytesIO()  # Reset buffer
            self._write_batch_to_buffer()
            self._buffer.seek(0)  # Reset to beginning for reading

        # Read from buffer
        if n == -1:
            return self._buffer.read()
        else:
            return self._buffer.read(n)
    
    def get_row_count(self):
        return self._row_count

class AzureBlobStreamWrapper(RawIOBase):
    """
    A wrapper for the Azure Blob Storage stream to make it compatible with
    libraries that expect a standard raw IO stream (like ijson's C backend).

    - Implements readable() which is missing from the Azure stream object.
    - Implements a correct readinto() using the stream's read() method,
      bypassing a bug in some versions of the Azure SDK's readinto().
    """
    def __init__(self, download_stream):
        self._stream = download_stream

    def readable(self):
        return True

    def readinto(self, b):
        """Reads up to len(b) bytes into b, and returns the number of bytes read."""
        chunk = self._stream.read(len(b))
        bytes_read = len(chunk)
        b[:bytes_read] = chunk
        return bytes_read

    def read(self, n=-1):
        return self._stream.read(n)

def main():
    """Main function to orchestrate the JSON to CSV conversion."""
    start_time = time.time()
    
    args = parse_args()
    
    # Performance tuning parameters (can be overridden by environment variables)
    BUFFER_SIZE = int(os.environ.get('JSON2CSV_BUFFER_SIZE', 64 * 1024 * 1024))      # Default 64MB
    CSV_BATCH_SIZE = int(os.environ.get('JSON2CSV_BATCH_SIZE', 10000))                 # Default 10k rows
    PROGRESS_INTERVAL = int(os.environ.get('JSON2CSV_PROGRESS_INTERVAL', 50000))      # Default 50k rows
    CHUNK_TARGET_SIZE_MB = int(os.environ.get('JSON2CSV_CHUNK_TARGET_SIZE_MB', 150))   # Default 150MB
    # Azure SDK tuning
    UPLOAD_CHUNK_SIZE = int(os.environ.get('JSON2CSV_UPLOAD_CHUNK_SIZE', 8 * 1024 * 1024)) # Default 8MB block size
    MAX_SINGLE_PUT_SIZE = int(os.environ.get('JSON2CSV_MAX_SINGLE_PUT_SIZE', 64 * 1024 * 1024)) # Default 64MB threshold for single upload

    print(f"=== Performance Configuration ===")
    print(f"Read Buffer size: {BUFFER_SIZE / (1024 * 1024):.0f} MB")
    print(f"CSV Batch size: {CSV_BATCH_SIZE:,} rows")
    print(f"Progress Interval: {PROGRESS_INTERVAL:,} rows")
    print(f"Upload Block Size: {UPLOAD_CHUNK_SIZE / (1024 * 1024):.0f} MB")
    print(f"Max Single Put Size: {MAX_SINGLE_PUT_SIZE / (1024 * 1024):.0f} MB")
    print(f"CSV Chunk Target Size: {CHUNK_TARGET_SIZE_MB} MB")
    print()
    
    blob_service = BlobServiceClient.from_connection_string(
        args.AZURE_STORAGE_CONNECTION_STRING,
        max_block_size=UPLOAD_CHUNK_SIZE,
        max_single_put_size=MAX_SINGLE_PUT_SIZE
    )

    # Find the blob that matches the prefix
    container_client = blob_service.get_container_client(args.INPUT_CONTAINER_NAME)
    matching_blobs = list(container_client.list_blobs(name_starts_with=args.INPUT_BLOB_PATH_PREFIX))

    if not matching_blobs:
        print(f"No blob found with prefix: {args.INPUT_BLOB_PATH_PREFIX}")
        return
    elif len(matching_blobs) > 1:
        print(f"Multiple blobs found with prefix: {args.INPUT_BLOB_PATH_PREFIX}. Expected only one.")
        return

    blob_name = matching_blobs[0].name
    input_blob_client = container_client.get_blob_client(blob_name)

    print(f"Downloading and processing blob: {blob_name}")

    # Get blob properties to show file size
    blob_properties = input_blob_client.get_blob_properties()
    blob_size_mb = blob_properties.size / (1024 * 1024)
    print(f"Blob size: {blob_size_mb:.2f} MB")
    
    # Get the blob input stream directly. download_blob() returns a BlobStream object
    # which is a file-like object that reads chunks on demand.
    download_stream = input_blob_client.download_blob()

    # Wrap the Azure stream in our custom wrapper, then in a BufferedReader
    # for full compatibility with ijson's C backend.
    raw_wrapper = AzureBlobStreamWrapper(download_stream)
    buffered_stream = BufferedReader(raw_wrapper, buffer_size=BUFFER_SIZE)

    # Determine the ijson path based on NESTED_PATH
    ijson_path = f'{args.NESTED_PATH}.item' if args.NESTED_PATH else 'item'
    
    # Use ijson_backend.items directly with the download_stream
    # This is highly memory-efficient as ijson pulls data as needed.
    print(f"Streaming JSON items from path: {ijson_path}")
    json_iterator = ijson_backend.items(buffered_stream, ijson_path)
    
    # Create a processing pipeline using generators
     # Flatten and expand only top-level list-of-dict fields (no cross-joins)
    def expanded_generator():
        row_count = 0
        parse_start = time.time()
        last_time = parse_start
        
        for obj in json_iterator:
            for row in expand_rows_generator(flatten_json(obj)):
                row_count += 1
                if row_count % PROGRESS_INTERVAL == 0:
                    current_time = time.time()
                    elapsed = current_time - parse_start
                    interval_time = current_time - last_time
                    interval_speed = PROGRESS_INTERVAL / interval_time
                    overall_speed = row_count / elapsed
                    mb_processed = (row_count / num_rows * blob_size_mb) if 'num_rows' in locals() else 0
                    
                    print(f"Processed {row_count:,} rows | Last {PROGRESS_INTERVAL//1000}k: {interval_speed:,.0f} rows/sec | Overall: {overall_speed:,.0f} rows/sec")
                    last_time = current_time
                yield row

    expanded_row_iterator = expanded_generator()

    
    # Get the first row to determine headers
    try:
        first_row = next(expanded_row_iterator)
        headers = list(first_row.keys())
    except StopIteration:
        print("Warning: JSON stream was empty, no CSV file created.")
        return

    # This iterator now contains ALL rows for the entire file.
    full_row_iterator = itertools.chain([first_row], expanded_row_iterator)

    # --- Main Chunking Loop ---
    chunk_number = 1
    total_rows_written = 0
    base_name = os.path.splitext(os.path.basename(args.INPUT_BLOB_PATH_PREFIX))[0]
    nested_part = sanitize_filename(args.NESTED_PATH) if args.NESTED_PATH else ""
    CHUNK_TARGET_SIZE_BYTES = CHUNK_TARGET_SIZE_MB * 1024 * 1024

    # Use a sentinel to cleanly check if the iterator is exhausted
    row_sentinel = object()
    row = next(full_row_iterator, row_sentinel)

    while row is not row_sentinel:
        csv_chunk_filename = f"{base_name}{'_' + nested_part if nested_part else ''}_part_{chunk_number:03}.csv"
        output_path = os.path.join(args.OUTPUT_BLOB_PATH_PREFIX, csv_chunk_filename)
        output_blob_client = blob_service.get_blob_client(container=args.OUTPUT_CONTAINER_NAME, blob=output_path)

        print(f"\nStarting chunk {chunk_number}: Uploading to {output_path}")

        # This generator function will provide rows for exactly one chunk
        def chunk_generator():
            nonlocal row, row_sentinel, full_row_iterator
            bytes_in_chunk = 0
            
            # Loop until the chunk is full or we run out of rows
            while row is not row_sentinel and bytes_in_chunk < CHUNK_TARGET_SIZE_BYTES:
                # Estimate the size of the row before yielding it
                bytes_in_chunk += len((','.join(escape_csv_value(row.get(h)) for h in headers) + '\n'))
                
                # Yield the current row, then grab the next one to check in the next iteration
                current_row = row
                row = next(full_row_iterator, row_sentinel)
                yield current_row

        # Create and upload the stream for this chunk
        csv_streamer = CsvStreamer(chunk_generator(), headers, batch_size=CSV_BATCH_SIZE)
        output_blob_client.upload_blob(
            csv_streamer,
            overwrite=True,
            content_settings=ContentSettings(content_type='text/csv')
        )
        
        rows_in_this_chunk = csv_streamer.get_row_count()
        total_rows_written += rows_in_this_chunk
        print(f"Successfully uploaded chunk {chunk_number} with {rows_in_this_chunk:,} rows.")
        chunk_number += 1
    
    # --- Final Summary ---
    total_time = time.time() - start_time
    print(f"\n=== Job Complete ===")
    print(f"Total rows processed: {total_rows_written:,}")
    print(f"Total columns: {len(headers)}")
    print(f"Input size: {blob_size_mb:.2f} MB")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    if total_time > 0:
        print(f"Processing speed: {total_rows_written/total_time:,.0f} rows/sec")
        print(f"Data throughput: {blob_size_mb/total_time:.2f} MB/sec")
    print(f"Total chunks created: {chunk_number - 1}")
    
if __name__ == "__main__":
    main()