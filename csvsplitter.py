import os

def escape_field(field):
    """Escape quotes and backslashes using backslash escaping for ADF compatibility"""
    # Replace backslashes first to avoid double-escaping
    field = field.replace('\\', '\\\\')
    # Then replace quotes with backslash-escaped quotes
    field = field.replace('"', '\\"')
    return f'"{field}"'

def write_csv_row(file_handle, row):
    """Write a CSV row with custom escaping that matches ADF configuration"""
    escaped_row = [escape_field(field) for field in row]
    line = ','.join(escaped_row) + '\n'
    file_handle.write(line)
    return len(line.encode('utf-8'))

def split_csv_by_size(input_csv_path, output_dir, max_bytes=100 * 1024 * 1024):
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_csv_path))[0]

    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
        # Read the input CSV using standard CSV reader
        import csv
        reader = csv.reader(infile)
        header = next(reader)

        chunk_number = 1
        chunk_path = os.path.join(output_dir, f'{base_filename}_{chunk_number}.csv')
        outfile = open(chunk_path, mode='w', newline='', encoding='utf-8')
        
        # Write header
        write_csv_row(outfile, header)
        current_size = outfile.tell()

        for row in reader:
            # Calculate the size this row would add
            row_size = write_csv_row(outfile, row)
            current_size = outfile.tell()

            if current_size >= max_bytes:
                outfile.close()
                chunk_number += 1
                chunk_path = os.path.join(output_dir, f'{base_filename}_{chunk_number}.csv')
                outfile = open(chunk_path, mode='w', newline='', encoding='utf-8')
                # Write header to new chunk
                write_csv_row(outfile, header)
                current_size = outfile.tell()

        outfile.close()

    print(f"✅ Done. {chunk_number} chunk(s) created in '{output_dir}'.")

# Run it
if __name__ == "__main__":
    split_csv_by_size("client_t.csv", "client_t_chunks", max_bytes=100 * 1024 * 1024)
