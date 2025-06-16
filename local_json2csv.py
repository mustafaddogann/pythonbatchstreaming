import os
import re
import json
import argparse
import itertools
from typing import Any, Dict, List, Generator, Tuple
from io import BytesIO, RawIOBase
import ijson

# Try using C backend
try:
    import ijson.backends.c_yajl2 as ijson_backend
except ImportError:
    import ijson.backends.python as ijson_backend
    print("⚠️ Falling back to slower Python backend.")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert large local JSON to CSV")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    parser.add_argument("--nested_path", default="", help="Optional dot path like 'data.items'")
    parser.add_argument("--max_records", type=int, default=None, help="Limit processing to first N records.")
    return parser.parse_args()

def flatten_json(y: Any, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    out = {}
    def flatten(x: Any, name: str = ''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}{sep}")
        elif isinstance(x, list):
            # Only dump lists of non-dictionaries. Lists of dictionaries are handled by expand_rows_generator.
            if not (x and isinstance(x[0], dict)):
                out[name[:-1]] = json.dumps(x)
            else:
                # If it's a list of dicts, it will be expanded, so we don't flatten it here.
                # However, for the initial flattened object, we need to preserve it.
                # This key will be picked up by expand_rows_generator.
                out[name[:-1]] = x 
        else:
            out[name[:-1]] = x
    flatten(y, parent_key)
    return out

def expand_rows_generator(row: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    # Separate base fields from lists of dictionaries to be expanded
    base_row = {k: v for k, v in row.items() if not (isinstance(v, list) and v and isinstance(v[0], dict))}
    lists_to_expand = {k: v for k, v in row.items() if isinstance(v, list) and v and isinstance(v[0], dict)}

    if not lists_to_expand:
        yield base_row
        return

    # Prepare iterables for each list of dictionaries
    # Each iterable will yield flattened versions of the items in the list
    expanded_iterables = []
    for key, items in lists_to_expand.items():
        # Flatten each item in the list and prepend the key
        expanded_items_for_key = [flatten_json(item, f"{key}_") for item in items]
        expanded_iterables.append(expanded_items_for_key)

    # Use itertools.product to get all combinations of expanded items
    # This is still combinatorial, but operates on already flattened small dicts,
    # and combines them in a generator fashion, not building a huge queue.
    # The crucial part is that the input to product are lists of dictionaries,
    # not full rows. This manages memory better.
    for combination in itertools.product(*expanded_iterables):
        current_combined_row = base_row.copy()
        for part_dict in combination:
            current_combined_row.update(part_dict)
        yield current_combined_row

def escape_csv_value(value: Any) -> str:
    if value is None:
        return ''
    # Ensure value is converted to string before attempting replace
    s_value = str(value)
    # Check if value contains comma or double quote, then wrap in quotes and escape
    if ',' in s_value or '"' in s_value or '\n' in s_value: # Added newline for robustness
        return f'"{s_value.replace("\"", "\"\"")}"'
    return s_value # No quotes needed if no special characters

def main():
    args = parse_args()
    ijson_path = f'{args.nested_path}.item' if args.nested_path else 'item'

    print(f"📥 Reading from: {args.input_file}")
    
    # Open file here to ensure it's managed correctly
    with open(args.input_file, 'rb') as f:
        json_iter = ijson_backend.items(f, ijson_path)

        # Generator to flatten each object from ijson
        flattened_items_generator = (flatten_json(obj) for obj in json_iter)

        # Generator to expand each flattened item into multiple rows
        expanded_rows_generator = (row for flat in flattened_items_generator for row in expand_rows_generator(flat))

        # Limit records if specified
        if args.max_records:
            expanded_rows_generator = itertools.islice(expanded_rows_generator, args.max_records)

        try:
            # Get the first row to determine headers
            first_row = next(expanded_rows_generator)
        except StopIteration:
            print("⚠️ No data found in JSON path or max_records limit resulted in no data.")
            return

        headers = list(first_row.keys())
        
        # Chain the first row back into the generator
        all_rows_for_csv = itertools.chain([first_row], expanded_rows_generator)

        print(f"✍️ Writing to: {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8', newline='') as out_file:
            # Write header
            out_file.write(','.join(escape_csv_value(h) for h in headers) + '\n')
            
            row_count = 0
            for row in all_rows_for_csv:
                # Ensure all headers are present in the row dictionary to avoid KeyError
                # and handle missing values gracefully by outputting an empty string.
                line = ','.join(escape_csv_value(row.get(h)) for h in headers) + '\n'
                out_file.write(line)
                row_count += 1

        print(f"✅ Wrote {row_count} rows with {len(headers)} columns to {args.output_file}")

if __name__ == "__main__":
    main()
