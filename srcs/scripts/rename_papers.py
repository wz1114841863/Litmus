import sys
from pathlib import Path
import os
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config


def sanitize_filenames():
    """Scan the PDF directory and rename files to replace spaces with underscores.
    This helps avoid issues with file handling in various scripts, and shoudl execute before data_ingestion.py .
    """
    print("--- Starting PDF Filename Sanitization ---")
    pdf_directory = config.PDF_DIR

    if not pdf_directory.exists():
        print(f"Error: PDF directory not found at '{pdf_directory}'")
        return

    all_pdfs = list(pdf_directory.rglob("*.pdf"))

    renamed_count = 0
    for pdf_path in all_pdfs:
        if " " in pdf_path.name:
            new_name = pdf_path.name.replace(" ", "_")
            new_path = pdf_path.with_name(new_name)

            try:
                pdf_path.rename(new_path)
                print(f"  Renamed: '{pdf_path.name}' -> '{new_path.name}'")
                renamed_count += 1
            except OSError as e:
                print(f"  Error renaming '{pdf_path.name}': {e}")

    print("\n--- Sanitization Complete ---")
    if renamed_count > 0:
        print(f"Successfully renamed {renamed_count} files.")
    else:
        print("All filenames are already clean (no spaces found).")


if __name__ == "__main__":
    sanitize_filenames()
