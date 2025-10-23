import os
import sys
import pathlib
import xml.etree.ElementTree as ET

from grobid_client.grobid_client import GrobidClient

GROBID_URL = "http://localhost:8070"
PDF_PATH = "/home/wz/AI/Litmus/tmp/pdfs/test.pdf"


def process_single_pdf(pdf_file_path: str):
    """Use GROBID to process a single PDF and return the extracted XML content."""
    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file does not exist -> {pdf_file_path}")
        return

    print(f"Start processing file: {pdf_file_path}")
    client = GrobidClient(GROBID_URL)
    content = None
    try:
        status_code, _, content = client.process_pdf(
            "processFulltextDocument",
            pdf_file_path,
            generateIDs=False,
            consolidate_header=True,
            tei_coordinates=False,
            consolidate_citations=False,
            include_raw_citations=False,
            include_raw_affiliations=False,
            segment_sentences=False,
        )

        if status_code == 200:
            print("GROBID extract successfully!")
            print("\n--- content of XML ---")
            print(content[:50] + "...")
        else:
            print(f"GROBID extract failed!")
            print(f"HTTP State code: {status_code}")
            print(f"Server error message: {content}")

    except Exception as e:
        print(f"An unexpected error occurred during the request: {e}")

    print(f"Finished processing file: {pdf_file_path}")
    return content


# TODO: cannot extract information correctly, need to fix it
def parse_grobid_xml(xml_content: str) -> dict:
    """Parse GROBID TEI XML content and extract metadata such as title, authors, and abstract."""
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return {}

    title_element = root.find(".//tei:titleStmt/tei:title", ns)
    title = title_element.text if title_element is not None else "N/A"

    authors_list = []
    author_elements = root.findall(".//tei:analytic/tei:author/tei:persName", ns)
    for author_element in author_elements:
        firstname = author_element.find(".//tei:forename", ns)
        surname = author_element.find(".//tei:surname", ns)

        full_name = []
        if firstname is not None and firstname.text:
            full_name.append(firstname.text)
        if surname is not None and surname.text:
            full_name.append(surname.text)

        if full_name:
            authors_list.append(" ".join(full_name))

    abstract_element = root.find(".//tei:profileDesc/tei:abstract/tei:p", ns)
    abstract = abstract_element.text if abstract_element is not None else "N/A"

    return {"title": title, "authors": authors_list, "abstract": abstract}


if __name__ == "__main__":
    content = process_single_pdf(PDF_PATH)
    with open("/home/wz/AI/Litmus/tmp/grobid_output.xml", "w") as f:
        f.write(content)
    result = parse_grobid_xml(content)
    print("\n--- Extracted Metadata ---")
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Authors: {', '.join(result.get('authors', []))}")
    print(f"Abstract: {result.get('abstract', 'N/A')}")
