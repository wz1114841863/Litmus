import os
import sys
import pathlib
import json
from lxml import etree
from io import BytesIO

from grobid_client.grobid_client import GrobidClient

GROBID_URL = "http://localhost:8070"
TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}


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


def parse_xml(path):
    """Parse the GROBID TEI XML file and return the root element."""
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    return etree.parse(path, parser)


def parse_grobid_content(content):
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    else:
        content_bytes = content
    return etree.parse(BytesIO(content_bytes), parser)


def get_title(tree):
    """Extract the title from the GROBID TEI XML tree."""
    res = tree.xpath(
        "//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title[@type='main' or @level='a']/text()",
        namespaces=NS,
    )
    if not res:
        res = tree.xpath(
            "//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:analytic//tei:title/text()",
            namespaces=NS,
        )
    if res:
        return res[0].strip()
    return None


def get_authors(tree):
    authors = []
    author_nodes = tree.xpath(
        "//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:analytic//tei:author",
        namespaces=NS,
    )
    if not author_nodes:
        author_nodes = tree.xpath(
            "//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:author | //tei:author",
            namespaces=NS,
        )

    for author in author_nodes:
        firstname = author.xpath(".//tei:forename/text()", namespaces=NS)
        surname = author.xpath(".//tei:surname/text()", namespaces=NS)
        full_name = " ".join(name[0] for name in [firstname, surname] if name)
        if full_name:
            authors.append(full_name.strip())
    return authors


def get_abstract(tree):
    nodes = tree.xpath("//tei:teiHeader//tei:profileDesc//tei:abstract", namespaces=NS)
    if not nodes:
        nodes = tree.xpath("//tei:abstract", namespaces=NS)
    if not nodes:
        return None
    texts = []
    for n in nodes:
        paras = n.xpath(".//tei:p//text() | .//text()", namespaces=NS)
        if paras:
            ps = []
            for pnode in n.xpath(".//tei:p", namespaces=NS):
                pt = " ".join(
                    [t.strip() for t in pnode.xpath(".//text()") if t.strip()]
                )
                if pt:
                    ps.append(pt)
            if ps:
                texts.append("\n\n".join(ps))
            else:
                t = " ".join([t.strip() for t in paras if t.strip()])
                if t:
                    texts.append(t)
    return "\n\n".join([t for t in texts if t]) if texts else None


def get_body(tree):
    body_nodes = tree.xpath(
        "/tei:TEI/tei:text/tei:body | //tei:text//tei:body", namespaces=NS
    )
    if not body_nodes:
        return []
    body = body_nodes[0]
    blocks = []

    divs = body.xpath(".//tei:div", namespaces=NS)
    if not divs:
        nodes = body.xpath(".//tei:head | .//tei:p", namespaces=NS)
        for n in nodes:
            tag = etree.QName(n).localname
            txt = " ".join([t.strip() for t in n.xpath(".//text()") if t.strip()])
            if txt:
                blocks.append(
                    {
                        "type": "heading" if tag.lower() in ("head", "title") else "p",
                        "text": txt,
                    }
                )
        return blocks

    for d in divs:
        heads = d.xpath("./tei:head", namespaces=NS)
        for h in heads:
            txt = " ".join([t.strip() for t in h.xpath(".//text()") if t.strip()])
            if txt:
                blocks.append({"type": "heading", "text": txt})
        for p in d.xpath(".//tei:p", namespaces=NS):
            txt = " ".join([t.strip() for t in p.xpath(".//text()") if t.strip()])
            if txt:
                blocks.append({"type": "p", "text": txt})
    return blocks


def extract_all(tree):
    return {
        "title": get_title(tree),
        "abstract": get_abstract(tree),
        "authors": get_authors(tree),
        "body": get_body(tree),
    }


if __name__ == "__main__":
    PDF_PATH = "/home/wz/AI/Litmus/tmp/pdfs/test1.pdf"
    content = process_single_pdf(PDF_PATH)
    with open("/home/wz/AI/Litmus/tmp/grobid_output_test1.xml", "w") as f:
        f.write(content)
    # content = None
    # with open("/home/wz/AI/Litmus/tmp/grobid_output.xml", "rb") as f:
    #     content = f.read()
    tree = parse_grobid_content(content)
    result = extract_all(tree)
    print("\n--- Extracted Metadata ---")
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Authors: {', '.join(result.get('authors', []))}")
    print(f"Abstract: {result.get('abstract', 'N/A')}")
    print(f"Body: {result.get('body', [])}")
