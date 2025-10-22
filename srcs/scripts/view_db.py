import sqlite3
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config
from textwrap import shorten

ROWS_PER_PAGE = 20


def schema(conn):
    """print database schema"""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    )
    tables = [row[0] for row in cur.fetchall()]
    for tbl in tables:
        print(f"\n=== {tbl} ===")
        cur.execute(f"PRAGMA table_info({tbl})")
        for cid, name, typ, notnull, dflt, pk in cur.fetchall():
            print(f"  {cid}: {name} {typ} {'PK' if pk else ''}")


def browse_table(conn, table, limit=50, offset=0):
    """print rows from a table with pagination"""
    cur = conn.execute(f"SELECT * FROM {table} LIMIT ? OFFSET ?", (limit, offset))
    rows = cur.fetchall()
    if not rows:
        print("No more rows.")
        return

    col_names = [d[0] for d in cur.description]
    print("\t".join(col_names))
    for row in rows:
        cleaned = [shorten(str(cell), 60) for cell in row]
        print("\t".join(cleaned))


def search_papers(conn, keyword=None, year=None, conf=None, limit=50):
    """search papers by keyword, year, conference"""
    sql = "SELECT id, title, authors, conference, year, file_path FROM papers WHERE 1=1"
    params = []

    if keyword:
        sql += " AND (title LIKE ? OR authors LIKE ? OR author_keywords LIKE ?)"
        kw = f"%{keyword}%"
        params.extend([kw, kw, kw])
    if year:
        sql += " AND year=?"
        params.append(year)
    if conf:
        sql += " AND conference LIKE ?"
        params.append(f"%{conf}%")

    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    if not rows:
        print("No matching papers.")
        return
    print("\nid\ttitle\tauthors\tconference\tyear\tfile_path")
    for r in rows:
        print("\t".join(map(str, r)))


def chunks_for_paper(conn, paper_id, limit=10):
    """print chunks for a given paper_id"""
    cur = conn.execute(
        "SELECT chunk_index, chunk_text FROM chunks WHERE paper_id=? ORDER BY chunk_index LIMIT ?",
        (paper_id, limit),
    )
    for idx, txt in cur.fetchall():
        print(f"\n--- chunk {idx} ---")
        print(shorten(txt, 200))


def main():
    parser = argparse.ArgumentParser(
        description="Read-only SQLite viewer for Litmus DB"
    )
    parser.add_argument("--schema", action="store_true", help="show all tables schema")
    parser.add_argument("-t", "--table", help="table name to browse")
    parser.add_argument("-l", "--limit", type=int, default=20, help="rows to display")
    parser.add_argument(
        "-o", "--offset", type=int, default=0, help="offset for pagination"
    )
    parser.add_argument(
        "-s", "--search", help="keyword search in title/authors/keywords"
    )
    parser.add_argument("-y", "--year", type=int, help="filter by year")
    parser.add_argument("-c", "--conf", help="filter by conference (like ICML)")
    parser.add_argument("--chunks", type=int, help="show chunks for given paper_id")

    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)

    try:
        if args.schema:
            return schema(conn)
        if args.chunks:
            return chunks_for_paper(conn, args.chunks, args.limit)
        if args.search or args.year or args.conf:
            return search_papers(conn, args.search, args.year, args.conf, args.limit)
        if args.table:
            return browse_table(conn, args.table, args.limit, args.offset)

        parser.print_help()
    finally:
        conn.close()


if __name__ == "__main__":
    """
    Example usages:
        # 1. check database schema
        python view_db.py --schema

        # 2. check first 10 rows of papers table
        python view_db.py --table papers --limit 10

        # 3. search papers with "transformer" in title
        python view_db.py --search transformer

        # 4. search 2024 HPCA papers
        python view_db.py --year 2024 --conf HPCA
    """
    main()
