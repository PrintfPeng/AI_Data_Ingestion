from backend.services.loader import load_document_bundle
from backend.services.chunking import (
    image_items_to_chunks,
    table_items_to_chunks,
    text_items_to_chunks,
)
from backend.services.vector_store import index_chunks, search_similar


# รายการเอกสารที่อยาก ingest เข้า vector DB
DOCS = [
    ("doc_001", "sample_data/doc_001"),
    ("doc_002", "sample_data/doc_002"),
]


def main():
    all_chunks = []

    print("=== Ingestion: start ===")
    for doc_id, base_dir in DOCS:
        print(f"\n[DOC] {doc_id} from {base_dir}")

        # 1) โหลดเอกสารจาก 4 ไฟล์ (text/table/image/metadata)
        bundle = load_document_bundle(base_dir, doc_id)

        # 2) แปลงเป็น chunks
        text_chunks = text_items_to_chunks(bundle)
        table_chunks = table_items_to_chunks(bundle)
        image_chunks = image_items_to_chunks(bundle)

        doc_chunks = text_chunks + table_chunks + image_chunks

        print(f"  text chunks : {len(text_chunks)}")
        print(f"  table chunks: {len(table_chunks)}")
        print(f"  image chunks: {len(image_chunks)}")
        print(f"  total chunks: {len(doc_chunks)}")

        all_chunks.extend(doc_chunks)

    print(f"\n[SUMMARY] total chunks from all docs: {len(all_chunks)}")

    # 3) index chunks ทั้งหมดเข้า Chroma
    index_chunks(all_chunks)
    print("\nIndexed all chunks into Chroma.")

    # 4) ทดลอง search เบื้องต้น เอกสารละหนึ่งคำถาม
    test_queries = [
        ("ยอดคงเหลือรวมสิ้นงวด", ["doc_001"]),
        ("ยอดที่ต้องชำระทั้งหมด", ["doc_002"]),
    ]

    for query, doc_ids in test_queries:
        print("\n" + "=" * 60)
        print(f"Test search with query: {query!r} (doc_ids={doc_ids})")

        docs = search_similar(query=query, k=3, doc_ids=doc_ids)

        if not docs:
            print("  -> No results")
            continue

        for i, doc in enumerate(docs, start=1):
            print(f"\nResult #{i}")
            print("  content :", doc.page_content)
            print("  metadata:", doc.metadata)

    print("\n=== Ingestion: done ===")


if __name__ == "__main__":
    main()
