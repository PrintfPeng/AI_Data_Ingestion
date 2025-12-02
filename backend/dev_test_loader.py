from backend.services.loader import load_document_bundle

if __name__ == "__main__":
    bundle = load_document_bundle("sample_data/doc_001", "doc_001")
    print("DOC:", bundle.metadata.doc_id, bundle.metadata.file_name)
    print("texts:", len(bundle.texts))
    print("tables:", len(bundle.tables))
    print("images:", len(bundle.images))

    if bundle.texts:
        print("first text content:", bundle.texts[0].content)
