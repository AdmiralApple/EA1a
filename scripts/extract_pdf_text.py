from pypdf import PdfReader


def main():
    input_path = "assignment.pdf"
    output_path = "assignment_extracted.txt"
    reader = PdfReader(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            f.write(f"\n--- Page {i} ---\n")
            f.write(text)
    print(f"Wrote {len(reader.pages)} pages to {output_path}")


if __name__ == "__main__":
    main()