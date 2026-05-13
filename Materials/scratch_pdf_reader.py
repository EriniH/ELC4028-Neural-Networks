import PyPDF2

pdf_path = r"d:\EECE4\ELC4028-Neural-Networks\Materials\Assignment 3_V3.pdf"
with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

with open(r"d:\EECE4\ELC4028-Neural-Networks\Materials\pdf_text.txt", "w", encoding="utf-8") as out_file:
    out_file.write(text)
print("PDF text extracted successfully")
