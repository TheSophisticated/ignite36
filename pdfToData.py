import pdfplumber
import csv

pdf_path = "dataset1.pdf"
csv_path = "output.csv"

FROM_PAGES = 6000
TO_PAGES = 8647

with pdfplumber.open(pdf_path) as pdf:
    total_pages = min(len(pdf.pages), TO_PAGES)
    print(f"Process {TO_PAGES - FROM_PAGES} pages...")


    with open(csv_path, "a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)

        for i,page in enumerate(pdf.pages[FROM_PAGES:TO_PAGES], start = 1):
            text = page.extract_text()

            if text:
                for line in text.split("\n"):
                    row = line.split()
                    writer.writerow(row)

            if(i%50 == 0):
                print(f"Process {i}/{TO_PAGES-FROM_PAGES} pages ... ")

print("Data Extracted Successfully")