import PyPDF2
import os
os.chdir(f'{os.getcwd()}/pdfMerger')
pdfFiles=os.listdir()
def formatChecker(file):
    file=file.split(".")
    return file[len(file)-1]=="pdf"

print(formatChecker("sample.pdf"))

pdfFiles =list(filter(formatChecker,pdfFiles))

merger = PyPDF2.PdfWriter()

for file in pdfFiles:
    merger.append(file)

merger.write("merged-pdf.pdf")
merger.close()