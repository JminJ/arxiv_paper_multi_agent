import wget
import fitz


def paper_pdf_download(http_pdf_path:str, pdf_download_path:str):
    wget.download(url=http_pdf_path, out=pdf_download_path)

def paper_pdf_load(pdf_path:str)->fitz.Document:
    pdf_file = fitz.open(pdf_path)

    return pdf_file
