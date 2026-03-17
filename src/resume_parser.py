import pdfplumber

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + " "
    except Exception as e:
        print("Error reading PDF:", e)
    
    return text