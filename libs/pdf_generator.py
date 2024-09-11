import re
import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 0, "C")

def sanitize_content(content):
    return content.encode('utf-8', 'ignore').decode('utf-8')

def replace_problematic_characters(content):
    replacements = {
        '\u2013': '-', '\u2014': '--', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2010': '-',
        '\u2022': '*', '\u2122': 'TM'
    }
    return ''.join(replacements.get(c, c) for c in content)

def process_markdown_line(pdf, line):
    if line.startswith('#'):
        header_level = min(line.count('#'), 4)
        header_text = re.sub(r'\*{2,}', '', line.strip('# ').strip())
        pdf.set_font('Arial', 'B', 12 + (4 - header_level) * 2)
        pdf.multi_cell(0, 10, header_text)
        pdf.set_font('Arial', '', 12)
    else:
        parts = re.split(r'(\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*.*?\*|\[.*?\]\(.*?\))', line)
        for part in parts:
            process_markdown_part(pdf, part)
        pdf.ln(10)

def process_markdown_part(pdf, part):
    if re.match(r'\*\*\*.*?\*\*\*', part):
        text = part.strip('*')
        pdf.set_font('Arial', 'BI', 12)
    elif re.match(r'\*\*.*?\*\*', part):
        text = part.strip('*')
        pdf.set_font('Arial', 'B', 12)
    elif re.match(r'\*.*?\*', part):
        text = part.strip('*')
        pdf.set_font('Arial', 'I', 12)
    elif re.match(r'\[.*?\]\(.*?\)', part):
        display_text = re.search(r'\[(.*?)\]', part).group(1)
        url = re.search(r'\((.*?)\)', part).group(1)
        pdf.set_text_color(0, 0, 255)
        pdf.set_font('', 'U')
        pdf.write(10, display_text, url)
    else:
        text = part
    
    if 'text' in locals():
        pdf.write(10, text)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 12)

def generate_pdf_from_md(content, filename='output.pdf'):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font('Arial', '', 12)

        sanitized_content = sanitize_content(content)
        sanitized_content = replace_problematic_characters(sanitized_content)

        for line in sanitized_content.split('\n'):
            process_markdown_line(pdf, line)

        pdf.output(filename)
        return f"PDF generado: {filename}"
    except Exception as e:
        return f"Error al generar el PDF: {str(e)}"

def generate_pdf(content, filename):
    return generate_pdf_from_md(content, filename)