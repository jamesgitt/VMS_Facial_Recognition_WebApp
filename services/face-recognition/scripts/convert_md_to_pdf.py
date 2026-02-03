#!/usr/bin/env python3
"""
Convert MODEL_COMPARISON_RESULTS.md to PDF format.
"""

import os
import re
from fpdf import FPDF, XPos, YPos


# Unicode box-drawing to ASCII mapping
BOX_CHAR_MAP = {
    '╔': '+', '╗': '+', '╚': '+', '╝': '+',
    '╠': '+', '╣': '+', '╦': '+', '╩': '+', '╬': '+',
    '═': '=', '║': '|',
    '─': '-', '│': '|',
    '┌': '+', '┐': '+', '└': '+', '┘': '+',
    '├': '+', '┤': '+', '┬': '+', '┴': '+', '┼': '+',
    '★': '*',
}


def sanitize_text(text: str) -> str:
    """Replace Unicode characters with ASCII equivalents."""
    for char, replacement in BOX_CHAR_MAP.items():
        text = text.replace(char, replacement)
    # Replace any remaining non-ASCII characters
    return text.encode('ascii', 'replace').decode('ascii')


class MarkdownPDF(FPDF):
    """Custom PDF class for markdown conversion."""
    
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        pass
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
    def add_title(self, text: str):
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 10, sanitize_text(text))
        self.ln(5)
        
    def add_h2(self, text: str):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.multi_cell(0, 8, sanitize_text(text))
        self.ln(3)
        
    def add_h3(self, text: str):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 7, sanitize_text(text))
        self.ln(2)
        
    def add_paragraph(self, text: str):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        # Handle bold text
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        self.multi_cell(0, 5, sanitize_text(text))
        self.ln(2)
        
    def add_code_block(self, text: str):
        self.set_font('Courier', '', 6)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(0, 0, 0)
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                safe_line = sanitize_text(line[:100])
                self.cell(0, 3.5, safe_line, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(2)
        
    def add_table(self, rows: list):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        
        if not rows:
            return
            
        # Calculate column widths
        num_cols = len(rows[0])
        col_width = (self.w - 20) / num_cols
        
        for i, row in enumerate(rows):
            if i == 0:
                # Header row
                self.set_font('Helvetica', 'B', 9)
                self.set_fill_color(66, 133, 244)
                self.set_text_color(255, 255, 255)
            elif i == 1 and all(c.strip() == '-' * len(c.strip()) or c.strip().replace('-', '').replace('|', '') == '' for c in row):
                continue  # Skip separator row
            else:
                self.set_font('Helvetica', '', 9)
                self.set_fill_color(248, 249, 250) if i % 2 == 0 else self.set_fill_color(255, 255, 255)
                self.set_text_color(0, 0, 0)
            
            for cell in row:
                cell_text = cell.strip()
                # Clean markdown formatting
                cell_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', cell_text)
                cell_text = sanitize_text(cell_text[:30])
                self.cell(col_width, 6, cell_text, border=1, fill=True)
            self.ln()
        self.set_x(self.l_margin)  # Reset x position
        self.ln(3)
        
    def add_bullet(self, text: str):
        self.set_x(self.l_margin)  # Reset to left margin
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        self.multi_cell(0, 5, sanitize_text(f"  - {text}"))


def parse_table(lines: list) -> list:
    """Parse markdown table lines into rows."""
    rows = []
    for line in lines:
        if '|' in line:
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]  # Remove empty cells
            if cells:
                rows.append(cells)
    return rows


def convert_markdown_to_pdf(md_path: str, pdf_path: str):
    """Convert markdown file to PDF."""
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pdf = MarkdownPDF()
    lines = content.split('\n')
    
    i = 0
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []
    
    while i < len(lines):
        line = lines[i]
        
        # Code block handling
        if line.strip().startswith('```'):
            if in_code_block:
                pdf.add_code_block('\n'.join(code_lines))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        # Table handling
        if line.strip().startswith('|') and '|' in line[1:]:
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            rows = parse_table(table_lines)
            if rows:
                pdf.add_table(rows)
            in_table = False
            table_lines = []
            # Don't increment i, process current line
            
        # Headers
        if line.startswith('# '):
            pdf.add_title(line[2:].strip())
        elif line.startswith('## '):
            pdf.add_h2(line[3:].strip())
        elif line.startswith('### '):
            pdf.add_h3(line[4:].strip())
        elif line.startswith('- '):
            pdf.add_bullet(line[2:].strip())
        elif line.startswith('> '):
            pdf.add_paragraph(f"Note: {line[2:].strip()}")
        elif line.strip() == '---':
            pdf.ln(3)
        elif line.strip().startswith('<details>') or line.strip().startswith('</details>') or line.strip().startswith('<summary>'):
            pass  # Skip HTML details tags
        elif line.strip():
            pdf.add_paragraph(line.strip())
            
        i += 1
    
    # Handle any remaining table
    if in_table and table_lines:
        rows = parse_table(table_lines)
        if rows:
            pdf.add_table(rows)
    
    pdf.output(pdf_path)
    print(f"PDF created: {pdf_path}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(script_dir, '..', 'docs')
    
    md_path = os.path.join(docs_dir, 'MODEL_COMPARISON_RESULTS.md')
    pdf_path = os.path.join(docs_dir, 'MODEL_COMPARISON_RESULTS.pdf')
    
    convert_markdown_to_pdf(md_path, pdf_path)
