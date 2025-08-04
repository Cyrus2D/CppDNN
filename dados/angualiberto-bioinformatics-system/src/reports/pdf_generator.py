from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
from typing import Dict

def generate_pdf_report(data: Dict, filename: str) -> str:
    """Generates a PDF report based on the provided data."""
    pdf_path = f"./reports/{filename}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("Bioinformatics Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Metadata
    metadata = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    story.append(metadata)
    story.append(Spacer(1, 12))

    # Data Table
    if 'analysis_data' in data:
        table_data = [['Parameter', 'Value']]
        for key, value in data['analysis_data'].items():
            table_data.append([key, str(value)])
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    return pdf_path