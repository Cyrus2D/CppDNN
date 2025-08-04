# HTML Report Generator for AnguaLiberto Bioinformatics System

from datetime import datetime
from pathlib import Path
from typing import Dict

class HTMLReportGenerator:
    def __init__(self, author: str, version: str, license: str):
        self.author = author
        self.version = version
        self.license = license

    def generate_report(self, data: Dict, filename: str = None) -> str:
        if not filename:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        report_path = Path(f"./reports/{filename}")
        
        html_content = self.create_html_template(data)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)

    def create_html_template(self, data: Dict) -> str:
        timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
        data_section = self.format_data_section(data)

        html_template = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Relatório AnguaLiberto</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f4f4f4;
                }}
                .header {{
                    background-color: #3366CC;
                    color: white;
                    padding: 10px;
                    text-align: center;
                }}
                .content {{
                    margin: 20px 0;
                    padding: 20px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    font-size: 0.8em;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Relatório AnguaLiberto</h1>
                <p>Versão: {self.version} | Autor: {self.author} | Licença: {self.license}</p>
                <p>Data: {timestamp}</p>
            </div>
            <div class="content">
                <h2>Dados da Análise</h2>
                {data_section}
            </div>
            <div class="footer">
                <p>Gerado pelo Sistema AnguaLiberto</p>
            </div>
        </body>
        </html>
        """
        return html_template

    def format_data_section(self, data: Dict) -> str:
        data_section = "<ul>"
        for key, value in data.items():
            data_section += f"<li><strong>{key}:</strong> {value}</li>"
        data_section += "</ul>"
        return data_section