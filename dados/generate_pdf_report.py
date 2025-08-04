#!/usr/bin/env python3
"""
Script para gerar relatórios em PDF/HTML a partir dos dados de análise
do sistema de bioinformática C++.

Requisitos:
pip install pandas matplotlib seaborn jinja2 weasyprint
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from jinja2 import Template
import base64
from io import BytesIO

class BioinformaticsReportGenerator:
    def __init__(self, data_dir="./dados"):
        self.data_dir = data_dir
        self.visualization_dir = os.path.join(data_dir, "visualizacao")
        self.logs_dir = os.path.join(data_dir, "logs")
        self.validation_dir = os.path.join(data_dir, "validacao")
        
    def load_data(self):
        """Carregar dados de impacto IA e validação"""
        try:
            # Carregar dados de impacto
            impact_file = os.path.join(self.visualization_dir, "impact_data.csv")
            if os.path.exists(impact_file):
                self.impact_data = pd.read_csv(impact_file)
            else:
                print(f"Arquivo não encontrado: {impact_file}")
                return False
            
            # Carregar logs de validação
            validation_file = os.path.join(self.validation_dir, "validation_report.txt")
            self.validation_report = ""
            if os.path.exists(validation_file):
                with open(validation_file, 'r', encoding='utf-8') as f:
                    self.validation_report = f.read()
            
            # Carregar relatório de impacto
            impact_report_file = os.path.join(self.logs_dir, "impact_report.txt")
            self.impact_report = ""
            if os.path.exists(impact_report_file):
                with open(impact_report_file, 'r', encoding='utf-8') as f:
                    self.impact_report = f.read()
                    
            return True
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False
    
    def generate_plots(self):
        """Gerar gráficos para o relatório"""
        if not hasattr(self, 'impact_data'):
            return {}
        
        plots = {}
        plt.style.use('seaborn-v0_8')
        
        # Gráfico 1: Distribuição do Impacto IA
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.impact_data['impact_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Distribuição do Impacto IA', fontsize=14, fontweight='bold')
        ax.set_xlabel('Score de Impacto')
        ax.set_ylabel('Frequência')
        ax.grid(True, alpha=0.3)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plots['impact_distribution'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Gráfico 2: GC vs Impacto
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(self.impact_data['gc_content'], self.impact_data['impact_score'], 
                           alpha=0.6, c=self.impact_data['sequence_length'], 
                           cmap='viridis', s=50)
        ax.set_title('Conteúdo GC vs Impacto IA', fontsize=14, fontweight='bold')
        ax.set_xlabel('Conteúdo GC (%)')
        ax.set_ylabel('Score de Impacto')
        ax.grid(True, alpha=0.3)
        
        # Adicionar barra de cores
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Comprimento da Sequência')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plots['gc_vs_impact'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Gráfico 3: Boxplot por Confiança
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=self.impact_data, x='prediction_confidence', y='impact_score', ax=ax)
        ax.set_title('Distribuição do Impacto por Nível de Confiança', fontsize=14, fontweight='bold')
        ax.set_xlabel('Nível de Confiança')
        ax.set_ylabel('Score de Impacto')
        ax.grid(True, alpha=0.3)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plots['confidence_boxplot'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Gráfico 4: Heatmap de Correlação
        if len(self.impact_data.columns) > 3:
            numeric_columns = ['impact_score', 'gc_content', 'sequence_length']
            correlation_matrix = self.impact_data[numeric_columns].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={'label': 'Correlação'})
            ax.set_title('Matriz de Correlação', fontsize=14, fontweight='bold')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plots['correlation_heatmap'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
        
        return plots
    
    def calculate_statistics(self):
        """Calcular estatísticas resumidas"""
        if not hasattr(self, 'impact_data'):
            return {}
        
        stats = {
            'total_sequences': len(self.impact_data),
            'avg_impact': self.impact_data['impact_score'].mean(),
            'std_impact': self.impact_data['impact_score'].std(),
            'max_impact': self.impact_data['impact_score'].max(),
            'min_impact': self.impact_data['impact_score'].min(),
            'avg_gc': self.impact_data['gc_content'].mean(),
            'avg_length': self.impact_data['sequence_length'].mean(),
            'high_confidence_count': len(self.impact_data[self.impact_data['prediction_confidence'] == 'ALTA']),
            'medium_confidence_count': len(self.impact_data[self.impact_data['prediction_confidence'] == 'MÉDIA']),
            'low_confidence_count': len(self.impact_data[self.impact_data['prediction_confidence'] == 'BAIXA'])
        }
        
        return stats
    
    def generate_html_report(self):
        """Gerar relatório em HTML"""
        plots = self.generate_plots()
        stats = self.calculate_statistics()
        
        html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Bioinformática - Análise de Impacto IA</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }
        h3 {
            color: #7f8c8d;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stats-card {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stats-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .stats-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .plot-container {
            text-align: center;
            margin: 30px 0;
            page-break-inside: avoid;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        .report-section {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .timestamp {
            text-align: right;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Relatório de Bioinformática</h1>
        <h2>Análise de Impacto IA - Sequências de Proteínas</h2>
        
        <div class="warning">
            <strong>⚠️ Atenção:</strong> Este relatório inclui análise de códons inválidos encontrados nas sequências.
            Sequências com códons incompletos ou não mapeados (como '?') foram processadas e validadas.
        </div>
        
        <h2>📊 Estatísticas Gerais</h2>
        <div class="stats-grid">
            <div class="stats-card">
                <div class="stats-value">{{ stats.total_sequences }}</div>
                <div class="stats-label">Total de Sequências</div>
            </div>
            <div class="stats-card">
                <div class="stats-value">{{ "%.4f"|format(stats.avg_impact) }}</div>
                <div class="stats-label">Impacto Médio</div>
            </div>
            <div class="stats-card">
                <div class="stats-value">{{ "%.2f"|format(stats.avg_gc) }}%</div>
                <div class="stats-label">GC Médio</div>
            </div>
            <div class="stats-card">
                <div class="stats-value">{{ "%.1f"|format(stats.avg_length) }}</div>
                <div class="stats-label">Comprimento Médio</div>
            </div>
            <div class="stats-card">
                <div class="stats-value">{{ stats.high_confidence_count }}</div>
                <div class="stats-label">Alta Confiança</div>
            </div>
            <div class="stats-card">
                <div class="stats-value">{{ stats.medium_confidence_count }}</div>
                <div class="stats-label">Média Confiança</div>
            </div>
        </div>
        
        <h2>📈 Visualizações</h2>
        
        {% if plots.impact_distribution %}
        <div class="plot-container">
            <h3>Distribuição do Impacto IA</h3>
            <img src="data:image/png;base64,{{ plots.impact_distribution }}" alt="Distribuição do Impacto IA">
        </div>
        {% endif %}
        
        {% if plots.gc_vs_impact %}
        <div class="plot-container">
            <h3>Conteúdo GC vs Impacto IA</h3>
            <img src="data:image/png;base64,{{ plots.gc_vs_impact }}" alt="GC vs Impacto">
        </div>
        {% endif %}
        
        {% if plots.confidence_boxplot %}
        <div class="plot-container">
            <h3>Distribuição por Nível de Confiança</h3>
            <img src="data:image/png;base64,{{ plots.confidence_boxplot }}" alt="Confiança Boxplot">
        </div>
        {% endif %}
        
        {% if plots.correlation_heatmap %}
        <div class="plot-container">
            <h3>Matriz de Correlação</h3>
            <img src="data:image/png;base64,{{ plots.correlation_heatmap }}" alt="Correlação">
        </div>
        {% endif %}
        
        <h2>📋 Relatório de Validação</h2>
        <div class="report-section">{{ validation_report }}</div>
        
        <h2>🔍 Relatório de Impacto IA</h2>
        <div class="report-section">{{ impact_report }}</div>
        
        <div class="warning">
            <strong>🔧 Recomendações:</strong>
            <ul>
                <li>Sequências com códons inválidos (?) foram identificadas e tratadas</li>
                <li>Considere usar o sistema de versionamento para modelos futuros</li>
                <li>Monitore regularmente o impacto IA para detectar mudanças</li>
                <li>Mantenha backup das versões de modelo com melhor performance</li>
            </ul>
        </div>
        
        <div class="timestamp">
            Relatório gerado em: {{ timestamp }}
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            stats=stats,
            plots=plots,
            validation_report=self.validation_report,
            impact_report=self.impact_report,
            timestamp=datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
        )
        
        return html_content
    
    def save_html_report(self, filename=None):
        """Salvar relatório HTML"""
        if filename is None:
            filename = os.path.join(self.data_dir, "relatorio_bioinformatica.html")
        
        html_content = self.generate_html_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Relatório HTML salvo em: {filename}")
        return filename
    
    def save_pdf_report(self, filename=None):
        """Salvar relatório PDF (requer weasyprint)"""
        try:
            import weasyprint
            
            if filename is None:
                filename = os.path.join(self.data_dir, "relatorio_bioinformatica.pdf")
            
            html_content = self.generate_html_report()
            weasyprint.HTML(string=html_content).write_pdf(filename)
            
            print(f"✅ Relatório PDF salvo em: {filename}")
            return filename
            
        except ImportError:
            print("❌ Para gerar PDF, instale: pip install weasyprint")
            return None
        except Exception as e:
            print(f"❌ Erro ao gerar PDF: {e}")
            return None

def main():
    print("🧬 Gerador de Relatórios de Bioinformática")
    print("=========================================")
    
    # Criar gerador de relatórios
    generator = BioinformaticsReportGenerator()
    
    # Carregar dados
    if not generator.load_data():
        print("❌ Não foi possível carregar os dados. Execute primeiro a análise C++.")
        return
    
    print("✅ Dados carregados com sucesso!")
    
    # Gerar relatório HTML
    html_file = generator.save_html_report()
    
    # Tentar gerar relatório PDF
    pdf_file = generator.save_pdf_report()
    
    print(f"\n📁 Arquivos gerados:")
    print(f"   HTML: {html_file}")
    if pdf_file:
        print(f"   PDF:  {pdf_file}")
    
    print(f"\n🌐 Para visualizar o relatório HTML:")
    print(f"   Abra o arquivo {html_file} no seu navegador")
    
    print(f"\n📈 Relatório inclui:")
    print(f"   - Análise de códons inválidos")
    print(f"   - Estatísticas de impacto IA")
    print(f"   - Gráficos interativos")
    print(f"   - Recomendações de melhorias")

if __name__ == "__main__":
    main()
