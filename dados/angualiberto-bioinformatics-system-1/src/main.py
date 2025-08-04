#!/usr/bin/env python3
"""Sistema Integrado de Bioinformática e Análise de Proteínas
Autor Principal: AnguaLiberto
Licença: MIT License (Open Source)
Data: 2024
"""

from core.system import AnguaLibertoSystem

def main():
    """Função principal que inicializa o sistema e inicia os processos de análise."""
    print("🚀 Iniciando Sistema AnguaLiberto...")
    sistema = AnguaLibertoSystem("Projeto AnguaLiberto")
    # Aqui você pode adicionar chamadas para métodos de análise ou geração de relatórios
    # Exemplo: sistema.generate_complete_report(sequence, analysis_type)

if __name__ == "__main__":
    main()