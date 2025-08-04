#!/usr/bin/env python3
"""
Sistema de Bioinformática Integrado
Ponto de entrada principal do sistema
Autor: AnguaLiberto System
"""

import sys
import os
from pathlib import Path

# Adicionar src ao Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """Ponto de entrada principal"""
    
    print("🧬 === SISTEMA DE BIOINFORMÁTICA INTEGRADO ===")
    print("🚀 AnguaLiberto Professional Platform")
    print("=" * 50)
    
    # Verificar estrutura do projeto
    required_dirs = ['src', 'data', 'config']
    missing_dirs = [d for d in required_dirs if not (project_root / d).exists()]
    
    if missing_dirs:
        print(f"❌ Diretórios ausentes: {', '.join(missing_dirs)}")
        print("💡 Execute o script de organização primeiro")
        return 1
    
    print("\n🎯 Módulos disponíveis:")
    print("1. 📊 Visualização 3D de Proteínas")
    print("2. 🧪 Análise Avançada PDB")
    print("3. 🔬 Simulador de Mutações")
    print("4. 🌐 Interface Web")
    print("5. 🤖 Análise com IA")
    print("6. 📈 Relatórios e Dashboard")
    print("7. 🛠️ Diagnóstico do Sistema")
    
    try:
        choice = input("\n👉 Selecione um módulo (1-7): ").strip()
        
        if choice == "1":
            from src.visualization.visualizador_3d_proteinas import main as viz_main
            viz_main()
        elif choice == "2":
            from src.analysis.analisador_pdb_avancado import main as pdb_main
            pdb_main()
        elif choice == "3":
            from src.analysis.simulador_mutacoes import main as sim_main
            sim_main()
        elif choice == "4":
            from src.interfaces.interface_dinamica import main as web_main
            web_main()
        elif choice == "5":
            from src.analysis.ia_classi import main as ia_main
            ia_main()
        elif choice == "6":
            print("📈 Gerando relatórios...")
            generate_reports()
        elif choice == "7":
            system_diagnostics()
        else:
            print("❌ Opção inválida")
            return 1
            
    except ImportError as e:
        print(f"❌ Erro ao importar módulo: {e}")
        print("💡 Verifique se os arquivos estão organizados corretamente")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Sistema finalizado pelo usuário")
        return 0
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return 1
    
    return 0

def generate_reports():
    """Gera relatórios do sistema"""
    output_dir = project_root / "data" / "output"
    
    html_files = list(output_dir.glob("**/*.html"))
    images = list(output_dir.glob("**/*.png"))
    reports = list(output_dir.glob("**/*.json"))
    
    print(f"\n📊 Estatísticas do projeto:")
    print(f"   🌐 Visualizações HTML: {len(html_files)}")
    print(f"   🖼️ Imagens geradas: {len(images)}")
    print(f"   📋 Relatórios JSON: {len(reports)}")

def system_diagnostics():
    """Diagnóstico do sistema"""
    print("\n🛠️ Executando diagnóstico do sistema...")
    
    # Verificar Python packages
    required_packages = [
        'numpy', 'matplotlib', 'plotly', 'biopython', 
        'pandas', 'scipy', 'flask'
    ]
    
    print("\n📦 Verificando pacotes Python:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NÃO INSTALADO")
    
    # Verificar estrutura de arquivos
    print("\n📁 Verificando estrutura:")
    important_dirs = [
        'src/core', 'src/analysis', 'src/visualization',
        'data/input/pdb', 'data/output/html', 'config/settings'
    ]
    
    for directory in important_dirs:
        dir_path = project_root / directory
        if dir_path.exists():
            count = len(list(dir_path.glob("*")))
            print(f"   ✅ {directory}/ ({count} arquivos)")
        else:
            print(f"   ❌ {directory}/ - AUSENTE")

if __name__ == "__main__":
    sys.exit(main())

"""
visualizador_3d_proteinas.py
Módulo de visualização 3D de proteínas
"""

def main():
    print("📊 Visualização 3D de proteínas iniciada (placeholder)")
