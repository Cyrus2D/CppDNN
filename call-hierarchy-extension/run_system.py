#!/usr/bin/env python3
"""
Launcher Simples para Sistema de Bioinformática
"""

import sys
import os
from pathlib import Path
import subprocess

def main():
    print("🧬 Sistema de Bioinformática - Launcher Simples")
    print("=" * 50)
    
    # Verificar se há arquivos Python no diretório atual
    python_files = list(Path('.').glob('*.py'))
    
    if not python_files:
        print("❌ Nenhum arquivo Python encontrado no diretório atual")
        return
    
    print("\n📁 Arquivos Python disponíveis:")
    for i, file in enumerate(python_files, 1):
        print(f"{i}. {file.name}")
    
    try:
        choice = input("\nDigite o número do arquivo para executar (ou Enter para sair): ")
        
        if not choice:
            print("👋 Saindo...")
            return
            
        file_index = int(choice) - 1
        selected_file = python_files[file_index]
        
        print(f"\n🚀 Executando: {selected_file}")
        subprocess.run([sys.executable, str(selected_file)])
        
    except (ValueError, IndexError):
        print("❌ Seleção inválida")
    except KeyboardInterrupt:
        print("\n👋 Operação cancelada")
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
