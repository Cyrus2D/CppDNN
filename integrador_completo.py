#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Integração do Sistema de Bioinformática
===============================================

Este script integra todos os componentes do sistema:
- Simulador de mutações
- Análise de sequências com IA
- Integração com CppDNN
- Geração de datasets para treino

Resolve problemas de importação e dependências
"""

import sys
import os
import subprocess
from pathlib import Path

def instalar_dependencias():
    """Instala dependências Python necessárias"""
    print("=== Instalando Dependências Python ===")
    
    dependencias = [
        'numpy',
        'tensorflow',
        'keras',
        'matplotlib',
        'scikit-learn',
        'biopython',
        'pandas'
    ]
    
    for dep in dependencias:
        try:
            print(f"Verificando {dep}...")
            __import__(dep)
            print(f"✓ {dep} já instalado")
        except ImportError:
            print(f"Instalando {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep, '--user'], 
                         check=True, capture_output=True)
            print(f"✓ {dep} instalado com sucesso")

def verificar_estrutura_diretorios():
    """Verifica e cria estrutura de diretórios necessária"""
    print("\n=== Verificando Estrutura de Diretórios ===")
    
    diretorios = [
        'dados',
        'dados/sequencias',
        'dados/modelos',
        'dados/treinamento',
        'dados/mutacoes',
        'dados/resultados'
    ]
    
    for diretorio in diretorios:
        caminho = Path(diretorio)
        if not caminho.exists():
            caminho.mkdir(parents=True, exist_ok=True)
            print(f"✓ Criado diretório: {diretorio}")
        else:
            print(f"✓ Diretório existe: {diretorio}")

def executar_simulador_mutacoes():
    """Executa o simulador de mutações"""
    print("\n=== Executando Simulador de Mutações ===")
    
    try:
        # Importa o simulador
        sys.path.append('.')
        from simulador_mutacoes import SimuladorMutacoes
        
        # Inicializa simulador
        simulador = SimuladorMutacoes(taxa_mutacao=0.03)
        
        # Gera sequências de exemplo
        print("Gerando sequências de DNA de exemplo...")
        sequencias = [
            simulador.gerar_sequencia_aleatoria(150, 'dna'),
            simulador.gerar_sequencia_aleatoria(200, 'dna'),
            simulador.gerar_sequencia_aleatoria(120, 'dna'),
            simulador.gerar_sequencia_aleatoria(180, 'dna')
        ]
        
        # Gera dataset de mutações
        print("Gerando dataset de mutações...")
        dataset = simulador.gerar_dataset_mutacoes(
            sequencias, 
            num_mutacoes_por_seq=4,
            tipo_sequencia='dna'
        )
        
        # Prepara dados para DNN
        print("Preparando dados para DNN...")
        X, y = simulador.preparar_dados_para_dnn(dataset, 'dna')
        
        # Salva dataset
        simulador.salvar_dataset(dataset, 'dataset_integrado_mutacoes')
        
        # Salva dados para CppDNN
        dados_dnn = {
            'X': X.tolist(),
            'y': y.tolist(),
            'input_size': X.shape[1],
            'num_samples': X.shape[0],
            'num_classes': 2,
            'tipo_sequencia': 'dna'
        }
        
        caminho_dnn = Path('dados/dados_dnn_integrado.json')
        import json
        with open(caminho_dnn, 'w') as f:
            json.dump(dados_dnn, f, indent=2)
        
        print(f"✓ Dataset salvo: {X.shape[0]} amostras, {X.shape[1]} features")
        print(f"✓ Dados para CppDNN salvos em: {caminho_dnn}")
        
        # Gera relatório
        relatorio = simulador.gerar_relatorio_mutacoes(dataset)
        print("\n" + relatorio)
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no simulador de mutações: {e}")
        return False

def executar_sistema_principal():
    """Executa o sistema principal de bioinformática"""
    print("\n=== Executando Sistema Principal ===")
    
    try:
        # Adiciona o diretório pai ao path para importar ia_classi
        parent_dir = Path('..').resolve()
        if str(parent_dir) not in sys.path:
            sys.path.append(str(parent_dir))
        
        # Tenta importar do diretório atual ou do diretório pai
        import importlib.util

        def importar_sistema_integrado():
            caminhos = [
                Path('ia_classi.py'),
                Path('../ia_classi.py')
            ]
            for caminho in caminhos:
                if caminho.exists():
                    spec = importlib.util.spec_from_file_location("ia_classi", str(caminho))
                    ia_classi = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ia_classi)
                    return ia_classi.SistemaIntegradoBioinformatica
            raise ImportError("Arquivo 'ia_classi.py' não encontrado nos diretórios esperados.")

        SistemaIntegradoBioinformatica = importar_sistema_integrado()
        
        # Inicializa sistema
        sistema = SistemaIntegradoBioinformatica("./dados")
        
        # Cria sequências de teste
        sequencias_teste = {
            "Seq_Teste_001": "ATGAAATTTGGGCCCATTCAGAAAGCTGACCTGTCGAATAG",
            "Seq_Teste_002": "ATGCCCGGTTATCGAAATTTCGGCCCTATAG",
            "Seq_Teste_003": "ATGGGGATTTCCCAAAGCTGATCCTTTAG",
            "Seq_Teste_004": "ATGCTGAAATTTCCCGGGATTCAGAAAGCTGATCCTGTAG"
        }
        
        # Salva sequências em arquivo FASTA
        print("Criando arquivo FASTA de teste...")
        arquivo_fasta = sistema.processador_fasta.escrever_fasta(
            sequencias_teste, 
            "sequencias_teste_integrado.fasta"
        )
        
        # Processa arquivo
        print("Processando sequências com IA...")
        resultados = sistema.processar_arquivo_fasta(arquivo_fasta)
        
        # Mostra resultados
        print(f"✓ Processadas {len(resultados)} sequências")
        for nome, dados in resultados.items():
            print(f"  - {nome}: Proteína={dados['proteina'][:20]}..., IA={dados['impacto_ia']:.3f}")
        
        # Salva modelos
        sistema.salvar_modelos()
        print("✓ Modelos de IA salvos")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no sistema principal: {e}")
        import traceback
        traceback.print_exc()
        return False

def compilar_cppdnn():
    """Compila o sistema CppDNN"""
    print("\n=== Compilando CppDNN ===")
    
    try:
        # Verifica se o script de build existe
        script_build = Path('build_and_run.sh')
        if not script_build.exists():
            print("✗ Script build_and_run.sh não encontrado")
            return False
        
        # Torna o script executável
        os.chmod(script_build, 0o755)
        
        # Executa compilação
        print("Compilando código C++...")
        resultado = subprocess.run(['./build_and_run.sh', 'build'], 
                                 capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print("✓ CppDNN compilado com sucesso")
            return True
        else:
            print(f"✗ Erro na compilação: {resultado.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Erro ao compilar CppDNN: {e}")
        return False

def executar_exemplo_cppdnn():
    """Executa exemplo de integração CppDNN"""
    print("\n=== Executando Exemplo CppDNN ===")
    
    try:
        # Executa exemplo de bioinformática
        print("Executando exemplo de integração bioinformática...")
        resultado = subprocess.run(['./build_and_run.sh', 'run_bio'], 
                                 capture_output=True, text=True, timeout=30)
        
        if resultado.returncode == 0:
            print("✓ Exemplo CppDNN executado com sucesso")
            print("Saída:")
            print(resultado.stdout)
            return True
        else:
            print(f"✗ Erro na execução: {resultado.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout na execução do exemplo")
        return False
    except Exception as e:
        print(f"✗ Erro ao executar exemplo: {e}")
        return False

def gerar_relatorio_final():
    """Gera relatório final do sistema integrado"""
    print("\n=== Relatório Final do Sistema ===")
    
    # Verifica arquivos gerados
    arquivos_esperados = [
        'dados/dataset_integrado_mutacoes.json',
        'dados/dados_dnn_integrado.json',
        'dados/sequencias/sequencias_teste_integrado.fasta',
        'dados/modelos/ia_memoria.pkl'
    ]
    
    print("Arquivos gerados:")
    for arquivo in arquivos_esperados:
        caminho = Path(arquivo)
        if caminho.exists():
            tamanho = caminho.stat().st_size
            print(f"  ✓ {arquivo} ({tamanho} bytes)")
        else:
            print(f"  ✗ {arquivo} (não encontrado)")
    
    # Verifica diretórios
    print("\nDiretórios criados:")
    for pasta in ['dados', 'dados/sequencias', 'dados/modelos', 'dados/treinamento']:
        caminho = Path(pasta)
        if caminho.exists():
            num_arquivos = len(list(caminho.glob('*')))
            print(f"  ✓ {pasta} ({num_arquivos} arquivos)")
        else:
            print(f"  ✗ {pasta} (não encontrado)")

def main():
    """Função principal de integração"""
    print("=" * 60)
    print("SISTEMA INTEGRADO DE BIOINFORMÁTICA E DEEP LEARNING")
    print("=" * 60)
    
    sucesso_total = True
    
    # 1. Instalar dependências
    try:
        instalar_dependencias()
    except Exception as e:
        print(f"✗ Erro ao instalar dependências: {e}")
        sucesso_total = False
    
    # 2. Verificar estrutura
    verificar_estrutura_diretorios()
    
    # 3. Executar simulador de mutações
    if not executar_simulador_mutacoes():
        sucesso_total = False
    
    # 4. Executar sistema principal
    if not executar_sistema_principal():
        sucesso_total = False
    
    # 5. Compilar CppDNN
    if not compilar_cppdnn():
        sucesso_total = False
    
    # 6. Executar exemplo CppDNN
    if not executar_exemplo_cppdnn():
        print("⚠ Exemplo CppDNN não executado (pode ser normal se não há dados)")
    
    # 7. Gerar relatório
    gerar_relatorio_final()
    
    # Resultado final
    print("\n" + "=" * 60)
    if sucesso_total:
        print("✓ SISTEMA INTEGRADO CONFIGURADO COM SUCESSO!")
        print("\nPróximos passos:")
        print("1. Execute: python integrador_completo.py")
        print("2. Execute: ./build_and_run.sh run_bio")
        print("3. Verifique os dados em: ./dados/")
    else:
        print("✗ ALGUNS COMPONENTES FALHARAM")
        print("Verifique os erros acima e corrija os problemas")
    print("=" * 60)

if __name__ == "__main__":
    main()
