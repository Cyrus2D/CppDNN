#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador de Mutações para Análise Bioinformática
=================================================

Este módulo implementa um simulador de mutações genéticas para:
- Gerar sequências mutadas a partir de sequências originais
- Simular diferentes tipos de mutações (substituição, inserção, deleção)
- Criar datasets de treino para redes neurais profundas
- Integrar com o sistema CppDNN para análise de mutações

Autor: Sistema Integrado de Bioinformática
"""

import random
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class SimuladorMutacoes:
    """
    Classe para simular mutações genéticas em sequências de DNA/RNA/Proteínas
    """
    
    def __init__(self, taxa_mutacao: float = 0.01):
        """
        Inicializa o simulador de mutações
        
        Args:
            taxa_mutacao: Taxa de mutação (0.0 a 1.0)
        """
        self.taxa_mutacao = taxa_mutacao
        self.tipos_mutacao = ['substituicao', 'insercao', 'delecao']
        self.bases_dna = ['A', 'T', 'G', 'C']
        self.bases_rna = ['A', 'U', 'G', 'C']
        self.aminoacidos = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        
        # Diretório para salvar dados
        self.dados_dir = Path("dados")
        self.dados_dir.mkdir(exist_ok=True)
        
        # Histórico de mutações
        self.historico_mutacoes = []
    
    def gerar_sequencia_aleatoria(self, tamanho: int, tipo: str = 'dna') -> str:
        """
        Gera uma sequência aleatória
        
        Args:
            tamanho: Tamanho da sequência
            tipo: Tipo de sequência ('dna', 'rna', 'proteina')
            
        Returns:
            Sequência gerada
        """
        if tipo.lower() == 'dna':
            alfabeto = self.bases_dna
        elif tipo.lower() == 'rna':
            alfabeto = self.bases_rna
        elif tipo.lower() == 'proteina':
            alfabeto = self.aminoacidos
        else:
            alfabeto = self.bases_dna
            
        return ''.join(random.choices(alfabeto, k=tamanho))
    
    def aplicar_substituicao(self, sequencia: str, posicao: int, tipo: str = 'dna') -> Tuple[str, Dict]:
        """
        Aplica mutação por substituição
        
        Args:
            sequencia: Sequência original
            posicao: Posição da mutação
            tipo: Tipo de sequência
            
        Returns:
            Tupla com (sequência mutada, info da mutação)
        """
        if posicao >= len(sequencia):
            return sequencia, {}
            
        if tipo.lower() == 'dna':
            alfabeto = self.bases_dna
        elif tipo.lower() == 'rna':
            alfabeto = self.bases_rna
        else:
            alfabeto = self.aminoacidos
            
        original = sequencia[posicao]
        # Escolhe uma base/aminoácido diferente
        opcoes = [x for x in alfabeto if x != original]
        nova_base = random.choice(opcoes)
        
        sequencia_mutada = sequencia[:posicao] + nova_base + sequencia[posicao+1:]
        
        info_mutacao = {
            'tipo': 'substituicao',
            'posicao': posicao,
            'original': original,
            'mutacao': nova_base
        }
        
        return sequencia_mutada, info_mutacao
    
    def aplicar_insercao(self, sequencia: str, posicao: int, tipo: str = 'dna') -> Tuple[str, Dict]:
        """
        Aplica mutação por inserção
        
        Args:
            sequencia: Sequência original
            posicao: Posição da inserção
            tipo: Tipo de sequência
            
        Returns:
            Tupla com (sequência mutada, info da mutação)
        """
        if tipo.lower() == 'dna':
            alfabeto = self.bases_dna
        elif tipo.lower() == 'rna':
            alfabeto = self.bases_rna
        else:
            alfabeto = self.aminoacidos
            
        nova_base = random.choice(alfabeto)
        sequencia_mutada = sequencia[:posicao] + nova_base + sequencia[posicao:]
        
        info_mutacao = {
            'tipo': 'insercao',
            'posicao': posicao,
            'insercao': nova_base
        }
        
        return sequencia_mutada, info_mutacao
    
    def aplicar_delecao(self, sequencia: str, posicao: int) -> Tuple[str, Dict]:
        """
        Aplica mutação por deleção
        
        Args:
            sequencia: Sequência original
            posicao: Posição da deleção
            
        Returns:
            Tupla com (sequência mutada, info da mutação)
        """
        if posicao >= len(sequencia) or len(sequencia) <= 1:
            return sequencia, {}
            
        base_deletada = sequencia[posicao]
        sequencia_mutada = sequencia[:posicao] + sequencia[posicao+1:]
        
        info_mutacao = {
            'tipo': 'delecao',
            'posicao': posicao,
            'deletado': base_deletada
        }
        
        return sequencia_mutada, info_mutacao
    
    def simular_mutacoes(self, sequencia: str, num_mutacoes: Optional[int] = None, 
                        tipo_sequencia: str = 'dna') -> Tuple[str, List[Dict]]:
        """
        Simula múltiplas mutações em uma sequência
        
        Args:
            sequencia: Sequência original
            num_mutacoes: Número de mutações (None para usar taxa_mutacao)
            tipo_sequencia: Tipo de sequência
            
        Returns:
            Tupla com (sequência mutada, lista de mutações)
        """
        if num_mutacoes is None:
            num_mutacoes = max(1, int(len(sequencia) * self.taxa_mutacao))
        
        sequencia_atual = sequencia
        mutacoes_aplicadas = []
        
        for _ in range(num_mutacoes):
            # Escolhe tipo de mutação aleatoriamente
            tipo_mutacao = random.choice(self.tipos_mutacao)
            posicao = random.randint(0, len(sequencia_atual) - 1)
            
            if tipo_mutacao == 'substituicao':
                sequencia_atual, info = self.aplicar_substituicao(
                    sequencia_atual, posicao, tipo_sequencia
                )
            elif tipo_mutacao == 'insercao':
                sequencia_atual, info = self.aplicar_insercao(
                    sequencia_atual, posicao, tipo_sequencia
                )
            elif tipo_mutacao == 'delecao':
                sequencia_atual, info = self.aplicar_delecao(
                    sequencia_atual, posicao
                )
            
            if info:  # Se a mutação foi aplicada com sucesso
                mutacoes_aplicadas.append(info)
        
        return sequencia_atual, mutacoes_aplicadas
    
    def gerar_dataset_mutacoes(self, sequencias_originais: List[str], 
                              num_mutacoes_por_seq: int = 5,
                              tipo_sequencia: str = 'dna') -> Dict:
        """
        Gera um dataset de mutações para treino
        
        Args:
            sequencias_originais: Lista de sequências originais
            num_mutacoes_por_seq: Número de mutações por sequência
            tipo_sequencia: Tipo de sequência
            
        Returns:
            Dataset com sequências originais e mutadas
        """
        dataset = {
            'originais': [],
            'mutadas': [],
            'mutacoes_info': [],
            'labels': []  # 0 = original, 1 = mutada
        }
        
        # Adiciona sequências originais
        for seq in sequencias_originais:
            dataset['originais'].append(seq)
            dataset['mutadas'].append(seq)
            dataset['mutacoes_info'].append([])
            dataset['labels'].append(0)
        
        # Gera mutações
        for seq_original in sequencias_originais:
            for _ in range(num_mutacoes_por_seq):
                seq_mutada, mutacoes = self.simular_mutacoes(
                    seq_original, tipo_sequencia=tipo_sequencia
                )
                
                dataset['originais'].append(seq_original)
                dataset['mutadas'].append(seq_mutada)
                dataset['mutacoes_info'].append(mutacoes)
                dataset['labels'].append(1)
        
        return dataset
    
    def codificar_sequencia_para_dnn(self, sequencia: str, tipo: str = 'dna') -> np.ndarray:
        """
        Codifica sequência em formato numérico para DNN
        
        Args:
            sequencia: Sequência a ser codificada
            tipo: Tipo de sequência
            
        Returns:
            Array numpy com codificação one-hot
        """
        if tipo.lower() == 'dna':
            mapeamento = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
            tamanho_alfabeto = 4
        elif tipo.lower() == 'rna':
            mapeamento = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
            tamanho_alfabeto = 4
        else:  # proteína
            mapeamento = {aa: i for i, aa in enumerate(self.aminoacidos)}
            tamanho_alfabeto = 20
        
        # One-hot encoding
        matriz = np.zeros((len(sequencia), tamanho_alfabeto))
        for i, base in enumerate(sequencia):
            if base in mapeamento:
                matriz[i, mapeamento[base]] = 1
        
        return matriz.flatten()  # Flatten para entrada da DNN
    
    def preparar_dados_para_dnn(self, dataset: Dict, tipo: str = 'dna') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara dados do dataset para treino da DNN
        
        Args:
            dataset: Dataset gerado
            tipo: Tipo de sequência
            
        Returns:
            Tupla com (X, y) para treino
        """
        X = []
        y = []
        
        for seq, label in zip(dataset['mutadas'], dataset['labels']):
            # Codifica sequência
            seq_codificada = self.codificar_sequencia_para_dnn(seq, tipo)
            X.append(seq_codificada)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def salvar_dataset(self, dataset: Dict, nome_arquivo: str):
        """
        Salva dataset em arquivo JSON
        
        Args:
            dataset: Dataset a ser salvo
            nome_arquivo: Nome do arquivo
        """
        caminho = self.dados_dir / f"{nome_arquivo}.json"
        
        # Converte numpy arrays para listas para serialização JSON
        dataset_serializable = {}
        for chave, valor in dataset.items():
            if isinstance(valor, np.ndarray):
                dataset_serializable[chave] = valor.tolist()
            else:
                dataset_serializable[chave] = valor
        
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dataset_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset salvo em: {caminho}")
    
    def carregar_dataset(self, nome_arquivo: str) -> Dict:
        """
        Carrega dataset de arquivo JSON
        
        Args:
            nome_arquivo: Nome do arquivo
            
        Returns:
            Dataset carregado
        """
        caminho = self.dados_dir / f"{nome_arquivo}.json"
        
        with open(caminho, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        return dataset
    
    def gerar_relatorio_mutacoes(self, dataset: Dict) -> str:
        """
        Gera relatório das mutações no dataset
        
        Args:
            dataset: Dataset para análise
            
        Returns:
            Relatório em formato string
        """
        relatorio = []
        relatorio.append("=== RELATÓRIO DE MUTAÇÕES ===\n")
        
        # Estatísticas gerais
        total_sequencias = len(dataset['labels'])
        originais = sum(1 for label in dataset['labels'] if label == 0)
        mutadas = sum(1 for label in dataset['labels'] if label == 1)
        
        relatorio.append(f"Total de sequências: {total_sequencias}")
        relatorio.append(f"Sequências originais: {originais}")
        relatorio.append(f"Sequências mutadas: {mutadas}")
        relatorio.append("")
        
        # Tipos de mutações
        tipos_mutacao = {}
        for mutacoes_seq in dataset['mutacoes_info']:
            for mutacao in mutacoes_seq:
                if 'tipo' in mutacao:
                    tipo = mutacao['tipo']
                    tipos_mutacao[tipo] = tipos_mutacao.get(tipo, 0) + 1
        
        relatorio.append("Tipos de mutações:")
        for tipo, count in tipos_mutacao.items():
            relatorio.append(f"  {tipo}: {count}")
        
        return "\n".join(relatorio)


def main():
    """
    Função principal para demonstração do simulador
    """
    print("=== Simulador de Mutações - Sistema Integrado ===\n")
    
    # Inicializa simulador
    simulador = SimuladorMutacoes(taxa_mutacao=0.05)
    
    # Gera algumas sequências de exemplo
    print("Gerando sequências de exemplo...")
    sequencias_exemplo = [
        simulador.gerar_sequencia_aleatoria(100, 'dna'),
        simulador.gerar_sequencia_aleatoria(150, 'dna'),
        simulador.gerar_sequencia_aleatoria(200, 'dna')
    ]
    
    print(f"Geradas {len(sequencias_exemplo)} sequências de DNA")
    
    # Gera dataset de mutações
    print("\nGerando dataset de mutações...")
    dataset = simulador.gerar_dataset_mutacoes(
        sequencias_exemplo, 
        num_mutacoes_por_seq=3,
        tipo_sequencia='dna'
    )
    
    # Prepara dados para DNN
    print("Preparando dados para DNN...")
    X, y = simulador.preparar_dados_para_dnn(dataset, 'dna')
    print(f"Shape dos dados: X = {X.shape}, y = {y.shape}")
    
    # Salva dataset
    print("Salvando dataset...")
    simulador.salvar_dataset(dataset, 'dataset_mutacoes_dna')
    
    # Gera relatório
    print("\nRelatório das mutações:")
    relatorio = simulador.gerar_relatorio_mutacoes(dataset)
    print(relatorio)
    
    # Salva dados para integração com CppDNN
    dados_dnn = {
        'X': X.tolist(),
        'y': y.tolist(),
        'input_size': X.shape[1],
        'num_samples': X.shape[0],
        'num_classes': 2
    }
    
    caminho_dnn = simulador.dados_dir / 'dados_dnn_mutacoes.json'
    with open(caminho_dnn, 'w') as f:
        json.dump(dados_dnn, f, indent=2)
    
    print(f"\nDados para CppDNN salvos em: {caminho_dnn}")
    print("Sistema pronto para integração com CppDNN!")


if __name__ == "__main__":
    main()
