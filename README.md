# Sistema Integrado de Bioinformática com Deep Learning

## Versão 2.0 - IA Melhorada

Este sistema integra análise de bioinformática com redes neurais profundas melhoradas, fornecendo uma plataforma completa para análise de sequências de proteínas e DNA.

## 🧬 Funcionalidades Principais

### Rede Neural Melhorada (C++)
- **Múltiplos Otimizadores**: SGD com momentum, Adam, RMSprop
- **Regularização**: Dropout, weight decay
- **Early Stopping**: Prevenção de overfitting
- **Validação Cruzada**: Divisão automática de dados
- **Estatísticas de Treinamento**: Acompanhamento detalhado do progresso
- **Análise de Feature Importance**: Identificação de características importantes

### Processamento de Bioinformática (Python)
- **Processamento FASTA**: Leitura e escrita de arquivos FASTA com nomenclatura melhorada
- **Análise de Sequências**: Estatísticas detalhadas de composição
- **Codificação Genética**: Transcrição DNA→RNA e tradução RNA→Proteína
- **IA para Classificação**: Sistema de aprendizado para análise de impacto
- **Integração C++/Python**: Comunicação entre sistemas

## 📁 Estrutura do Projeto

```
CppDNN/
├── src/
│   ├── DeepNueralNetwork.h      # Rede neural melhorada
│   ├── BioinformaticsIntegration.h  # Integração com bioinformática
│   ├── Layer.h                  # Camadas da rede neural
│   └── Function.h               # Funções de ativação
├── example/
│   ├── bioinformatics_main.cpp  # Exemplo principal
│   └── simple_main/             # Exemplos simples
├── dados/                       # Diretório de dados (criado automaticamente)
│   ├── modelos/                 # Modelos treinados
│   ├── treinamento/             # Dados de treinamento
│   ├── sequencias/              # Arquivos FASTA
│   ├── resultados/              # Resultados de análises
│   └── datasets/                # Datasets processados
├── build_and_run.sh            # Script de compilação e execução
├── sistema_principal.py        # Sistema Python integrado
└── README.md                   # Este arquivo
```

## 🚀 Instalação e Uso

### Pré-requisitos

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev python3 python3-pip
pip3 install numpy pandas matplotlib seaborn biopython
```

### Compilação e Execução

#### Opção 1: Execução Completa (Recomendado)
```bash
./build_and_run.sh --all
```

#### Opção 2: Passo a Passo
```bash
# Verificar dependências
./build_and_run.sh --check-deps

# Configurar ambiente Python
./build_and_run.sh --setup-python

# Compilar projeto
./build_and_run.sh --compile

# Executar exemplos
./build_and_run.sh --run
```


### How to use the library
There is an example in CppDNN/example/simple_main/main.cpp

```
int main()
{
    DeepNueralNetwork dnn;
    dnn.ReadFromKeras("/home/nader/workspace/github/CppDNN/example/keras_simple/simple.txt");
    MatrixXd input(5,1);
    input(0,0) = 1;
    input(1,0) = 2;
    input(2,0) = 1;
    input(3,0) = 2;
    input(4,0) = 1;
    dnn.Calculate(input);
    cout<<dnn.mOutput<<std::endl;
}
```
