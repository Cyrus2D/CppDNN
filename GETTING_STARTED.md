# 🧬 Sistema de Bioinformática C++ - Deep Neural Network

Um sistema avançado de análise de sequências de proteínas usando redes neurais profundas em C++, com foco especial no tratamento de códons inválidos e análise de impacto IA.

## 🚀 Funcionalidades Principais

### ✅ Validação de Sequências
- **Detecção de códons inválidos**: Identifica e trata códons como `?`, `*`, `-`
- **Limpeza automática**: Remove ou substitui resíduos inválidos
- **Relatórios detalhados**: Gera relatórios completos de validação
- **Suporte a resíduos ambíguos**: Trata códons como `X`, `B`, `Z`, `J`, `U`, `O`

### 📁 Versionamento de Modelos
- **Backup automático**: Modelos salvos com timestamp único
- **Histórico completo**: Rastreamento de performance e configurações
- **Metadados JSON**: Informações detalhadas de cada versão
- **Recuperação fácil**: Carregamento de versões específicas

### 📊 Análise de Impacto IA
- **Logging detalhado**: Registra contribuições de features
- **Métricas de confiança**: Classifica predições (ALTA/MÉDIA/BAIXA)
- **Análise de importância**: Identifica features mais relevantes
- **Correlações**: GC content vs impacto, comprimento vs performance

### 📈 Visualização e Relatórios
- **Relatórios HTML/PDF**: Documentação completa com gráficos
- **Scripts Python**: Geração automática de visualizações
- **Dashboards interativos**: Análise gráfica dos resultados
- **Exportação CSV**: Dados prontos para análise externa

## 🔧 Instalação e Configuração

### Dependências

#### C++
```bash
sudo apt install build-essential libeigen3-dev pkg-config
```

#### Python (para relatórios)
```bash
pip3 install pandas matplotlib seaborn jinja2 weasyprint numpy scipy
```

### Compilação

```bash
# Clonar/baixar o projeto
cd CppDNN

# Verificar dependências
make check-deps

# Instalar dependências Python
make install-deps

# Compilar projeto completo
make all
```

## 📋 Uso Rápido

### Exemplo Básico
```bash
# Executar análise de exemplo
make run-example

# Análise completa com relatórios
make run-analysis

# Testar especificamente códons inválidos
make test-invalid-codons
```

## 🔍 Tratamento de Códons Inválidos

### Problema Identificado
O sistema agora detecta e trata automaticamente:

- **`?` (códon incompleto)**: Códons não mapeados ou incompletos
- **`*` (códon de parada)**: Stop codons em posições inadequadas  
- **`-` (gaps)**: Lacunas na sequência
- **Caracteres inválidos**: Qualquer resíduo não padrão

### Solução Implementada

1. **Validação Prévia**
   ```cpp
   auto result = dnn.ValidateProteinSequence(sequence, "Sequencia_Teste_002");
   result.PrintReport(); // Mostra problemas encontrados
   ```

2. **Limpeza Automática**
   ```cpp
   std::string clean = dnn.CleanProteinSequence(sequence, true);
   // Remove códons inválidos automaticamente
   ```

3. **Logging Detalhado**
   ```cpp
   dnn.GenerateValidationReport(sequences, ids);
   // Cria relatório completo em ./dados/validacao/
   ```

## 🐛 Problemas Resolvidos

### ⚠️ Códon inválido em Sequencia_Teste_002
**Status**: ✅ **RESOLVIDO**

- **Problema**: Presença de `?` indicando códons incompletos
- **Solução**: Sistema de validação automática implementado
- **Resultado**: Detecção, logging e limpeza automática
- **Arquivos**: Relatórios em `./dados/validacao/`

### 📁 Backup/Versão de Modelos  
**Status**: ✅ **IMPLEMENTADO**

- **Feature**: Versionamento automático com timestamp
- **Benefício**: Histórico completo de treinamentos
- **Uso**: `dnn.SaveVersionedModel("descrição")`
- **Localização**: `./dados/modelos/v{timestamp}/`

### 📈 Expansão do "Impacto IA"
**Status**: ✅ **IMPLEMENTADO**

- **Feature**: Logging detalhado de features e métricas
- **Análise**: Correlações, importância, confiança
- **Relatórios**: HTML/PDF com gráficos interativos
- **Scripts**: Geração automática de visualizações

---

**Para usar o sistema, execute**: `make run-analysis`
