# Makefile para o projeto CppDNN - Sistema de Bioinformática
# Compilação do sistema de redes neurais para análise de proteínas

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
INCLUDES = -Isrc/ -I/usr/include/eigen3
LIBS = -lm

# Diretórios
SRC_DIR = src
BUILD_DIR = build
EXAMPLES_DIR = examples
DATA_DIR = dados

# Arquivos fonte (você precisará adicionar os outros arquivos .cpp conforme necessário)
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Arquivos de exemplo
EXAMPLE_SOURCES = $(wildcard $(EXAMPLES_DIR)/*.cpp)
EXAMPLES = $(EXAMPLE_SOURCES:$(EXAMPLES_DIR)/%.cpp=$(BUILD_DIR)/%)

# Alvo principal
TARGET = $(BUILD_DIR)/bioinformatics_dnn

.PHONY: all clean setup examples run-example install-deps help

# Compilar tudo
all: setup $(TARGET) examples

# Configurar diretórios
setup:
	@echo "🔧 Configurando diretórios..."
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(DATA_DIR)/modelos
	@mkdir -p $(DATA_DIR)/logs
	@mkdir -p $(DATA_DIR)/validacao
	@mkdir -p $(DATA_DIR)/visualizacao
	@mkdir -p $(DATA_DIR)/treinamento
	@mkdir -p $(DATA_DIR)/sequencias
	@echo "✅ Diretórios criados!"

# Compilar arquivos objeto
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "🔨 Compilando $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Linkar programa principal
$(TARGET): $(OBJECTS)
	@echo "🔗 Linkando programa principal..."
	@$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LIBS)
	@echo "✅ Programa principal compilado: $@"

# Compilar exemplos
examples: $(EXAMPLES)

$(BUILD_DIR)/%: $(EXAMPLES_DIR)/%.cpp $(OBJECTS)
	@echo "🔨 Compilando exemplo $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS)) -o $@ $(LIBS)

# Executar exemplo
run-example: $(BUILD_DIR)/example_usage
	@echo "🚀 Executando exemplo de uso..."
	@./$(BUILD_DIR)/example_usage

# Instalar dependências Python
install-deps:
	@echo "📦 Instalando dependências Python..."
	@pip3 install pandas matplotlib seaborn jinja2 weasyprint numpy scipy
	@echo "✅ Dependências instaladas!"

# Executar análise completa
run-analysis: all
	@echo "🧬 Executando análise completa de bioinformática..."
	@./$(BUILD_DIR)/example_usage
	@echo "📊 Gerando relatórios..."
	@cd $(DATA_DIR) && python3 generate_pdf_report.py
	@echo "📈 Gerando visualizações..."
	@cd $(DATA_DIR)/visualizacao && python3 plot_impact.py
	@echo "✅ Análise completa finalizada!"

# Executar análise de exemplo (para testar códons inválidos)
test-invalid-codons: all
	@echo "🧪 Testando análise de códons inválidos..."
	@echo "Executando análise de sequências com códons '?' e outros inválidos..."
	@./$(BUILD_DIR)/example_usage
	@echo "📋 Verificando relatórios gerados..."
	@ls -la $(DATA_DIR)/validacao/
	@ls -la $(DATA_DIR)/logs/
	@echo "✅ Teste de códons inválidos concluído!"

# Limpar arquivos compilados
clean:
	@echo "🧹 Limpando arquivos compilados..."
	@rm -rf $(BUILD_DIR)
	@echo "✅ Limpeza concluída!"

# Limpar dados gerados
clean-data:
	@echo "🧹 Limpando dados gerados..."
	@rm -rf $(DATA_DIR)/logs/*
	@rm -rf $(DATA_DIR)/validacao/*
	@rm -rf $(DATA_DIR)/visualizacao/*.csv
	@rm -rf $(DATA_DIR)/visualizacao/*.png
	@rm -rf $(DATA_DIR)/*.html
	@rm -rf $(DATA_DIR)/*.pdf
	@echo "✅ Dados limpos!"

# Backup de modelos
backup-models:
	@echo "💾 Fazendo backup de modelos..."
	@tar -czf modelos_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz $(DATA_DIR)/modelos/
	@echo "✅ Backup criado!"

# Verificar dependências do sistema
check-deps:
	@echo "🔍 Verificando dependências..."
	@echo "Verificando Eigen3..."
	@pkg-config --exists eigen3 && echo "✅ Eigen3 encontrado" || echo "❌ Eigen3 não encontrado - instale: sudo apt install libeigen3-dev"
	@echo "Verificando Python3..."
	@python3 --version && echo "✅ Python3 encontrado" || echo "❌ Python3 não encontrado"
	@echo "Verificando pip3..."
	@pip3 --version && echo "✅ pip3 encontrado" || echo "❌ pip3 não encontrado"
	@echo "Verificando g++..."
	@$(CXX) --version && echo "✅ g++ encontrado" || echo "❌ g++ não encontrado"

# Mostrar informações do projeto
info:
	@echo "🧬 Sistema de Bioinformática - CppDNN"
	@echo "====================================="
	@echo "Funcionalidades implementadas:"
	@echo "  ✅ Validação de sequências de proteínas"
	@echo "  ✅ Tratamento de códons inválidos (?, *, -)"
	@echo "  ✅ Versionamento automático de modelos"
	@echo "  ✅ Logging detalhado de impacto IA"
	@echo "  ✅ Geração de relatórios HTML/PDF"
	@echo "  ✅ Scripts de visualização Python"
	@echo "  ✅ Pipeline completo de análise"
	@echo ""
	@echo "Arquivos principais:"
	@echo "  📄 src/DeepNueralNetwork.h - Classe principal"
	@echo "  📄 examples/example_usage.cpp - Exemplo de uso"
	@echo "  📄 dados/generate_pdf_report.py - Gerador de relatórios"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  make all                 - Compilar tudo"
	@echo "  make run-example         - Executar exemplo"
	@echo "  make run-analysis        - Análise completa"
	@echo "  make test-invalid-codons - Testar códons inválidos"
	@echo "  make install-deps        - Instalar dependências Python"
	@echo "  make clean               - Limpar compilados"
	@echo "  make clean-data          - Limpar dados gerados"

# Ajuda
help: info

# Instalar no sistema (opcional)
install: all
	@echo "📦 Instalando no sistema..."
	@sudo cp $(TARGET) /usr/local/bin/
	@sudo cp -r $(DATA_DIR) /usr/local/share/bioinformatics-dnn/
	@echo "✅ Instalado em /usr/local/bin/bioinformatics_dnn"

# Desinstalar do sistema
uninstall:
	@echo "🗑️ Desinstalando do sistema..."
	@sudo rm -f /usr/local/bin/bioinformatics_dnn
	@sudo rm -rf /usr/local/share/bioinformatics-dnn/
	@echo "✅ Desinstalado!"

# Executar testes unitários (para implementação futura)
test:
	@echo "🧪 Executando testes..."
	@echo "ℹ️ Testes unitários não implementados ainda"
	@echo "💡 Execute 'make test-invalid-codons' para testar funcionalidades específicas"

# Verificar problemas de códons inválidos especificamente
validate-sequences:
	@echo "🔍 Validando sequências de exemplo..."
	@./$(BUILD_DIR)/example_usage | grep -A 10 "VALIDAÇÃO DE SEQUÊNCIAS"
	@echo "📋 Verifique o arquivo de validação:"
	@cat $(DATA_DIR)/validacao/validation_report.txt | head -20
