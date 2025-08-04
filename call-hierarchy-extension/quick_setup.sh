#!/bin/bash
echo "🚀 Setup Rápido - Bioinformática"

# Criar ambiente virtual
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    echo "✅ Ambiente virtual criado"
fi

# Ativar ambiente
source venv/bin/activate
echo "✅ Ambiente virtual ativado"

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependências instaladas"

echo "🎉 Setup concluído!"
echo "Para usar: source venv/bin/activate"
