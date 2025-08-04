/**
 * Exemplo de uso das novas funcionalidades do DeepNeuralNetwork
 * 
 * Este arquivo demonstra como usar:
 * - Validação de sequências com códons inválidos
 * - Versionamento de modelos
 * - Logging detalhado de impacto IA
 * - Geração de relatórios e visualizações
 */

#include "../src/DeepNueralNetwork.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "=== EXEMPLO DE USO - SISTEMA DE BIOINFORMÁTICA C++ ===" << std::endl;
    
    // Criar instância da rede neural
    DeepNeuralNetwork dnn;
    
    // Configurar parâmetros de treinamento
    TrainingConfig config;
    config.learning_rate = 0.001;
    config.optimizer = "adam";
    config.batch_size = 16;
    config.epochs = 50;
    config.early_stopping = true;
    
    dnn.SetTrainingConfig(config);
    
    // =================================================================
    // EXEMPLO 1: VALIDAÇÃO DE SEQUÊNCIAS
    // =================================================================
    std::cout << "\n### EXEMPLO 1: VALIDAÇÃO DE SEQUÊNCIAS ###" << std::endl;
    
    std::vector<std::string> sequences = {
        "MKLLVVLGAAVVWRYTADGEMPQLTQQQHQQLQHH",   // Normal
        "ARNDCQEGHILKMFPSTWYV?ARNDCQEGHILKMF",   // Com '?' (códon inválido)
        "MKLLVVLGAAVVWRYTADG*MPQLTQQQHQQLQHH",   // Com '*' (stop códon)
        "ARNDCQEGHILKMFPSTWYV-ARNDCQEGHILKMF",   // Com '-' (gap)
        "mkllvvlgaavvwrytadgempqltqqqhqqlqhh"    // Minúsculas
    };
    
    std::vector<std::string> seq_ids = {
        "Protein_A_Normal",
        "Protein_B_Invalid_Codon",  // Esta é similar à "Sequencia_Teste_002" mencionada
        "Protein_C_Stop_Codon",
        "Protein_D_With_Gaps",
        "Protein_E_Lowercase"
    };
    
    // Validar cada sequência individualmente
    for (size_t i = 0; i < sequences.size(); ++i) {
        std::cout << "\n--- Validando " << seq_ids[i] << " ---" << std::endl;
        auto result = dnn.ValidateProteinSequence(sequences[i], seq_ids[i]);
        result.PrintReport();
        
        // Demonstrar limpeza de sequência
        if (!result.is_valid || !result.warnings.empty()) {
            std::string cleaned = dnn.CleanProteinSequence(sequences[i], true);
            std::cout << "Sequência original: " << sequences[i] << std::endl;
            std::cout << "Sequência limpa:    " << cleaned << std::endl;
        }
    }
    
    // =================================================================
    // EXEMPLO 2: ANÁLISE DE IMPACTO IA (simulada)
    // =================================================================
    std::cout << "\n### EXEMPLO 2: ANÁLISE DE IMPACTO IA ###" << std::endl;
    
    // Nota: Para este exemplo, vamos simular uma rede treinada
    // Em uso real, você carregaria um modelo já treinado
    std::cout << "Simulando análise de impacto IA..." << std::endl;
    
    // Analisar impacto de cada sequência
    std::vector<double> impacts;
    for (size_t i = 0; i < sequences.size(); ++i) {
        // Limpar sequência primeiro
        std::string clean_seq = dnn.CleanProteinSequence(sequences[i], true);
        
        if (!clean_seq.empty()) {
            // Simular cálculo de impacto (em uso real usaria modelo treinado)
            double simulated_impact = 0.5 + (i * 0.1) + (clean_seq.length() * 0.001);
            if (simulated_impact > 1.0) simulated_impact = 1.0;
            
            impacts.push_back(simulated_impact);
            
            std::cout << seq_ids[i] << ": Impacto = " << std::fixed 
                     << std::setprecision(4) << simulated_impact << std::endl;
        } else {
            impacts.push_back(0.0);
            std::cout << seq_ids[i] << ": Sequência inválida, Impacto = 0.0000" << std::endl;
        }
    }
    
    // =================================================================
    // EXEMPLO 3: GERAÇÃO DE RELATÓRIOS
    // =================================================================
    std::cout << "\n### EXEMPLO 3: GERAÇÃO DE RELATÓRIOS ###" << std::endl;
    
    // Gerar relatório de validação completo
    dnn.GenerateValidationReport(sequences, seq_ids);
    
    // =================================================================
    // EXEMPLO 4: VERSIONAMENTO DE MODELOS
    // =================================================================
    std::cout << "\n### EXEMPLO 4: VERSIONAMENTO DE MODELOS ###" << std::endl;
    
    // Definir versão do modelo
    dnn.SetModelVersion("Modelo exemplo com validação de códons");
    
    // Mostrar histórico de versões (se existir)
    dnn.ShowVersionHistory();
    
    // =================================================================
    // EXEMPLO 5: PIPELINE COMPLETO
    // =================================================================
    std::cout << "\n### EXEMPLO 5: PIPELINE COMPLETO ###" << std::endl;
    
    // Executar pipeline completo de análise
    dnn.RunCompleteAnalysisPipeline(sequences, seq_ids, "Exemplo - Análise de Códons Inválidos");
    
    // =================================================================
    // EXEMPLO 6: ANÁLISE DE EXEMPLO PREDEFINIDA
    // =================================================================
    std::cout << "\n### EXEMPLO 6: ANÁLISE DE EXEMPLO PREDEFINIDA ###" << std::endl;
    
    // Executar análise de exemplo que inclui a "Sequencia_Teste_002" mencionada
    dnn.RunExampleAnalysis();
    
    // =================================================================
    // RECOMENDAÇÕES FINAIS
    // =================================================================
    std::cout << "\n=== RECOMENDAÇÕES E PRÓXIMOS PASSOS ===" << std::endl;
    std::cout << "1. ⚠️  Códons inválidos detectados e tratados automaticamente" << std::endl;
    std::cout << "2. 📁 Modelos agora são versionados automaticamente" << std::endl;
    std::cout << "3. 📈 Logs detalhados do impacto IA salvos para análise" << std::endl;
    std::cout << "4. 📊 Scripts de visualização gerados automaticamente" << std::endl;
    std::cout << "5. 🔍 Relatórios completos disponíveis em ./dados/" << std::endl;
    
    std::cout << "\nPara visualizar gráficos:" << std::endl;
    std::cout << "cd ./dados/visualizacao && python3 plot_impact.py" << std::endl;
    
    std::cout << "\nPara gerar relatório PDF/HTML:" << std::endl;
    std::cout << "python3 generate_pdf_report.py" << std::endl;
    
    return 0;
}
