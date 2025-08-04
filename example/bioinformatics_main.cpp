#include "BioinformaticsIntegration.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "=== Sistema Integrado de Bioinformática com Deep Learning ===" << std::endl;
    std::cout << "Versão 2.0 - Melhorado com IA Avançada" << std::endl;
    std::cout << std::endl;

    try {
        // Inicializar sistema de integração
        BioinformaticsIntegration bio_system("./dados");
        
        std::cout << "Sistema inicializado com sucesso!" << std::endl;
        std::cout << "Diretório de dados: " << bio_system.GetDataDirectory() << std::endl;
        std::cout << std::endl;

        // Exemplo 1: Configurar rede neural para análise de proteínas
        std::cout << "--- Configurando Rede Neural ---" << std::endl;
        bio_system.SetupNetworkForProteins(100, 5); // 100 aminoácidos, 5 classes
        std::cout << std::endl;

        // Exemplo 2: Dados sintéticos para demonstração
        std::vector<std::string> protein_sequences = {
            "MKFLVNVALVFMVVYISYIYGCQEIFSLKIESNVAGADIIVYEQLGFLLDFHGQVVASDLLQSSGATVGISYETRM",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALAPNGGALVVGRWGS",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDD",
            "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRG",
            "MSIKKGAINNMIEEMLRSGGNPSQEELRVDVIVDNSCEPGWEYDSRMKQTLDNLVIQLTKTVEYKKFVHSSLEGVG",
            "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIP",
            "MAWHERAHFAIGLVQVFYAQGPPSLVVPSHLALLVRLVFTLLQNWPVLRTALVGVGKSFLICLLACDPGDWGWCNG",
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEQHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTC"
        };

        std::vector<int> protein_labels = {0, 1, 2, 3, 4, 0, 1, 2}; // Classes sintéticas

        std::cout << "--- Análise Estatística das Sequências ---" << std::endl;
        bio_system.AnalyzeSequenceStatistics(protein_sequences);
        std::cout << std::endl;

        // Exemplo 3: Treinamento do classificador
        std::cout << "--- Treinamento do Classificador ---" << std::endl;
        bool training_success = bio_system.TrainProteinClassifier(
            protein_sequences, 
            protein_labels, 
            "exemplo_classificador"
        );

        if (training_success) {
            std::cout << "Treinamento concluído com sucesso!" << std::endl;
        } else {
            std::cerr << "Erro no treinamento!" << std::endl;
            return 1;
        }
        std::cout << std::endl;

        // Exemplo 4: Teste do classificador
        std::cout << "--- Testando Classificador ---" << std::endl;
        std::vector<std::string> test_sequences = {
            "MKFLVNVALVFMVVYISYIYGCQEIFSLKIESN", // Sequência teste 1
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"  // Sequência teste 2
        };

        auto predictions = bio_system.ClassifySequences(test_sequences);
        
        for (size_t i = 0; i < test_sequences.size(); ++i) {
            std::cout << "Sequência " << i + 1 << ": Classe predita = " << predictions[i] << std::endl;
        }
        std::cout << std::endl;

        // Exemplo 5: Avaliação do modelo
        std::cout << "--- Avaliação do Modelo ---" << std::endl;
        std::vector<int> test_labels = {0, 1}; // Labels verdadeiros para teste
        bio_system.EvaluateModel(test_sequences, test_labels);
        std::cout << std::endl;

        // Exemplo 6: Informações da rede neural
        std::cout << "--- Informações da Rede Neural ---" << std::endl;
        auto* neural_net = bio_system.GetNeuralNetwork();
        neural_net->PrintNetworkInfo();
        std::cout << std::endl;

        // Exemplo 7: Teste com arquivo FASTA (se disponível)
        std::cout << "--- Teste com Arquivo FASTA ---" << std::endl;
        std::string fasta_file = "./dados/exemplo_sequencias.fasta";
        
        // Verificar se arquivo existe
        std::ifstream test_file(fasta_file);
        if (test_file.is_open()) {
            test_file.close();
            std::cout << "Processando arquivo FASTA: " << fasta_file << std::endl;
            
            bool fasta_success = bio_system.ProcessFASTAWithPython(fasta_file);
            if (fasta_success) {
                std::cout << "Arquivo FASTA processado com sucesso!" << std::endl;
            } else {
                std::cout << "Processamento FASTA não disponível (Python não configurado)." << std::endl;
            }
        } else {
            std::cout << "Arquivo FASTA de exemplo não encontrado. Pulando teste." << std::endl;
        }
        std::cout << std::endl;

        // Demonstração de funcionalidades avançadas da IA
        std::cout << "--- Funcionalidades Avançadas da IA ---" << std::endl;
        
        // Configurar parâmetros avançados de treinamento
        TrainingConfig advanced_config;
        advanced_config.learning_rate = 0.0005;
        advanced_config.batch_size = 8;
        advanced_config.epochs = 50;
        advanced_config.dropout_rate = 0.4;
        advanced_config.optimizer = "adam";
        advanced_config.momentum = 0.95;
        advanced_config.weight_decay = 0.0002;
        advanced_config.validation_split = 0.25;
        advanced_config.early_stopping = true;
        advanced_config.patience = 8;

        neural_net->SetTrainingConfig(advanced_config);
        std::cout << "Configuração avançada aplicada à rede neural." << std::endl;

        // Análise de importância das features (simulada)
        std::cout << "Analisando importância das features..." << std::endl;
        
        // Converter sequências para formato da rede
        std::vector<MatrixXd> feature_inputs, feature_outputs;
        for (size_t i = 0; i < protein_sequences.size(); ++i) {
            // Simulação de codificação (implementação completa seria na classe)
            MatrixXd input(100, 1);
            input.setRandom();
            MatrixXd output(5, 1);
            output.setZero();
            output(protein_labels[i], 0) = 1.0;
            
            feature_inputs.push_back(input);
            feature_outputs.push_back(output);
        }
        
        // Esta funcionalidade estaria implementada no BioinformaticsIntegration
        std::cout << "Análise de features concluída (simulada)." << std::endl;
        std::cout << std::endl;

        std::cout << "=== Demonstração Completa Finalizada ===" << std::endl;
        std::cout << "Todas as funcionalidades do sistema foram testadas!" << std::endl;
        std::cout << "Verifique os arquivos gerados em: " << bio_system.GetDataDirectory() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Erro durante a execução: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// Função auxiliar para criar dados de exemplo mais realistas
std::vector<std::string> GenerateProteinSequences(int num_sequences = 20) {
    std::vector<std::string> sequences;
    
    // Aminoácidos comuns
    std::string amino_acids = "ACDEFGHIKLMNPQRSTVWY";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> length_dist(50, 200);
    std::uniform_int_distribution<> amino_dist(0, amino_acids.size() - 1);
    
    for (int i = 0; i < num_sequences; ++i) {
        int length = length_dist(gen);
        std::string sequence;
        sequence.reserve(length);
        
        for (int j = 0; j < length; ++j) {
            sequence += amino_acids[amino_dist(gen)];
        }
        
        sequences.push_back(sequence);
    }
    
    return sequences;
}

// Função para criar labels baseados em características das sequências
std::vector<int> GenerateLabelsFromSequences(const std::vector<std::string>& sequences) {
    std::vector<int> labels;
    labels.reserve(sequences.size());
    
    for (const auto& seq : sequences) {
        // Classificação simples baseada no comprimento
        if (seq.length() < 75) {
            labels.push_back(0); // Proteína pequena
        } else if (seq.length() < 125) {
            labels.push_back(1); // Proteína média
        } else if (seq.length() < 175) {
            labels.push_back(2); // Proteína grande
        } else {
            labels.push_back(3); // Proteína muito grande
        }
        
        // Ou baseado na composição de aminoácidos
        int hydrophobic_count = 0;
        for (char aa : seq) {
            if (aa == 'A' || aa == 'V' || aa == 'I' || aa == 'L' || 
                aa == 'M' || aa == 'F' || aa == 'W' || aa == 'Y') {
                hydrophobic_count++;
            }
        }
        
        // Modifica label baseado na hidrofobicidade
        double hydrophobic_ratio = static_cast<double>(hydrophobic_count) / seq.length();
        if (hydrophobic_ratio > 0.4) {
            labels.back() = 4; // Proteína hidrofóbica
        }
    }
    
    return labels;
}
