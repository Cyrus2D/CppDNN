#pragma once

#include "DeepNueralNetwork.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <algorithm>
#include <numeric>

class BioinformaticsIntegration {
private:
    DeepNeuralNetwork* neural_network;
    std::string data_directory;
    std::map<std::string, std::vector<std::string>> protein_databases;
    
public:
    BioinformaticsIntegration(const std::string& data_dir = "./dados") 
        : data_directory(data_dir) {
        neural_network = new DeepNeuralNetwork();
        InitializeDirectories();
    }
    
    ~BioinformaticsIntegration() {
        delete neural_network;
    }
    
    void InitializeDirectories() {
        // Criar estrutura de diretórios
        std::vector<std::string> dirs = {
            data_directory,
            data_directory + "/modelos",
            data_directory + "/treinamento", 
            data_directory + "/sequencias",
            data_directory + "/resultados",
            data_directory + "/datasets"
        };
        
        for (const auto& dir : dirs) {
            std::string cmd = "mkdir -p " + dir;
            system(cmd.c_str());
        }
        
        std::cout << "Estrutura de diretórios criada em: " << data_directory << std::endl;
    }
    
    // Configurar rede neural para bioinformática
    void SetupNetworkForProteins(int sequence_length = 100, int num_classes = 10) {
        neural_network->Reset();
        
        // Configuração para análise de proteínas
        TrainingConfig config;
        config.learning_rate = 0.001;
        config.batch_size = 16;
        config.epochs = 200;
        config.dropout_rate = 0.3;
        config.optimizer = "adam";
        config.validation_split = 0.2;
        config.early_stopping = true;
        config.patience = 15;
        
        neural_network->SetTrainingConfig(config);
        
        // Adicionar camadas adequadas para sequências de proteínas
        MatrixXd W1 = MatrixXd::Random(128, sequence_length) * 0.1;
        MatrixXd B1 = MatrixXd::Zero(128, 1);
        neural_network->AddLayer(Layer(W1, B1, ReLU));
        
        MatrixXd W2 = MatrixXd::Random(64, 128) * 0.1;
        MatrixXd B2 = MatrixXd::Zero(64, 1);
        neural_network->AddLayer(Layer(W2, B2, ReLU));
        
        MatrixXd W3 = MatrixXd::Random(32, 64) * 0.1;
        MatrixXd B3 = MatrixXd::Zero(32, 1);
        neural_network->AddLayer(Layer(W3, B3, ReLU));
        
        MatrixXd W4 = MatrixXd::Random(num_classes, 32) * 0.1;
        MatrixXd B4 = MatrixXd::Zero(num_classes, 1);
        neural_network->AddLayer(Layer(W4, B4, Softmax));
        
        std::cout << "Rede neural configurada para análise de proteínas:" << std::endl;
        neural_network->PrintNetworkInfo();
    }
    
    // Integração com Python para processamento FASTA
    bool ProcessFASTAWithPython(const std::string& fasta_file) {
        std::string python_script = data_directory + "/../sistema_principal.py";
        std::string cmd = "cd " + data_directory + " && python3 " + python_script + 
                         " process_fasta " + fasta_file;
        
        std::cout << "Executando: " << cmd << std::endl;
        int result = system(cmd.c_str());
        
        if (result == 0) {
            std::cout << "Processamento FASTA concluído com sucesso!" << std::endl;
            return LoadProcessedSequences();
        } else {
            std::cerr << "Erro no processamento FASTA." << std::endl;
            return false;
        }
    }
    
    // Carregar sequências processadas
    bool LoadProcessedSequences() {
        std::string sequences_file = data_directory + "/sequencias_processadas.txt";
        std::ifstream file(sequences_file);
        
        if (!file.is_open()) {
            std::cerr << "Erro: Não foi possível abrir " << sequences_file << std::endl;
            return false;
        }
        
        std::string line;
        std::string current_protein;
        std::vector<std::string> sequences;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            if (line[0] == '>') {
                if (!current_protein.empty()) {
                    sequences.push_back(current_protein);
                    current_protein.clear();
                }
            } else {
                current_protein += line;
            }
        }
        
        if (!current_protein.empty()) {
            sequences.push_back(current_protein);
        }
        
        file.close();
        
        std::cout << "Carregadas " << sequences.size() << " sequências de proteínas." << std::endl;
        protein_databases["processed"] = sequences;
        
        return true;
    }
    
    // Treinar modelo com sequências de proteínas
    bool TrainProteinClassifier(const std::vector<std::string>& sequences,
                               const std::vector<int>& labels,
                               const std::string& model_name = "protein_classifier") {
        
        if (sequences.size() != labels.size()) {
            std::cerr << "Erro: Número de sequências deve ser igual ao número de labels." << std::endl;
            return false;
        }
        
        std::vector<MatrixXd> inputs, outputs;
        
        // Converter sequências e labels para formato da rede neural
        for (size_t i = 0; i < sequences.size(); ++i) {
            MatrixXd input = EncodeProteinSequence(sequences[i]);
            MatrixXd output = CreateOneHotVector(labels[i], GetNumClasses(labels));
            
            inputs.push_back(input);
            outputs.push_back(output);
        }
        
        std::cout << "Iniciando treinamento do classificador de proteínas..." << std::endl;
        std::cout << "Dataset: " << inputs.size() << " amostras" << std::endl;
        
        bool success = neural_network->Train(inputs, outputs);
        
        if (success) {
            // Salvar modelo treinado
            std::string model_dir = data_directory + "/modelos/" + model_name;
            std::string mkdir_cmd = "mkdir -p " + model_dir;
            system(mkdir_cmd.c_str());
            
            neural_network->SaveModel(model_dir);
            SaveTrainingInfo(model_dir, sequences.size(), GetNumClasses(labels));
            
            std::cout << "Modelo salvo em: " << model_dir << std::endl;
            return true;
        }
        
        return false;
    }
    
    // Classificar novas sequências
    std::vector<int> ClassifySequences(const std::vector<std::string>& sequences) {
        std::vector<int> predictions;
        predictions.reserve(sequences.size());
        
        for (const auto& sequence : sequences) {
            MatrixXd input = EncodeProteinSequence(sequence);
            neural_network->Calculate(input, false);
            
            // Encontrar classe com maior probabilidade
            int predicted_class = 0;
            double max_prob = neural_network->mOutput(0, 0);
            
            for (int i = 1; i < neural_network->mOutput.rows(); ++i) {
                if (neural_network->mOutput(i, 0) > max_prob) {
                    max_prob = neural_network->mOutput(i, 0);
                    predicted_class = i;
                }
            }
            
            predictions.push_back(predicted_class);
        }
        
        return predictions;
    }
    
    // Análise estatística das sequências
    void AnalyzeSequenceStatistics(const std::vector<std::string>& sequences) {
        if (sequences.empty()) {
            std::cout << "Nenhuma sequência para analisar." << std::endl;
            return;
        }
        
        std::cout << "=== Análise Estatística das Sequências ===" << std::endl;
        
        // Estatísticas básicas
        std::vector<int> lengths;
        std::map<char, int> amino_count;
        
        for (const auto& seq : sequences) {
            lengths.push_back(seq.length());
            for (char aa : seq) {
                amino_count[std::toupper(aa)]++;
            }
        }
        
        // Comprimento das sequências
        int total_length = std::accumulate(lengths.begin(), lengths.end(), 0);
        double avg_length = static_cast<double>(total_length) / sequences.size();
        
        auto min_max = std::minmax_element(lengths.begin(), lengths.end());
        
        std::cout << "Número de sequências: " << sequences.size() << std::endl;
        std::cout << "Comprimento médio: " << avg_length << std::endl;
        std::cout << "Comprimento mínimo: " << *min_max.first << std::endl;
        std::cout << "Comprimento máximo: " << *min_max.second << std::endl;
        
        // Composição de aminoácidos
        std::cout << "\nComposição de aminoácidos:" << std::endl;
        for (const auto& pair : amino_count) {
            double percentage = (static_cast<double>(pair.second) / total_length) * 100.0;
            std::cout << pair.first << ": " << pair.second 
                     << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
        }
        
        // Salvar relatório
        SaveStatisticsReport(sequences, lengths, amino_count);
    }
    
    // Avaliar desempenho do modelo
    void EvaluateModel(const std::vector<std::string>& test_sequences,
                      const std::vector<int>& test_labels) {
        
        std::vector<MatrixXd> test_inputs, test_outputs;
        
        for (size_t i = 0; i < test_sequences.size(); ++i) {
            MatrixXd input = EncodeProteinSequence(test_sequences[i]);
            MatrixXd output = CreateOneHotVector(test_labels[i], GetNumClasses(test_labels));
            
            test_inputs.push_back(input);
            test_outputs.push_back(output);
        }
        
        neural_network->EvaluateModel(test_inputs, test_outputs);
        
        // Matriz de confusão simplificada
        auto predictions = ClassifySequences(test_sequences);
        CreateConfusionMatrix(test_labels, predictions);
    }
    
private:
    // Codificação de sequências de proteínas
    MatrixXd EncodeProteinSequence(const std::string& sequence) {
        std::map<char, double> amino_map = {
            {'A', 1}, {'R', 2}, {'N', 3}, {'D', 4}, {'C', 5},
            {'Q', 6}, {'E', 7}, {'G', 8}, {'H', 9}, {'I', 10},
            {'L', 11}, {'K', 12}, {'M', 13}, {'F', 14}, {'P', 15},
            {'S', 16}, {'T', 17}, {'W', 18}, {'Y', 19}, {'V', 20}
        };
        
        int max_length = 100;
        MatrixXd encoded(max_length, 1);
        encoded.setZero();
        
        for (size_t i = 0; i < sequence.length() && i < max_length; ++i) {
            char aa = std::toupper(sequence[i]);
            if (amino_map.find(aa) != amino_map.end()) {
                encoded(i, 0) = amino_map[aa] / 20.0; // Normalizar [0, 1]
            }
        }
        
        return encoded;
    }
    
    // Criar vetor one-hot
    MatrixXd CreateOneHotVector(int class_idx, int num_classes) {
        MatrixXd one_hot(num_classes, 1);
        one_hot.setZero();
        if (class_idx >= 0 && class_idx < num_classes) {
            one_hot(class_idx, 0) = 1.0;
        }
        return one_hot;
    }
    
    // Obter número de classes únicas
    int GetNumClasses(const std::vector<int>& labels) {
        if (labels.empty()) return 1;
        return *std::max_element(labels.begin(), labels.end()) + 1;
    }
    
    // Salvar informações do treinamento
    void SaveTrainingInfo(const std::string& model_dir, int num_samples, int num_classes) {
        std::ofstream info_file(model_dir + "/training_info.txt");
        info_file << "=== Informações do Treinamento ===" << std::endl;
        info_file << "Data: " << GetCurrentDateTime() << std::endl;
        info_file << "Número de amostras: " << num_samples << std::endl;
        info_file << "Número de classes: " << num_classes << std::endl;
        
        auto config = neural_network->GetTrainingConfig();
        info_file << "Taxa de aprendizado: " << config.learning_rate << std::endl;
        info_file << "Batch size: " << config.batch_size << std::endl;
        info_file << "Épocas: " << config.epochs << std::endl;
        info_file << "Otimizador: " << config.optimizer << std::endl;
        
        auto stats = neural_network->GetTrainingStats();
        info_file << "\n=== Resultados ===" << std::endl;
        info_file << "Melhor época: " << stats.best_epoch << std::endl;
        info_file << "Melhor loss de validação: " << stats.best_validation_loss << std::endl;
        info_file << "Melhor acurácia de validação: " << stats.best_validation_accuracy << std::endl;
        
        info_file.close();
    }
    
    // Salvar relatório estatístico
    void SaveStatisticsReport(const std::vector<std::string>& sequences,
                             const std::vector<int>& lengths,
                             const std::map<char, int>& amino_count) {
        
        std::string report_file = data_directory + "/resultados/statistics_report.txt";
        std::ofstream file(report_file);
        
        file << "=== Relatório Estatístico das Sequências ===" << std::endl;
        file << "Data: " << GetCurrentDateTime() << std::endl;
        file << "Número de sequências: " << sequences.size() << std::endl;
        
        int total_length = std::accumulate(lengths.begin(), lengths.end(), 0);
        double avg_length = static_cast<double>(total_length) / sequences.size();
        auto min_max = std::minmax_element(lengths.begin(), lengths.end());
        
        file << "Comprimento médio: " << avg_length << std::endl;
        file << "Comprimento mínimo: " << *min_max.first << std::endl;
        file << "Comprimento máximo: " << *min_max.second << std::endl;
        
        file << "\nComposição de aminoácidos:" << std::endl;
        for (const auto& pair : amino_count) {
            double percentage = (static_cast<double>(pair.second) / total_length) * 100.0;
            file << pair.first << ": " << pair.second 
                 << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
        }
        
        file.close();
        std::cout << "Relatório salvo em: " << report_file << std::endl;
    }
    
    // Criar matriz de confusão
    void CreateConfusionMatrix(const std::vector<int>& true_labels,
                              const std::vector<int>& predicted_labels) {
        if (true_labels.size() != predicted_labels.size()) return;
        
        int num_classes = GetNumClasses(true_labels);
        std::vector<std::vector<int>> matrix(num_classes, std::vector<int>(num_classes, 0));
        
        for (size_t i = 0; i < true_labels.size(); ++i) {
            if (true_labels[i] < num_classes && predicted_labels[i] < num_classes) {
                matrix[true_labels[i]][predicted_labels[i]]++;
            }
        }
        
        std::cout << "\n=== Matriz de Confusão ===" << std::endl;
        std::cout << "Verdadeiro \\ Predito: ";
        for (int i = 0; i < num_classes; ++i) {
            std::cout << "C" << i << "\t";
        }
        std::cout << std::endl;
        
        for (int i = 0; i < num_classes; ++i) {
            std::cout << "C" << i << ":\t\t";
            for (int j = 0; j < num_classes; ++j) {
                std::cout << matrix[i][j] << "\t";
            }
            std::cout << std::endl;
        }
    }
    
    // Obter data e hora atual
    std::string GetCurrentDateTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

public:
    // Método principal para processamento completo
    bool RunCompleteAnalysis(const std::string& fasta_file, 
                           const std::vector<int>& labels,
                           const std::string& model_name = "protein_analysis") {
        
        std::cout << "=== Iniciando Análise Completa de Bioinformática ===" << std::endl;
        
        // 1. Processar arquivo FASTA
        if (!ProcessFASTAWithPython(fasta_file)) {
            std::cerr << "Erro no processamento do arquivo FASTA." << std::endl;
            return false;
        }
        
        // 2. Carregar sequências processadas
        if (protein_databases.find("processed") == protein_databases.end()) {
            std::cerr << "Erro: Sequências não foram carregadas." << std::endl;
            return false;
        }
        
        auto& sequences = protein_databases["processed"];
        
        // 3. Análise estatística
        AnalyzeSequenceStatistics(sequences);
        
        // 4. Configurar e treinar rede neural
        SetupNetworkForProteins(100, GetNumClasses(labels));
        
        if (!TrainProteinClassifier(sequences, labels, model_name)) {
            std::cerr << "Erro no treinamento do modelo." << std::endl;
            return false;
        }
        
        // 5. Avaliação do modelo (se houver dados de teste)
        if (sequences.size() > 10) { // Usar parte dos dados para teste
            int test_size = sequences.size() / 5; // 20% para teste
            std::vector<std::string> test_sequences(sequences.end() - test_size, sequences.end());
            std::vector<int> test_labels(labels.end() - test_size, labels.end());
            
            EvaluateModel(test_sequences, test_labels);
        }
        
        std::cout << "=== Análise Completa Finalizada ===" << std::endl;
        return true;
    }
    
    // Accessor para a rede neural
    DeepNeuralNetwork* GetNeuralNetwork() { return neural_network; }
    
    // Obter diretório de dados
    std::string GetDataDirectory() const { return data_directory; }
};
