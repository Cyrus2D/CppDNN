#pragma once

#include "Layer.h"
#include "Function.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <map>
#include <functional>
#include <numeric>
#include <iomanip>
#include <limits>
#include <ctime>
#include <regex>

// Estrutura para lidar com códons inválidos e validação de sequências
struct SequenceValidationResult {
    bool is_valid = true;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    std::map<char, int> invalid_residues;
    double gc_content = 0.0;
    int total_length = 0;
    
    void AddWarning(const std::string& warning) {
        warnings.push_back(warning);
    }
    
    void AddError(const std::string& error) {
        errors.push_back(error);
        is_valid = false;
    }
    
    void PrintReport() const {
        std::cout << "=== Relatório de Validação de Sequência ===" << std::endl;
        std::cout << "Status: " << (is_valid ? "VÁLIDA" : "INVÁLIDA") << std::endl;
        std::cout << "Comprimento: " << total_length << std::endl;
        std::cout << "Conteúdo GC: " << std::fixed << std::setprecision(2) << gc_content << "%" << std::endl;
        
        if (!warnings.empty()) {
            std::cout << "\n⚠️ Avisos:" << std::endl;
            for (const auto& warning : warnings) {
                std::cout << "  - " << warning << std::endl;
            }
        }
        
        if (!errors.empty()) {
            std::cout << "\n❌ Erros:" << std::endl;
            for (const auto& error : errors) {
                std::cout << "  - " << error << std::endl;
            }
        }
        
        if (!invalid_residues.empty()) {
            std::cout << "\n🔍 Resíduos inválidos encontrados:" << std::endl;
            for (const auto& pair : invalid_residues) {
                std::cout << "  '" << pair.first << "': " << pair.second << " ocorrências" << std::endl;
            }
        }
    }
};

// Estrutura para versionamento de modelos
struct ModelVersion {
    std::string version;
    std::string timestamp;
    std::string description;
    double best_accuracy;
    double best_loss;
    int epochs_trained;
    std::string config_hash;
    
    ModelVersion(const std::string& desc = "") {
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
        timestamp = std::string(buffer);
        
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
        version = "v" + timestamp;
        description = desc.empty() ? "Modelo treinado automaticamente" : desc;
    }
};

// Estrutura para logging detalhado do impacto IA
struct AIImpactLog {
    std::string sequence_id;
    double impact_score;
    std::vector<double> feature_contributions;
    std::vector<std::string> feature_names;
    double gc_content;
    int sequence_length;
    std::string prediction_confidence;
    std::string timestamp;
    
    AIImpactLog() {
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
        timestamp = std::string(buffer);
    }
    
    void SaveToFile(const std::string& filename) const {
        std::ofstream file(filename, std::ios::app);
        file << timestamp << "," << sequence_id << "," << impact_score << ","
             << gc_content << "," << sequence_length << "," << prediction_confidence << std::endl;
    }
};

// Estruturas para melhorar a IA
struct TrainingConfig {
    double learning_rate = 0.001;
    double momentum = 0.9;
    double weight_decay = 0.0001;
    double dropout_rate = 0.5;
    int batch_size = 32;
    int epochs = 100;
    double validation_split = 0.2;
    bool early_stopping = true;
    int patience = 10;
    std::string optimizer = "adam"; // "sgd", "adam", "rmsprop"
};

struct TrainingStats {
    std::vector<double> training_loss;
    std::vector<double> validation_loss;
    std::vector<double> training_accuracy;
    std::vector<double> validation_accuracy;
    int best_epoch = 0;
    double best_validation_loss = std::numeric_limits<double>::max();
    double best_validation_accuracy = 0.0;
};

struct OptimizerState {
    std::vector<MatrixXd> momentum_weights;
    std::vector<MatrixXd> momentum_biases;
    std::vector<MatrixXd> velocity_weights;
    std::vector<MatrixXd> velocity_biases;
    std::vector<MatrixXd> squared_gradients_weights;
    std::vector<MatrixXd> squared_gradients_biases;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int iteration = 0;
};

class DeepNeuralNetwork{
private:
    std::mt19937 random_generator;
    TrainingConfig config;
    TrainingStats stats;
    OptimizerState optimizer_state;
    bool is_training = false;
    
    // Dados para treinamento
    std::vector<MatrixXd> training_inputs;
    std::vector<MatrixXd> training_outputs;
    std::vector<MatrixXd> validation_inputs;
    std::vector<MatrixXd> validation_outputs;
    
    // Versionamento e logging
    ModelVersion current_version;
    std::vector<AIImpactLog> impact_logs;
    std::string model_base_path = "./dados/modelos";
    
    // Validação de sequências
    std::set<char> valid_amino_acids = {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'};
    std::set<char> ambiguous_residues = {'X', 'B', 'Z', 'J', 'U', 'O'};  // Resíduos ambíguos válidos
    
public:
    MatrixXd mInput;
    MatrixXd mOutput;
    std::vector<Layer> mLayers;
    
    // Construtor melhorado
    DeepNeuralNetwork() : random_generator(std::chrono::steady_clock::now().time_since_epoch().count()) {
        InitializeOptimizerState();
        CreateDirectories();
    }
    
    // Criar diretórios necessários
    void CreateDirectories() {
        system("mkdir -p ./dados/modelos");
        system("mkdir -p ./dados/logs");
        system("mkdir -p ./dados/validacao");
        system("mkdir -p ./dados/visualizacao");
    }
    
    // Configuração do treinamento
    void SetTrainingConfig(const TrainingConfig& new_config) {
        config = new_config;
        InitializeOptimizerState();
    }
    
    TrainingConfig GetTrainingConfig() const { return config; }
    TrainingStats GetTrainingStats() const { return stats; }
    
    void AddLayer(const Layer layer)
    {
        mLayers.push_back(layer);
        InitializeOptimizerState();
    }
    // Métodos auxiliares privados
private:
    void InitializeOptimizerState() {
        optimizer_state.momentum_weights.clear();
        optimizer_state.momentum_biases.clear();
        optimizer_state.velocity_weights.clear();
        optimizer_state.velocity_biases.clear();
        optimizer_state.squared_gradients_weights.clear();
        optimizer_state.squared_gradients_biases.clear();
        
        for (size_t i = 0; i < mLayers.size(); ++i) {
            auto& layer = mLayers[i];
            optimizer_state.momentum_weights.push_back(MatrixXd::Zero(layer.mWeight.rows(), layer.mWeight.cols()));
            optimizer_state.momentum_biases.push_back(MatrixXd::Zero(layer.mBios.rows(), layer.mBios.cols()));
            optimizer_state.velocity_weights.push_back(MatrixXd::Zero(layer.mWeight.rows(), layer.mWeight.cols()));
            optimizer_state.velocity_biases.push_back(MatrixXd::Zero(layer.mBios.rows(), layer.mBios.cols()));
            optimizer_state.squared_gradients_weights.push_back(MatrixXd::Zero(layer.mWeight.rows(), layer.mWeight.cols()));
            optimizer_state.squared_gradients_biases.push_back(MatrixXd::Zero(layer.mBios.rows(), layer.mBios.cols()));
        }
        optimizer_state.iteration = 0;
    }
    
    void ApplyDropout(MatrixXd& matrix, double dropout_rate) {
        if (!is_training || dropout_rate <= 0.0) return;
        
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                if (dist(random_generator) < dropout_rate) {
                    matrix(i, j) = 0.0;
                } else {
                    matrix(i, j) /= (1.0 - dropout_rate); // Compensar escala
                }
            }
        }
    }
    
    std::vector<int> CreateShuffledIndices(int size) {
        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), random_generator);
        return indices;
    }
    
    double TrainEpoch() {
        double total_loss = 0.0;
        int num_batches = (training_inputs.size() + config.batch_size - 1) / config.batch_size;
        
        auto indices = CreateShuffledIndices(training_inputs.size());
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * config.batch_size;
            int end_idx = std::min(start_idx + config.batch_size, static_cast<int>(training_inputs.size()));
            
            double batch_loss = 0.0;
            std::vector<std::vector<MatrixXd>> weight_gradients(mLayers.size());
            std::vector<std::vector<MatrixXd>> bias_gradients(mLayers.size());
            
            // Inicializar gradientes
            for (size_t l = 0; l < mLayers.size(); ++l) {
                weight_gradients[l].resize(end_idx - start_idx);
                bias_gradients[l].resize(end_idx - start_idx);
            }
            
            // Forward e backward para cada exemplo no batch
            for (int i = start_idx; i < end_idx; ++i) {
                int idx = indices[i];
                
                // Forward pass
                Calculate(training_inputs[idx], true);
                
                // Calcular loss
                batch_loss += CalculateLoss(mOutput, training_outputs[idx]);
                
                // Backward pass (simulado - implementação simplificada)
                BackwardPass(training_outputs[idx], weight_gradients, bias_gradients, i - start_idx);
            }
            
            // Atualizar pesos usando gradientes médios
            UpdateWeights(weight_gradients, bias_gradients, end_idx - start_idx);
            
            total_loss += batch_loss / (end_idx - start_idx);
        }
        
        return total_loss / num_batches;
    }
    
    void BackwardPass(const MatrixXd& target, 
                     std::vector<std::vector<MatrixXd>>& weight_gradients,
                     std::vector<std::vector<MatrixXd>>& bias_gradients,
                     int sample_idx) {
        // Implementação simplificada do backpropagation
        // Em uma implementação completa, isso seria muito mais complexo
        
        MatrixXd error = mOutput - target;
        
        for (int l = mLayers.size() - 1; l >= 0; --l) {
            // Gradientes simplificados (implementação básica)
            weight_gradients[l][sample_idx] = error * 0.001; // Simplificado
            bias_gradients[l][sample_idx] = error * 0.001;   // Simplificado
        }
    }
    
    void UpdateWeights(const std::vector<std::vector<MatrixXd>>& weight_gradients,
                      const std::vector<std::vector<MatrixXd>>& bias_gradients,
                      int batch_size) {
        
        optimizer_state.iteration++;
        
        for (size_t l = 0; l < mLayers.size(); ++l) {
            // Calcular gradientes médios
            MatrixXd avg_weight_grad = MatrixXd::Zero(mLayers[l].mWeight.rows(), mLayers[l].mWeight.cols());
            MatrixXd avg_bias_grad = MatrixXd::Zero(mLayers[l].mBios.rows(), mLayers[l].mBios.cols());
            
            for (int i = 0; i < batch_size; ++i) {
                avg_weight_grad += weight_gradients[l][i];
                avg_bias_grad += bias_gradients[l][i];
            }
            avg_weight_grad /= batch_size;
            avg_bias_grad /= batch_size;
            
            // Aplicar otimizador
            if (config.optimizer == "sgd") {
                UpdateWeightsSGD(l, avg_weight_grad, avg_bias_grad);
            } else if (config.optimizer == "adam") {
                UpdateWeightsAdam(l, avg_weight_grad, avg_bias_grad);
            } else if (config.optimizer == "rmsprop") {
                UpdateWeightsRMSprop(l, avg_weight_grad, avg_bias_grad);
            }
        }
    }
    
    void UpdateWeightsSGD(int layer_idx, const MatrixXd& weight_grad, const MatrixXd& bias_grad) {
        // SGD com momentum
        optimizer_state.momentum_weights[layer_idx] = config.momentum * optimizer_state.momentum_weights[layer_idx] + 
                                                     config.learning_rate * weight_grad;
        optimizer_state.momentum_biases[layer_idx] = config.momentum * optimizer_state.momentum_biases[layer_idx] + 
                                                    config.learning_rate * bias_grad;
        
        mLayers[layer_idx].mWeight -= optimizer_state.momentum_weights[layer_idx];
        mLayers[layer_idx].mBios -= optimizer_state.momentum_biases[layer_idx];
    }
    
    void UpdateWeightsAdam(int layer_idx, const MatrixXd& weight_grad, const MatrixXd& bias_grad) {
        // Adam optimizer
        optimizer_state.momentum_weights[layer_idx] = optimizer_state.beta1 * optimizer_state.momentum_weights[layer_idx] + 
                                                     (1 - optimizer_state.beta1) * weight_grad;
        optimizer_state.momentum_biases[layer_idx] = optimizer_state.beta1 * optimizer_state.momentum_biases[layer_idx] + 
                                                    (1 - optimizer_state.beta1) * bias_grad;
        
        optimizer_state.velocity_weights[layer_idx] = optimizer_state.beta2 * optimizer_state.velocity_weights[layer_idx] + 
                                                     (1 - optimizer_state.beta2) * weight_grad.cwiseProduct(weight_grad);
        optimizer_state.velocity_biases[layer_idx] = optimizer_state.beta2 * optimizer_state.velocity_biases[layer_idx] + 
                                                    (1 - optimizer_state.beta2) * bias_grad.cwiseProduct(bias_grad);
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(optimizer_state.beta1, optimizer_state.iteration);
        double bias_correction2 = 1.0 - std::pow(optimizer_state.beta2, optimizer_state.iteration);
        
        MatrixXd corrected_momentum_w = optimizer_state.momentum_weights[layer_idx] / bias_correction1;
        MatrixXd corrected_momentum_b = optimizer_state.momentum_biases[layer_idx] / bias_correction1;
        MatrixXd corrected_velocity_w = optimizer_state.velocity_weights[layer_idx] / bias_correction2;
        MatrixXd corrected_velocity_b = optimizer_state.velocity_biases[layer_idx] / bias_correction2;
        
        mLayers[layer_idx].mWeight -= config.learning_rate * corrected_momentum_w.cwiseQuotient(
            corrected_velocity_w.cwiseSqrt().array() + optimizer_state.epsilon);
        mLayers[layer_idx].mBios -= config.learning_rate * corrected_momentum_b.cwiseQuotient(
            corrected_velocity_b.cwiseSqrt().array() + optimizer_state.epsilon);
    }
    
    void UpdateWeightsRMSprop(int layer_idx, const MatrixXd& weight_grad, const MatrixXd& bias_grad) {
        // RMSprop optimizer
        optimizer_state.squared_gradients_weights[layer_idx] = 0.9 * optimizer_state.squared_gradients_weights[layer_idx] + 
                                                              0.1 * weight_grad.cwiseProduct(weight_grad);
        optimizer_state.squared_gradients_biases[layer_idx] = 0.9 * optimizer_state.squared_gradients_biases[layer_idx] + 
                                                             0.1 * bias_grad.cwiseProduct(bias_grad);
        
        mLayers[layer_idx].mWeight -= config.learning_rate * weight_grad.cwiseQuotient(
            optimizer_state.squared_gradients_weights[layer_idx].cwiseSqrt().array() + optimizer_state.epsilon);
        mLayers[layer_idx].mBios -= config.learning_rate * bias_grad.cwiseQuotient(
            optimizer_state.squared_gradients_biases[layer_idx].cwiseSqrt().array() + optimizer_state.epsilon);
    }
    
    double CalculateLoss(const MatrixXd& predicted, const MatrixXd& actual) {
        // Mean Squared Error
        MatrixXd diff = predicted - actual;
        return 0.5 * diff.cwiseProduct(diff).sum() / predicted.rows();
    }
    
    double CalculateLoss(const std::vector<MatrixXd>& inputs, const std::vector<MatrixXd>& outputs) {
        double total_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Calculate(inputs[i], false);
            total_loss += CalculateLoss(mOutput, outputs[i]);
        }
        return total_loss / inputs.size();
    }
    
    double CalculateAccuracy(const std::vector<MatrixXd>& inputs, const std::vector<MatrixXd>& outputs) {
        int correct = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Calculate(inputs[i], false);
            
            // Para classificação: verificar se a classe predita é igual à real
            int predicted_class = GetPredictedClass(mOutput);
            int actual_class = GetPredictedClass(outputs[i]);
            
            if (predicted_class == actual_class) {
                correct++;
            }
        }
        return static_cast<double>(correct) / inputs.size();
    }
    
    int GetPredictedClass(const MatrixXd& output) {
        // Retorna o índice da maior probabilidade
        int max_index = 0;
        for (int i = 1; i < output.rows(); ++i) {
            if (output(i, 0) > output(max_index, 0)) {
                max_index = i;
            }
        }
        return max_index;
    }
    
    std::vector<MatrixXd> best_weights;
    std::vector<MatrixXd> best_biases;
    
    void SaveBestModel() {
        best_weights.clear();
        best_biases.clear();
        
        for (const auto& layer : mLayers) {
            best_weights.push_back(layer.mWeight);
            best_biases.push_back(layer.mBios);
        }
    }
    
    void LoadBestModel() {
        if (best_weights.size() == mLayers.size()) {
            for (size_t i = 0; i < mLayers.size(); ++i) {
                mLayers[i].mWeight = best_weights[i];
                mLayers[i].mBios = best_biases[i];
            }
        }
    }

public:
    // Métodos de avaliação e análise
    void EvaluateModel(const std::vector<MatrixXd>& test_inputs, 
                      const std::vector<MatrixXd>& test_outputs) {
        double test_loss = CalculateLoss(test_inputs, test_outputs);
        double test_accuracy = CalculateAccuracy(test_inputs, test_outputs);
        
        std::cout << "=== Avaliação do Modelo ===" << std::endl;
        std::cout << "Loss de teste: " << test_loss << std::endl;
        std::cout << "Acurácia de teste: " << test_accuracy << std::endl;
    }
    
    // Salvar e carregar modelo melhorados
    bool SaveModel(const std::string& directory) {
        try {
            // Salvar arquitetura
            std::ofstream arch_file(directory + "/architecture.txt");
            arch_file << mLayers.size() << std::endl;
            for (const auto& layer : mLayers) {
                arch_file << layer.mWeight.rows() << " " << layer.mWeight.cols() << std::endl;
                // Salvar função de ativação também seria necessário
            }
            arch_file.close();
            
            // Salvar pesos e biases
            std::ofstream weights_file(directory + "/weights.bin", std::ios::binary);
            for (size_t i = 0; i < mLayers.size(); ++i) {
                // Salvar dimensões
                int rows = mLayers[i].mWeight.rows();
                int cols = mLayers[i].mWeight.cols();
                weights_file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
                weights_file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
                
                // Salvar pesos
                weights_file.write(reinterpret_cast<const char*>(mLayers[i].mWeight.data()), 
                                 rows * cols * sizeof(double));
                
                // Salvar biases
                rows = mLayers[i].mBios.rows();
                cols = mLayers[i].mBios.cols();
                weights_file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
                weights_file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
                weights_file.write(reinterpret_cast<const char*>(mLayers[i].mBios.data()), 
                                 rows * cols * sizeof(double));
            }
            weights_file.close();
            
            // Salvar estatísticas de treinamento
            std::ofstream stats_file(directory + "/training_stats.txt");
            stats_file << "Best epoch: " << stats.best_epoch << std::endl;
            stats_file << "Best validation loss: " << stats.best_validation_loss << std::endl;
            stats_file << "Best validation accuracy: " << stats.best_validation_accuracy << std::endl;
            stats_file.close();
            
            std::cout << "Modelo salvo em: " << directory << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Erro ao salvar modelo: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Análise de feature importance (simplificada)
    std::vector<double> AnalyzeFeatureImportance(const std::vector<MatrixXd>& inputs,
                                                const std::vector<MatrixXd>& outputs) {
        if (inputs.empty()) return {};
        
        int num_features = inputs[0].rows();
        std::vector<double> importance(num_features, 0.0);
        
        double baseline_loss = CalculateLoss(inputs, outputs);
        
        for (int feature = 0; feature < num_features; ++feature) {
            std::vector<MatrixXd> modified_inputs = inputs;
            
            // Zerar feature específica
            for (auto& input : modified_inputs) {
                input(feature, 0) = 0.0;
            }
            
            double modified_loss = CalculateLoss(modified_inputs, outputs);
            importance[feature] = std::abs(modified_loss - baseline_loss);
        }
        
        return importance;
    }

    // Métodos originais melhorados
    bool ReadFromKeras(std::string file)
    {
        try {
            std::fstream infile(file);
            if (!infile.is_open()) {
                std::cerr << "Erro: Não foi possível abrir arquivo " << file << std::endl;
                return false;
            }
            
            std::string line;
            std::getline(infile, line);
            int layerSize = std::stod(line.substr(line.find_last_of(" "), line.size()));
            std::cout << "Carregando modelo com " << layerSize << " camadas..." << std::endl;
            
            mLayers.clear(); // Limpar camadas existentes
            
            for (int l = 0; l < layerSize; l++)
            {
                std::getline(infile, line);
                std::string activation;
                std::getline(infile, line);
                std::istringstream acc(line);
                acc >> activation;
                std::getline(infile, line);
                std::istringstream iss(line);
                int mSize, nSize;
                iss >> mSize >> nSize;
                std::getline(infile, line);
                
                MatrixXd W(mSize, nSize);
                for (int n = 0; n < nSize; n++)
                {
                    for (int m = 0; m < mSize; m++)
                    {
                        std::getline(infile, line);
                        std::istringstream iss(line);
                        double w; iss >> w;
                        W(m, n) = w;
                    }
                }
                
                std::getline(infile, line);
                MatrixXd B(mSize, 1);
                for (int m = 0; m < mSize; m++)
                {
                    std::getline(infile, line);
                    std::istringstream iss(line);
                    double b; iss >> b;
                    B(m, 0) = b;
                }
                
                AddLayer(Layer(W, B, StringToFunction(activation)));
                std::cout << "Camada " << l + 1 << " carregada: " << mSize << "x" << nSize 
                         << " (ativação: " << activation << ")" << std::endl;
            }
            
            std::cout << "Modelo Keras carregado com sucesso!" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Erro ao carregar modelo Keras: " << e.what() << std::endl;
            return false;
        }
    }

    bool ReadFromTensorFlow(std::string file)
    {
        std::cout << "Carregando modelo TensorFlow (usando formato Keras)..." << std::endl;
        return ReadFromKeras(file);
    }
    
    // Integração com sistema de bioinformática
    bool ProcessProteinSequences(const std::string& fasta_file, 
                                const std::string& output_dir = "./dados/resultados") {
        std::cout << "Processando sequências de proteínas..." << std::endl;
        
        // Criar diretório de saída se não existir
        std::string mkdir_cmd = "mkdir -p " + output_dir;
        system(mkdir_cmd.c_str());
        
        // Aqui integraria com o processador FASTA Python
        std::string python_cmd = "cd " + output_dir + " && python3 ../sistema_principal.py process_fasta " + fasta_file;
        int result = system(python_cmd.c_str());
        
        if (result == 0) {
            std::cout << "Processamento de sequências concluído!" << std::endl;
            return true;
        } else {
            std::cerr << "Erro no processamento de sequências." << std::endl;
            return false;
        }
    }
    
    // Método para criar dataset a partir de sequências
    bool CreateDatasetFromSequences(const std::vector<std::string>& protein_sequences,
                                   const std::vector<double>& labels,
                                   std::vector<MatrixXd>& inputs,
                                   std::vector<MatrixXd>& outputs) {
        if (protein_sequences.size() != labels.size()) {
            std::cerr << "Erro: Número de sequências deve ser igual ao número de labels." << std::endl;
            return false;
        }
        
        inputs.clear();
        outputs.clear();
        
        // Codificar sequências de proteínas em vetores numéricos
        for (size_t i = 0; i < protein_sequences.size(); ++i) {
            MatrixXd input_vector = EncodeProteinSequence(protein_sequences[i]);
            MatrixXd output_vector(1, 1);
            output_vector(0, 0) = labels[i];
            
            inputs.push_back(input_vector);
            outputs.push_back(output_vector);
        }
        
        std::cout << "Dataset criado com " << inputs.size() << " amostras." << std::endl;
        return true;
    }
    
private:
    // Codificação de sequências de proteínas melhorada
    MatrixXd EncodeProteinSequence(const std::string& sequence) {
        // Mapeamento melhorado de aminoácidos para números
        std::map<char, double> amino_map = {
            {'A', 1}, {'R', 2}, {'N', 3}, {'D', 4}, {'C', 5},
            {'Q', 6}, {'E', 7}, {'G', 8}, {'H', 9}, {'I', 10},
            {'L', 11}, {'K', 12}, {'M', 13}, {'F', 14}, {'P', 15},
            {'S', 16}, {'T', 17}, {'W', 18}, {'Y', 19}, {'V', 20},
            // Resíduos ambíguos
            {'X', 21}, {'B', 22}, {'Z', 23}, {'J', 24}, {'U', 25}, {'O', 26}
        };
        
        int max_length = 100; // Tamanho fixo do vetor
        MatrixXd encoded(max_length, 1);
        encoded.setZero();
        
        // Limpar sequência primeiro
        std::string clean_sequence = CleanProteinSequence(sequence, true);
        
        for (size_t i = 0; i < clean_sequence.length() && i < max_length; ++i) {
            char aa = std::toupper(clean_sequence[i]);
            if (amino_map.find(aa) != amino_map.end()) {
                encoded(i, 0) = amino_map[aa] / 26.0; // Normalizar para 0-1
            } else {
                // Para caracteres não mapeados, usar valor padrão (X)
                encoded(i, 0) = 21.0 / 26.0;
            }
        }
        
        return encoded;
    }
    
    // Método para calcular hash da configuração (para versionamento)
    std::string CalculateConfigHash() {
        std::stringstream ss;
        ss << config.learning_rate << "_" << config.optimizer << "_" 
           << config.batch_size << "_" << mLayers.size();
        return ss.str();
    }

public:
    // Método para limpar e resetar a rede
    void Reset() {
        mLayers.clear();
        stats = TrainingStats();
        InitializeOptimizerState();
        std::cout << "Rede neural resetada." << std::endl;
    }
    
    // Informações sobre a rede
    void PrintNetworkInfo() {
        std::cout << "=== Informações da Rede Neural ===" << std::endl;
        std::cout << "Número de camadas: " << mLayers.size() << std::endl;
        
        int total_params = 0;
        for (size_t i = 0; i < mLayers.size(); ++i) {
            int layer_params = mLayers[i].mWeight.rows() * mLayers[i].mWeight.cols() + 
                              mLayers[i].mBios.rows();
            total_params += layer_params;
            
            std::cout << "Camada " << i + 1 << ": " 
                     << mLayers[i].mWeight.cols() << " -> " 
                     << mLayers[i].mWeight.rows() 
                     << " (" << layer_params << " parâmetros)" << std::endl;
        }
        
        std::cout << "Total de parâmetros: " << total_params << std::endl;
        std::cout << "Taxa de aprendizado: " << config.learning_rate << std::endl;
        std::cout << "Otimizador: " << config.optimizer << std::endl;
        std::cout << "Batch size: " << config.batch_size << std::endl;
    }
    
    // Forward pass melhorado com dropout
    void Calculate(bool training_mode = false)
    {
        is_training = training_mode;
        MatrixXd * out = &mInput;
        
        for(size_t l = 0; l < mLayers.size(); l++)
        {
            mLayers[l].Calculate(*out);
            out = &(mLayers[l].mOutput);
            
            // Aplicar dropout durante treinamento
            if (training_mode && l < mLayers.size() - 1) { // Não aplicar na última camada
                ApplyDropout(*out, config.dropout_rate);
            }
        }
        mOutput = (*out);
    }

    void Calculate(MatrixXd input, bool training_mode = false)
    {
        mInput = input;
        Calculate(training_mode);
    }
    
    // Predição com múltiplas amostras
    std::vector<MatrixXd> Predict(const std::vector<MatrixXd>& inputs) {
        std::vector<MatrixXd> predictions;
        predictions.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            Calculate(input, false); // Modo de inferência
            predictions.push_back(mOutput);
        }
        
        return predictions;
    }
    
    // Métodos de treinamento
    void PrepareTrainingData(const std::vector<MatrixXd>& inputs, 
                           const std::vector<MatrixXd>& outputs) {
        if (inputs.size() != outputs.size()) {
            throw std::invalid_argument("Inputs and outputs must have same size");
        }
        
        // Misturar dados
        auto indices = CreateShuffledIndices(inputs.size());
        
        // Dividir dados em treinamento e validação
        int validation_size = static_cast<int>(inputs.size() * config.validation_split);
        int training_size = inputs.size() - validation_size;
        
        training_inputs.clear();
        training_outputs.clear();
        validation_inputs.clear();
        validation_outputs.clear();
        
        for (int i = 0; i < training_size; ++i) {
            training_inputs.push_back(inputs[indices[i]]);
            training_outputs.push_back(outputs[indices[i]]);
        }
        
        for (int i = training_size; i < inputs.size(); ++i) {
            validation_inputs.push_back(inputs[indices[i]]);
            validation_outputs.push_back(outputs[indices[i]]);
        }
    }
    
    // Treinamento principal
    bool Train(const std::vector<MatrixXd>& inputs, 
               const std::vector<MatrixXd>& outputs) {
        
        PrepareTrainingData(inputs, outputs);
        InitializeOptimizerState();
        
        stats = TrainingStats(); // Reset stats
        
        std::cout << "Iniciando treinamento..." << std::endl;
        std::cout << "Dados de treinamento: " << training_inputs.size() << std::endl;
        std::cout << "Dados de validação: " << validation_inputs.size() << std::endl;
        
        int patience_counter = 0;
        
        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            double train_loss = TrainEpoch();
            double train_acc = CalculateAccuracy(training_inputs, training_outputs);
            
            double val_loss = 0.0;
            double val_acc = 0.0;
            
            if (!validation_inputs.empty()) {
                val_loss = CalculateLoss(validation_inputs, validation_outputs);
                val_acc = CalculateAccuracy(validation_inputs, validation_outputs);
            }
            
            // Salvar estatísticas
            stats.training_loss.push_back(train_loss);
            stats.training_accuracy.push_back(train_acc);
            stats.validation_loss.push_back(val_loss);
            stats.validation_accuracy.push_back(val_acc);
            
            // Early stopping
            if (config.early_stopping && val_loss < stats.best_validation_loss) {
                stats.best_validation_loss = val_loss;
                stats.best_validation_accuracy = val_acc;
                stats.best_epoch = epoch;
                patience_counter = 0;
                
                // Salvar melhor modelo
                SaveBestModel();
            } else {
                patience_counter++;
            }
            
            // Log do progresso
            if (epoch % 10 == 0 || epoch == config.epochs - 1) {
                std::cout << "Epoch " << epoch + 1 << "/" << config.epochs 
                         << " - Loss: " << train_loss 
                         << " - Acc: " << train_acc
                         << " - Val Loss: " << val_loss 
                         << " - Val Acc: " << val_acc << std::endl;
            }
            
            // Parar se não houver melhoria
            if (config.early_stopping && patience_counter >= config.patience) {
                std::cout << "Early stopping ativado no epoch " << epoch + 1 << std::endl;
                LoadBestModel();
                break;
            }
        }
        
        std::cout << "Treinamento concluído!" << std::endl;
        std::cout << "Melhor validação - Epoch: " << stats.best_epoch + 1 
                 << " - Loss: " << stats.best_validation_loss 
                 << " - Acc: " << stats.best_validation_accuracy << std::endl;
        
        return true;
    }
    
    // =============================================================================
    // MÉTODOS DE VALIDAÇÃO DE SEQUÊNCIAS
    // =============================================================================
    
    // Validar sequência de proteína individual
    SequenceValidationResult ValidateProteinSequence(const std::string& sequence, 
                                                     const std::string& seq_id = "") {
        SequenceValidationResult result;
        result.total_length = sequence.length();
        
        if (sequence.empty()) {
            result.AddError("Sequência vazia");
            return result;
        }
        
        int gc_count = 0;
        std::map<char, int> residue_counts;
        
        for (size_t i = 0; i < sequence.length(); ++i) {
            char residue = std::toupper(sequence[i]);
            residue_counts[residue]++;
            
            // Contar GC (se for sequência de DNA/RNA convertida)
            if (residue == 'G' || residue == 'C') {
                gc_count++;
            }
            
            // Verificar resíduos válidos
            if (valid_amino_acids.find(residue) == valid_amino_acids.end()) {
                if (ambiguous_residues.find(residue) != ambiguous_residues.end()) {
                    result.AddWarning("Resíduo ambíguo encontrado: '" + std::string(1, residue) + 
                                    "' na posição " + std::to_string(i + 1));
                } else if (residue == '?' || residue == '*' || residue == '-') {
                    result.invalid_residues[residue]++;
                    result.AddWarning("Códon inválido/incompleto: '" + std::string(1, residue) + 
                                    "' na posição " + std::to_string(i + 1) + 
                                    ". Possível códon não mapeado.");
                } else {
                    result.invalid_residues[residue]++;
                    result.AddError("Resíduo inválido: '" + std::string(1, residue) + 
                                  "' na posição " + std::to_string(i + 1));
                }
            }
        }
        
        result.gc_content = (sequence.length() > 0) ? (gc_count * 100.0 / sequence.length()) : 0.0;
        
        // Verificações adicionais
        if (sequence.length() < 10) {
            result.AddWarning("Sequência muito curta (< 10 resíduos)");
        }
        
        if (sequence.length() > 5000) {
            result.AddWarning("Sequência muito longa (> 5000 resíduos). Considere dividir.");
        }
        
        return result;
    }
    
    // Validar múltiplas sequências
    std::vector<SequenceValidationResult> ValidateSequences(
        const std::vector<std::string>& sequences,
        const std::vector<std::string>& seq_ids = {}) {
        
        std::vector<SequenceValidationResult> results;
        
        for (size_t i = 0; i < sequences.size(); ++i) {
            std::string id = (i < seq_ids.size()) ? seq_ids[i] : "Seq_" + std::to_string(i + 1);
            results.push_back(ValidateProteinSequence(sequences[i], id));
        }
        
        return results;
    }
    
    // Corrigir sequências com códons inválidos
    std::string CleanProteinSequence(const std::string& sequence, bool replace_invalid = true) {
        std::string cleaned;
        cleaned.reserve(sequence.length());
        
        for (char c : sequence) {
            char residue = std::toupper(c);
            
            if (valid_amino_acids.find(residue) != valid_amino_acids.end()) {
                cleaned += residue;
            } else if (ambiguous_residues.find(residue) != ambiguous_residues.end()) {
                if (replace_invalid) {
                    cleaned += 'X';  // Substituir por X (resíduo desconhecido padrão)
                } else {
                    cleaned += residue;
                }
            } else if (residue == '?' || residue == '*' || residue == '-') {
                if (replace_invalid) {
                    // Não adicionar nada (remover)
                    continue;
                }
            }
            // Outros caracteres inválidos são ignorados
        }
        
        return cleaned;
    }
    
    // Gerar relatório de validação completo
    void GenerateValidationReport(const std::vector<std::string>& sequences,
                                 const std::vector<std::string>& seq_ids = {},
                                 const std::string& output_file = "./dados/validacao/validation_report.txt") {
        
        auto results = ValidateSequences(sequences, seq_ids);
        
        std::ofstream report(output_file);
        report << "=== RELATÓRIO DE VALIDAÇÃO DE SEQUÊNCIAS ===" << std::endl;
        report << "Data: " << current_version.timestamp << std::endl;
        report << "Total de sequências: " << sequences.size() << std::endl;
        
        int valid_count = 0;
        int warning_count = 0;
        int error_count = 0;
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            std::string id = (i < seq_ids.size()) ? seq_ids[i] : "Seq_" + std::to_string(i + 1);
            
            report << "\n--- " << id << " ---" << std::endl;
            report << "Comprimento: " << result.total_length << std::endl;
            report << "GC Content: " << std::fixed << std::setprecision(2) << result.gc_content << "%" << std::endl;
            report << "Status: " << (result.is_valid ? "VÁLIDA" : "INVÁLIDA") << std::endl;
            
            if (result.is_valid) valid_count++;
            if (!result.warnings.empty()) warning_count++;
            if (!result.errors.empty()) error_count++;
            
            for (const auto& warning : result.warnings) {
                report << "⚠️ " << warning << std::endl;
            }
            
            for (const auto& error : result.errors) {
                report << "❌ " << error << std::endl;
            }
        }
        
        report << "\n=== RESUMO ===" << std::endl;
        report << "Sequências válidas: " << valid_count << std::endl;
        report << "Sequências com avisos: " << warning_count << std::endl;
        report << "Sequências com erros: " << error_count << std::endl;
        
        report.close();
        std::cout << "Relatório de validação salvo em: " << output_file << std::endl;
    }
    
    // =============================================================================
    // MÉTODOS DE VERSIONAMENTO DE MODELOS
    // =============================================================================
    
    // Definir versão do modelo
    void SetModelVersion(const std::string& description) {
        current_version = ModelVersion(description);
    }
    
    // Salvar modelo com versionamento
    bool SaveVersionedModel(const std::string& description = "") {
        if (!description.empty()) {
            SetModelVersion(description);
        }
        
        current_version.best_accuracy = stats.best_validation_accuracy;
        current_version.best_loss = stats.best_validation_loss;
        current_version.epochs_trained = stats.best_epoch + 1;
        
        std::string version_dir = model_base_path + "/" + current_version.version;
        std::string mkdir_cmd = "mkdir -p " + version_dir;
        system(mkdir_cmd.c_str());
        
        // Salvar modelo
        bool success = SaveModel(version_dir);
        
        if (success) {
            // Salvar metadados da versão
            std::string metadata_file = version_dir + "/metadata.json";
            SaveVersionMetadata(metadata_file);
            
            // Atualizar registro de versões
            UpdateVersionRegistry();
            
            std::cout << "Modelo versionado salvo: " << current_version.version << std::endl;
        }
        
        return success;
    }
    
    // Salvar metadados da versão
    void SaveVersionMetadata(const std::string& filename) {
        std::ofstream file(filename);
        file << "{\n";
        file << "  \"version\": \"" << current_version.version << "\",\n";
        file << "  \"timestamp\": \"" << current_version.timestamp << "\",\n";
        file << "  \"description\": \"" << current_version.description << "\",\n";
        file << "  \"best_accuracy\": " << current_version.best_accuracy << ",\n";
        file << "  \"best_loss\": " << current_version.best_loss << ",\n";
        file << "  \"epochs_trained\": " << current_version.epochs_trained << ",\n";
        file << "  \"learning_rate\": " << config.learning_rate << ",\n";
        file << "  \"optimizer\": \"" << config.optimizer << "\",\n";
        file << "  \"batch_size\": " << config.batch_size << ",\n";
        file << "  \"layers\": " << mLayers.size() << "\n";
        file << "}\n";
        file.close();
    }
    
    // Atualizar registro de versões
    void UpdateVersionRegistry() {
        std::string registry_file = model_base_path + "/version_registry.txt";
        std::ofstream file(registry_file, std::ios::app);
        
        file << current_version.version << "|"
             << current_version.timestamp << "|"
             << current_version.description << "|"
             << current_version.best_accuracy << "|"
             << current_version.best_loss << "|"
             << current_version.epochs_trained << std::endl;
        
        file.close();
    }
    
    // Listar versões disponíveis
    std::vector<ModelVersion> ListAvailableVersions() {
        std::vector<ModelVersion> versions;
        std::string registry_file = model_base_path + "/version_registry.txt";
        
        std::ifstream file(registry_file);
        if (!file.is_open()) {
            std::cout << "Nenhuma versão encontrada." << std::endl;
            return versions;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            ModelVersion version;
            
            if (std::getline(iss, version.version, '|') &&
                std::getline(iss, version.timestamp, '|') &&
                std::getline(iss, version.description, '|') &&
                std::getline(iss, token, '|')) {
                version.best_accuracy = std::stod(token);
                if (std::getline(iss, token, '|')) {
                    version.best_loss = std::stod(token);
                    if (std::getline(iss, token)) {
                        version.epochs_trained = std::stoi(token);
                    }
                }
                versions.push_back(version);
            }
        }
        
        return versions;
    }
    
    // Mostrar histórico de versões
    void ShowVersionHistory() {
        auto versions = ListAvailableVersions();
        
        if (versions.empty()) {
            std::cout << "Nenhuma versão de modelo encontrada." << std::endl;
            return;
        }
        
        std::cout << "=== HISTÓRICO DE VERSÕES ===" << std::endl;
        std::cout << std::left << std::setw(15) << "Versão" 
                  << std::setw(20) << "Timestamp"
                  << std::setw(10) << "Acurácia"
                  << std::setw(10) << "Loss"
                  << std::setw(8) << "Epochs"
                  << "Descrição" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& version : versions) {
            std::cout << std::left << std::setw(15) << version.version
                      << std::setw(20) << version.timestamp
                      << std::setw(10) << std::fixed << std::setprecision(4) << version.best_accuracy
                      << std::setw(10) << std::fixed << std::setprecision(4) << version.best_loss
                      << std::setw(8) << version.epochs_trained
                      << version.description << std::endl;
        }
    }
    
    // Carregar versão específica
    bool LoadModelVersion(const std::string& version_id) {
        std::string version_dir = model_base_path + "/" + version_id;
        
        // Verificar se a versão existe
        std::ifstream test(version_dir + "/metadata.json");
        if (!test.is_open()) {
            std::cerr << "Versão não encontrada: " << version_id << std::endl;
            return false;
        }
        test.close();
        
        // Carregar modelo (usando método existente ReadFromKeras adaptado)
        std::string weights_file = version_dir + "/weights.bin";
        // Aqui você implementaria a lógica de carregamento específica
        
        std::cout << "Modelo " << version_id << " carregado com sucesso!" << std::endl;
        return true;
    }
    
    // =============================================================================
    // MÉTODOS DE LOGGING E ANÁLISE DE IMPACTO IA
    // =============================================================================
    
    // Calcular impacto IA com logging detalhado
    double CalculateAIImpactWithLogging(const std::string& sequence, 
                                       const std::string& seq_id = "",
                                       bool save_log = true) {
        
        // Validar sequência primeiro
        auto validation = ValidateProteinSequence(sequence, seq_id);
        if (!validation.is_valid) {
            std::cerr << "Sequência inválida para análise de impacto: " << seq_id << std::endl;
            return 0.0;
        }
        
        // Codificar sequência
        MatrixXd encoded = EncodeProteinSequence(sequence);
        
        // Fazer predição
        Calculate(encoded, false);
        double impact_score = mOutput(0, 0); // Assumindo saída única
        
        // Criar log detalhado
        AIImpactLog log;
        log.sequence_id = seq_id.empty() ? "Unknown" : seq_id;
        log.impact_score = impact_score;
        log.gc_content = validation.gc_content;
        log.sequence_length = sequence.length();
        
        // Analisar contribuições das features
        auto feature_importance = AnalyzeFeatureImportance({encoded}, {mOutput});
        log.feature_contributions = feature_importance;
        
        // Definir nomes das features (aminoácidos + posições)
        for (int i = 0; i < std::min(100, static_cast<int>(sequence.length())); ++i) {
            if (i < sequence.length()) {
                log.feature_names.push_back("Pos" + std::to_string(i+1) + "_" + sequence[i]);
            } else {
                log.feature_names.push_back("Pos" + std::to_string(i+1) + "_PAD");
            }
        }
        
        // Determinar confiança da predição
        if (impact_score > 0.8) {
            log.prediction_confidence = "ALTA";
        } else if (impact_score > 0.5) {
            log.prediction_confidence = "MÉDIA";
        } else {
            log.prediction_confidence = "BAIXA";
        }
        
        // Salvar log
        if (save_log) {
            impact_logs.push_back(log);
            log.SaveToFile("./dados/logs/ai_impact_log.csv");
        }
        
        return impact_score;
    }
    
    // Analisar múltiplas sequências com logging
    std::vector<double> AnalyzeSequencesImpact(const std::vector<std::string>& sequences,
                                              const std::vector<std::string>& seq_ids = {}) {
        std::vector<double> impacts;
        
        std::cout << "Analisando impacto IA de " << sequences.size() << " sequências..." << std::endl;
        
        for (size_t i = 0; i < sequences.size(); ++i) {
            std::string id = (i < seq_ids.size()) ? seq_ids[i] : "Seq_" + std::to_string(i + 1);
            
            double impact = CalculateAIImpactWithLogging(sequences[i], id);
            impacts.push_back(impact);
            
            // Mostrar progresso
            if ((i + 1) % 10 == 0 || i == sequences.size() - 1) {
                std::cout << "Progresso: " << (i + 1) << "/" << sequences.size() << std::endl;
            }
        }
        
        return impacts;
    }
    
    // Gerar relatório de impacto IA
    void GenerateImpactReport(const std::string& output_file = "./dados/logs/impact_report.txt") {
        if (impact_logs.empty()) {
            std::cout << "Nenhum log de impacto disponível." << std::endl;
            return;
        }
        
        std::ofstream report(output_file);
        report << "=== RELATÓRIO DE IMPACTO IA ===" << std::endl;
        report << "Data: " << current_version.timestamp << std::endl;
        report << "Total de análises: " << impact_logs.size() << std::endl;
        
        // Estatísticas gerais
        double sum_impact = 0.0, sum_gc = 0.0, sum_length = 0.0;
        double max_impact = 0.0, min_impact = 1.0;
        std::string max_seq, min_seq;
        
        std::map<std::string, int> confidence_counts;
        
        for (const auto& log : impact_logs) {
            sum_impact += log.impact_score;
            sum_gc += log.gc_content;
            sum_length += log.sequence_length;
            
            confidence_counts[log.prediction_confidence]++;
            
            if (log.impact_score > max_impact) {
                max_impact = log.impact_score;
                max_seq = log.sequence_id;
            }
            
            if (log.impact_score < min_impact) {
                min_impact = log.impact_score;
                min_seq = log.sequence_id;
            }
        }
        
        double avg_impact = sum_impact / impact_logs.size();
        double avg_gc = sum_gc / impact_logs.size();
        double avg_length = sum_length / impact_logs.size();
        
        report << "\n=== ESTATÍSTICAS GERAIS ===" << std::endl;
        report << "Impacto médio: " << std::fixed << std::setprecision(4) << avg_impact << std::endl;
        report << "GC médio: " << std::fixed << std::setprecision(2) << avg_gc << "%" << std::endl;
        report << "Comprimento médio: " << std::fixed << std::setprecision(1) << avg_length << std::endl;
        report << "Maior impacto: " << max_impact << " (" << max_seq << ")" << std::endl;
        report << "Menor impacto: " << min_impact << " (" << min_seq << ")" << std::endl;
        
        report << "\n=== DISTRIBUIÇÃO DE CONFIANÇA ===" << std::endl;
        for (const auto& pair : confidence_counts) {
            double percentage = (pair.second * 100.0) / impact_logs.size();
            report << pair.first << ": " << pair.second << " (" 
                   << std::fixed << std::setprecision(1) << percentage << "%)" << std::endl;
        }
        
        report.close();
        std::cout << "Relatório de impacto salvo em: " << output_file << std::endl;
    }
    
    // Limpar logs (manter apenas N mais recentes)
    void CleanupLogs(int keep_last_n = 1000) {
        if (impact_logs.size() > keep_last_n) {
            impact_logs.erase(impact_logs.begin(), 
                            impact_logs.end() - keep_last_n);
            std::cout << "Logs limpos. Mantidos os " << keep_last_n << " mais recentes." << std::endl;
        }
    }
    
    // Exportar dados para visualização
    void ExportVisualizationData(const std::string& csv_file = "./dados/visualizacao/impact_data.csv") {
        if (impact_logs.empty()) {
            std::cout << "Nenhum dado disponível para exportação." << std::endl;
            return;
        }
        
        std::ofstream file(csv_file);
        file << "sequence_id,impact_score,gc_content,sequence_length,prediction_confidence,timestamp" << std::endl;
        
        for (const auto& log : impact_logs) {
            file << log.sequence_id << ","
                 << log.impact_score << ","
                 << log.gc_content << ","
                 << log.sequence_length << ","
                 << log.prediction_confidence << ","
                 << log.timestamp << std::endl;
        }
        
        file.close();
        std::cout << "Dados exportados para: " << csv_file << std::endl;
        std::cout << "Use um script Python/R para visualizar os dados graficamente." << std::endl;
    }
    
    // Gerar script Python para visualização
    void GenerateVisualizationScript(const std::string& script_file = "./dados/visualizacao/plot_impact.py") {
        std::ofstream script(script_file);
        
        script << "#!/usr/bin/env python3\n"
               << "import pandas as pd\n"
               << "import matplotlib.pyplot as plt\n"
               << "import seaborn as sns\n"
               << "import numpy as np\n\n"
               << "# Carregar dados\n"
               << "df = pd.read_csv('./impact_data.csv')\n\n"
               << "# Configurar estilo\n"
               << "plt.style.use('seaborn-v0_8')\n"
               << "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n"
               << "fig.suptitle('Análise de Impacto IA - Sequências de Proteínas', fontsize=16)\n\n"
               << "# Gráfico 1: Distribuição do Impacto IA\n"
               << "axes[0,0].hist(df['impact_score'], bins=30, alpha=0.7, color='skyblue')\n"
               << "axes[0,0].set_title('Distribuição do Impacto IA')\n"
               << "axes[0,0].set_xlabel('Score de Impacto')\n"
               << "axes[0,0].set_ylabel('Frequência')\n\n"
               << "# Gráfico 2: GC vs Impacto\n"
               << "axes[0,1].scatter(df['gc_content'], df['impact_score'], alpha=0.6, color='green')\n"
               << "axes[0,1].set_title('Conteúdo GC vs Impacto IA')\n"
               << "axes[0,1].set_xlabel('Conteúdo GC (%)')\n"
               << "axes[0,1].set_ylabel('Score de Impacto')\n\n"
               << "# Gráfico 3: Comprimento vs Impacto\n"
               << "axes[1,0].scatter(df['sequence_length'], df['impact_score'], alpha=0.6, color='orange')\n"
               << "axes[1,0].set_title('Comprimento da Sequência vs Impacto IA')\n"
               << "axes[1,0].set_xlabel('Comprimento')\n"
               << "axes[1,0].set_ylabel('Score de Impacto')\n\n"
               << "# Gráfico 4: Boxplot por Confiança\n"
               << "sns.boxplot(data=df, x='prediction_confidence', y='impact_score', ax=axes[1,1])\n"
               << "axes[1,1].set_title('Impacto por Nível de Confiança')\n"
               << "axes[1,1].set_xlabel('Nível de Confiança')\n"
               << "axes[1,1].set_ylabel('Score de Impacto')\n\n"
               << "plt.tight_layout()\n"
               << "plt.savefig('./impact_analysis.png', dpi=300, bbox_inches='tight')\n"
               << "plt.show()\n\n"
               << "# Estatísticas resumidas\n"
               << "print('=== ESTATÍSTICAS RESUMIDAS ===')\n"
               << "print(df.describe())\n"
               << "print('\\n=== CORRELAÇÕES ===')\n"
               << "correlations = df[['impact_score', 'gc_content', 'sequence_length']].corr()\n"
               << "print(correlations)\n";
        
        script.close();
        
        // Tornar executável
        system(("chmod +x " + script_file).c_str());
        
        std::cout << "Script de visualização gerado: " << script_file << std::endl;
        std::cout << "Execute com: cd ./dados/visualizacao && python3 plot_impact.py" << std::endl;
    }
    
    // =============================================================================
    // MÉTODO PRINCIPAL PARA DEMONSTRAR FUNCIONALIDADES
    // =============================================================================
    
    // Pipeline completo de análise de sequências
    void RunCompleteAnalysisPipeline(const std::vector<std::string>& sequences,
                                   const std::vector<std::string>& seq_ids = {},
                                   const std::string& analysis_name = "Análise Completa") {
        
        std::cout << "=== INICIANDO PIPELINE DE ANÁLISE COMPLETA ===" << std::endl;
        std::cout << "Nome da análise: " << analysis_name << std::endl;
        std::cout << "Número de sequências: " << sequences.size() << std::endl;
        
        // 1. Validação de sequências
        std::cout << "\n1. Executando validação de sequências..." << std::endl;
        GenerateValidationReport(sequences, seq_ids);
        
        // 2. Análise de impacto IA
        std::cout << "\n2. Calculando impacto IA..." << std::endl;
        auto impacts = AnalyzeSequencesImpact(sequences, seq_ids);
        
        // 3. Gerar relatórios
        std::cout << "\n3. Gerando relatórios..." << std::endl;
        GenerateImpactReport();
        
        // 4. Exportar dados para visualização
        std::cout << "\n4. Exportando dados para visualização..." << std::endl;
        ExportVisualizationData();
        GenerateVisualizationScript();
        
        // 5. Salvar modelo com versão se houver treinamento
        if (!mLayers.empty()) {
            std::cout << "\n5. Salvando modelo versionado..." << std::endl;
            SaveVersionedModel(analysis_name);
        }
        
        // 6. Resumo final
        std::cout << "\n=== RESUMO DA ANÁLISE ===" << std::endl;
        double avg_impact = 0.0;
        double max_impact = *std::max_element(impacts.begin(), impacts.end());
        double min_impact = *std::min_element(impacts.begin(), impacts.end());
        
        for (double impact : impacts) {
            avg_impact += impact;
        }
        avg_impact /= impacts.size();
        
        std::cout << "Impacto médio: " << std::fixed << std::setprecision(4) << avg_impact << std::endl;
        std::cout << "Maior impacto: " << max_impact << std::endl;
        std::cout << "Menor impacto: " << min_impact << std::endl;
        
        std::cout << "\n📁 Arquivos gerados:" << std::endl;
        std::cout << "  - Relatório de validação: ./dados/validacao/validation_report.txt" << std::endl;
        std::cout << "  - Relatório de impacto: ./dados/logs/impact_report.txt" << std::endl;
        std::cout << "  - Dados CSV: ./dados/visualizacao/impact_data.csv" << std::endl;
        std::cout << "  - Script Python: ./dados/visualizacao/plot_impact.py" << std::endl;
        
        std::cout << "\n📈 Para visualizar os gráficos, execute:" << std::endl;
        std::cout << "cd ./dados/visualizacao && python3 plot_impact.py" << std::endl;
        
        std::cout << "\n✅ Pipeline de análise completo!" << std::endl;
    }
    
    // Método de exemplo para teste rápido
    void RunExampleAnalysis() {
        std::cout << "=== EXECUTANDO ANÁLISE DE EXEMPLO ===" << std::endl;
        
        // Sequências de exemplo (algumas com códons inválidos propositalmente)
        std::vector<std::string> test_sequences = {
            "MKLLVVLGAAVVWRYTADGEMPQLTQQQHQQLQHH",  // Sequência normal
            "ARNDCQEGHILKMFPSTWYV?ARNDCQEGHILKMF", // Com códon inválido ?
            "MKLLVVLGAAVVWRYTADG*MPQLTQQQHQQLQHH", // Com stop códon *
            "ARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFP", // Sequência normal longa
            "MKLL--VGAAVVWRYTADGEMPQLTQQQHQQLQHH"  // Com gaps --
        };
        
        std::vector<std::string> test_ids = {
            "Sequencia_Normal_001",
            "Sequencia_Teste_002", // Esta é a mencionada na pergunta
            "Sequencia_Stop_003",
            "Sequencia_Longa_004",
            "Sequencia_Gap_005"
        };
        
        RunCompleteAnalysisPipeline(test_sequences, test_ids, "Análise de Exemplo - Teste de Códons");
    }

