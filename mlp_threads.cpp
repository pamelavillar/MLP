#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <functional>
#include <fstream>
#include <cmath>
#include <thread>
#include <vector>
#include <mutex>
using namespace Eigen;
using namespace std;

class MLP{
private:
    vector<MatrixXd> pesos;
    vector<VectorXd> bias;
    vector<MatrixXd> pesos_gradientes;
    vector<VectorXd> bias_gradientes;
    vector<MatrixXd> capas_output;
    vector<MatrixXd> capas_activacion;
    vector<MatrixXd> capas_errores;

    double lr = 0.0;
    vector<int> capas;
    int num_threads;
    mutex mtx;

    void init(const vector<int>& capas){
        std::random_device rd;
        std::mt19937 mt(rd());

        for(size_t c = 1; c < capas.size(); c++){
            int input = capas[c-1];
            int output = capas[c];

            double limite = sqrt(6.0/(input+output));
            std::uniform_real_distribution<double> dist(-limite,limite);

            MatrixXd p(output,input);
            for(int f = 0; f < p.rows(); ++f){
                for(int c = 0; c < p.cols(); ++c){
                    p(f,c) = dist(mt);
                }
            }

            pesos.push_back(p);
            bias.push_back(VectorXd::Zero(output));
            pesos_gradientes.push_back(MatrixXd::Zero(output,input));
            bias_gradientes.push_back(VectorXd::Zero(output));
            capas_errores.push_back(MatrixXd::Zero(1,output));
        }
    }

    void shuffle(const MatrixXd& non_shuffled_features, MatrixXd& store_shuffled_features, 
        const MatrixXd& non_shuffled_labels, MatrixXd& store_shuffled_labels){

        std::random_device rd;
        std::mt19937 mt(rd());

        std::vector<int> indicies(non_shuffled_features.rows());
        for (std::size_t i {}; i<non_shuffled_features.rows(); ++i){
            indicies[i] = i;
        }

        std::shuffle(indicies.begin(), indicies.end(), mt);

        vector<thread> threads;
        int rows_per_thread = indicies.size() / num_threads;
        
        for(int t = 0; t < num_threads; ++t){
            int start = t * rows_per_thread;
            int end = (t == num_threads - 1) ? indicies.size() : (t + 1) * rows_per_thread;
            
            threads.emplace_back([&, start, end](){
                for(int i = start; i < end; ++i){
                    store_shuffled_features.row(i) = non_shuffled_features.row(indicies[i]);
                    store_shuffled_labels.row(i) = non_shuffled_labels.row(indicies[i]);
                }
            });
        }
        
        for(auto& t : threads) t.join();
    }

    double crossEntropyPerdida(MatrixXd& y, MatrixXd& y_hat){
        const double epsilon = 1e-15;
        MatrixXd y_hat_clamped = y_hat.array().min(1.0 - epsilon).max(epsilon);
        
        double loss = 0.0;
        vector<double> losses(num_threads, 0.0);
        vector<thread> threads;
        int rows_per_thread = y.rows() / num_threads;
        
        for(int t = 0; t < num_threads; ++t){
            int start = t * rows_per_thread;
            int end = (t == num_threads - 1) ? y.rows() : (t + 1) * rows_per_thread;
            
            threads.emplace_back([&, t, start, end](){
                for(int i = start; i < end; ++i){
                    losses[t] -= (y.row(i).array() * y_hat_clamped.row(i).array().log()).sum();
                }
            });
        }
        
        for(auto& t : threads) t.join();
        
        for(double l : losses) loss += l;
        return loss;
    }

    void feedForward(MatrixXd& batch){
        capas_output.clear();
        capas_activacion.clear();

        MatrixXd input_a_capa = batch;

        for(size_t c = 0; c < pesos.size(); ++c){
            MatrixXd suma = input_a_capa * pesos[c].transpose();
            suma.rowwise() += bias[c].transpose();

            MatrixXd activaciones;

            if(c == pesos.size()-1){
                activaciones = softmax(suma);
            }
            else{
                // Paralelizar aplicación de ReLU
                activaciones = suma;
                vector<thread> threads;
                int rows_per_thread = suma.rows() / num_threads;
                
                for(int t = 0; t < num_threads; ++t){
                    int start = t * rows_per_thread;
                    int end = (t == num_threads - 1) ? suma.rows() : (t + 1) * rows_per_thread;
                    
                    threads.emplace_back([&, start, end](){
                        for(int i = start; i < end; ++i){
                            for(int j = 0; j < activaciones.cols(); ++j){
                                activaciones(i,j) = relu(suma(i,j));
                            }
                        }
                    });
                }
                
                for(auto& t : threads) t.join();
            }

            capas_output.push_back(suma);
            capas_activacion.push_back(activaciones);
            input_a_capa = activaciones;
        }
    }

    void backPropagation(MatrixXd& batch, MatrixXd& labels, int batch_size){
        for(int c = static_cast<int>(capas.size())-2; c>=0 ; --c){
            if(c == capas.size()-2){
                MatrixXd gradiente_perdida = capas_activacion.back()-labels;
                capas_errores[c] = gradiente_perdida;
                continue;
            }

            MatrixXd error_propagado = capas_errores[c+1] * pesos[c+1];
            
            capas_errores[c] = error_propagado;
            vector<thread> threads;
            int rows_per_thread = error_propagado.rows() / num_threads;
            
            for(int t = 0; t < num_threads; ++t){
                int start = t * rows_per_thread;
                int end = (t == num_threads - 1) ? error_propagado.rows() : (t + 1) * rows_per_thread;
                
                threads.emplace_back([&, start, end](){
                    for(int i = start; i < end; ++i){
                        for(int j = 0; j < capas_errores[c].cols(); ++j){
                            capas_errores[c](i,j) *= relu_prima(capas_output[c](i,j));
                        }
                    }
                });
            }
            
            for(auto& t : threads) t.join();
        }

        for(int c = 0; c < pesos.size(); ++c){
            if(c==0){
                pesos_gradientes[c] = capas_errores[c].transpose()*batch;
                pesos_gradientes[c]/=batch_size;
            }
            else{
                pesos_gradientes[c] = capas_errores[c].transpose()*capas_activacion[c-1];
                pesos_gradientes[c]/=batch_size;
            }
            bias_gradientes[c] = capas_errores[c].colwise().mean();
        }
    }

    void actualizarPesos(){
        vector<thread> threads;
        
        for(size_t i = 0; i < pesos.size(); ++i){
            threads.emplace_back([&, i](){
                pesos[i] -= lr * pesos_gradientes[i];
                bias[i] -= lr * bias_gradientes[i];
            });
        }
        
        for(auto& t : threads) t.join();
    }

    double calculo_perdida(MatrixXd& outputs, MatrixXd& labels, int batch_size){
        double perdida = crossEntropyPerdida(labels, outputs);
        perdida /= batch_size;
        return perdida;
    }

    MatrixXd softmax(MatrixXd& z){
        MatrixXd result = z;
        
        vector<thread> threads;
        int rows_per_thread = z.rows() / num_threads;
        
        for(int t = 0; t < num_threads; ++t){
            int start = t * rows_per_thread;
            int end = (t == num_threads - 1) ? z.rows() : (t + 1) * rows_per_thread;
            
            threads.emplace_back([&, start, end](){
                for(int i = start; i < end; ++i){
                    double max = z.row(i).maxCoeff();
                    VectorXd exp_row = (z.row(i).array() - max).exp();
                    double sum = exp_row.sum();
                    result.row(i) = exp_row / sum;
                }
            });
        }
        
        for(auto& t : threads) t.join();
        return result;
    }

    double relu(double z){
        return max(0.0, z);
    }

    double relu_prima(double z){
        return z>0?1.0:0.0;
    }

public:
    MLP(const vector<int>& capas, int threads = 4): capas{capas}, num_threads{threads}{
        if(num_threads <= 0){
            num_threads = thread::hardware_concurrency();
            if(num_threads == 0) num_threads = 4; // fallback
        }
        cout << "Usando " << num_threads << " threads" << endl;
        init(capas);
    }

    void learning(MatrixXd& train_data, MatrixXd& train_labels, int epocas, 
                  double learning_rate, int batch_size, bool verbose=true){
        lr = learning_rate;

        MatrixXd train_data_s(train_data.rows(), train_data.cols());
        MatrixXd train_labels_s(train_labels.rows(), train_labels.cols());

        cout << "Aprendiendo................" << endl;

        int filas = static_cast<int>(train_data_s.rows());

        for(int e = 0; e < epocas; e++){
            shuffle(train_data, train_data_s, train_labels, train_labels_s);

            double perdida = 0.0;

            for(int ini = 0; ini < filas; ini += batch_size){
                int tam_batch = min(batch_size, filas - ini);

                MatrixXd batch_actual = train_data_s.block(ini, 0, tam_batch, train_data.cols());
                MatrixXd batch_labels = train_labels_s.block(ini, 0, tam_batch, train_labels.cols());

                feedForward(batch_actual);
                backPropagation(batch_actual, batch_labels, tam_batch);
                actualizarPesos();

                perdida += calculo_perdida(capas_activacion.back(), batch_labels, tam_batch);
            }
            
            if(verbose && e % 5 == 0){
                cout << "Epoch: " << e+1 << " Perdida: " << perdida << endl;
            }
        }
        cout << "Acabo aprendizaje....." << endl;
    }

    MatrixXd predecir(MatrixXd& a_predecir){
        feedForward(a_predecir);
        return capas_activacion.back();
    }

    double exactitud(MatrixXd& labels, MatrixXd& predecido){
        int correcto = 0;
        vector<int> correctos(num_threads, 0);
        vector<thread> threads;
        int rows_per_thread = labels.rows() / num_threads;
        
        for(int t = 0; t < num_threads; ++t){
            int start = t * rows_per_thread;
            int end = (t == num_threads - 1) ? labels.rows() : (t + 1) * rows_per_thread;
            
            threads.emplace_back([&, t, start, end](){
                for(int i = start; i < end; ++i){
                    Eigen::Index label_i, pred_i;
                    labels.row(i).maxCoeff(&label_i);
                    predecido.row(i).maxCoeff(&pred_i);
                    if(label_i == pred_i) ++correctos[t];
                }
            });
        }
        
        for(auto& t : threads) t.join();
        
        for(int c : correctos) correcto += c;
        return correcto / static_cast<double>(labels.rows());
    }

    vector<MatrixXd>& obtenerPesos(){return pesos;}
};