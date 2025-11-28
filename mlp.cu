#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void multiplicarMatrices(const float* A, const float* B, float* C,
                                    int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void multiplicarMatricesRapido(const float* A, const float* B, float* C,
                                          int m, int k, int n) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;
    for (int tile = 0; tile < (k + 15) / 16; tile++) {
        if (row < m && tile * 16 + threadIdx.x < k)
            As[threadIdx.y][threadIdx.x] = A[row * k + tile * 16 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < n && tile * 16 + threadIdx.y < k)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int i = 0; i < 16; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

__global__ void activarReLU(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void derivadaRELU(const float* data, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = data[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

__global__ void ponerBias(float* data, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        data[idx] += bias[col];
    }
}

__global__ void softmaxFilas(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float* row_data = data + row * cols;
        float max_val = row_data[0];
        for (int i = 1; i < cols; i++) {
            max_val = fmaxf(max_val, row_data[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            row_data[i] = expf(row_data[i] - max_val);
            sum += row_data[i];
        }
        for (int i = 0; i < cols; i++) {
            row_data[i] /= sum;
        }
    }
}

__global__ void perdidaCruzada(const float* y_pred, const float* y_true,
                               float* losses, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float loss = 0.0f;
        for (int i = 0; i < cols; i++) {
            float pred = fmaxf(fminf(y_pred[row * cols + i], 1.0f - 1e-7f), 1e-7f);
            loss -= y_true[row * cols + i] * logf(pred);
        }
        losses[row] = loss;
    }
}

__global__ void gradienteSalida(const float* y_pred, const float* y_true,
                                float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradient[idx] = y_pred[idx] - y_true[idx];
    }
}

__global__ void multiplicarElementos(float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= b[idx];
    }
}

__global__ void actualizarPesosGPU(float* weights, const float* gradients,
                                   float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * gradients[idx];
    }
}

__global__ void escalarDatos(float* data, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}

__global__ void gradienteBias(const float* errors, float* bias_grad,
                              int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += errors[row * cols + col];
        }
        bias_grad[col] = sum / rows;
    }
}

__global__ void transponer(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

class MLP {
private:
    vector<int> capas;
    vector<float*> pesos;
    vector<float*> bias;
    vector<float*> pesos_gradientes;
    vector<float*> bias_gradientes;
    vector<float*> capas_output;
    vector<float*> capas_activacion;
    vector<float*> capas_errores;
    float lr;

    void init(const vector<int>& capas) {
        random_device rd;
        mt19937 gen(rd());
        for (size_t i = 1; i < capas.size(); i++) {
            int input = capas[i-1];
            int output = capas[i];
            float limite = sqrt(6.0f / (input + output));
            uniform_real_distribution<float> dist(-limite, limite);
            vector<float> h_pesos(output * input);
            for (int j = 0; j < output * input; j++) {
                h_pesos[j] = dist(gen);
            }
            float* d_peso;
            CUDA_CHECK(cudaMalloc(&d_peso, output * input * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_peso, h_pesos.data(),
                                  output * input * sizeof(float),
                                  cudaMemcpyHostToDevice));
            pesos.push_back(d_peso);
            float* d_bias_layer;
            CUDA_CHECK(cudaMalloc(&d_bias_layer, output * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_bias_layer, 0, output * sizeof(float)));
            bias.push_back(d_bias_layer);
            float* d_peso_grad;
            CUDA_CHECK(cudaMalloc(&d_peso_grad, output * input * sizeof(float)));
            pesos_gradientes.push_back(d_peso_grad);
            float* d_bias_grad_layer;
            CUDA_CHECK(cudaMalloc(&d_bias_grad_layer, output * sizeof(float)));
            bias_gradientes.push_back(d_bias_grad_layer);
            float* d_error;
            CUDA_CHECK(cudaMalloc(&d_error, 256 * output * sizeof(float)));
            capas_errores.push_back(d_error);
        }
    }

public:
    MLP(const vector<int>& capas) : capas(capas) {
        init(capas);
        cout << "MLP CUDA inicializado: ";
        for (int c : capas) cout << c << " ";
        cout << endl;
    }

    ~MLP() {
        for (auto ptr : pesos) cudaFree(ptr);
        for (auto ptr : bias) cudaFree(ptr);
        for (auto ptr : pesos_gradientes) cudaFree(ptr);
        for (auto ptr : bias_gradientes) cudaFree(ptr);
        for (auto ptr : capas_activacion) cudaFree(ptr);
        for (auto ptr : capas_output) cudaFree(ptr);
        for (auto ptr : capas_errores) cudaFree(ptr);
    }

    void feedForward(float* d_input, int batch_size) {
        for (auto ptr : capas_activacion) cudaFree(ptr);
        for (auto ptr : capas_output) cudaFree(ptr);
        capas_activacion.clear();
        capas_output.clear();
        float* d_current_input = d_input;
        for (size_t layer = 0; layer < pesos.size(); layer++) {
            int input_size = capas[layer];
            int output_size = capas[layer + 1];
            float* d_output;
            float* d_activation;
            CUDA_CHECK(cudaMalloc(&d_output, batch_size * output_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_activation, batch_size * output_size * sizeof(float)));
            float* d_W_transpose;
            CUDA_CHECK(cudaMalloc(&d_W_transpose, input_size * output_size * sizeof(float)));
            dim3 transposeBlock(16, 16);
            dim3 transposeGrid((input_size + 15) / 16, (output_size + 15) / 16);
            transponer<<<transposeGrid, transposeBlock>>>(pesos[layer], d_W_transpose, output_size, input_size);
            dim3 blockDim(16, 16);
            dim3 gridDim((output_size + 15) / 16, (batch_size + 15) / 16);
            multiplicarMatricesRapido<<<gridDim, blockDim>>>(d_current_input, d_W_transpose, d_output, batch_size, input_size, output_size);
            cudaFree(d_W_transpose);
            int total = batch_size * output_size;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            ponerBias<<<blocks, threads>>>(d_output, bias[layer], batch_size, output_size);
            CUDA_CHECK(cudaMemcpy(d_activation, d_output,
                                  batch_size * output_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            if (layer == pesos.size() - 1) {
                softmaxFilas<<<batch_size, 1>>>(d_activation, batch_size, output_size);
            } else {
                activarReLU<<<blocks, threads>>>(d_activation, total);
            }
            capas_output.push_back(d_output);
            capas_activacion.push_back(d_activation);
            d_current_input = d_activation;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void backPropagation(float* d_input, float* d_labels, int batch_size) {
        int num_layers = pesos.size();
        int output_size = capas.back();
        int total = batch_size * output_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        gradienteSalida<<<blocks, threads>>>(capas_activacion.back(), d_labels, capas_errores.back(), total);
        for (int layer = num_layers - 1; layer >= 0; layer--) {
            int input_size = capas[layer];
            int output_size = capas[layer + 1];
            float* d_input_to_layer = (layer == 0) ? d_input : capas_activacion[layer - 1];
            float* d_errors_T;
            CUDA_CHECK(cudaMalloc(&d_errors_T, output_size * batch_size * sizeof(float)));
            dim3 transposeBlock(16, 16);
            dim3 transposeGrid((output_size + 15) / 16, (batch_size + 15) / 16);
            transponer<<<transposeGrid, transposeBlock>>>(capas_errores[layer], d_errors_T, batch_size, output_size);
            dim3 blockDim(16, 16);
            dim3 gridDim((input_size + 15) / 16, (output_size + 15) / 16);
            multiplicarMatricesRapido<<<gridDim, blockDim>>>(d_errors_T, d_input_to_layer, pesos_gradientes[layer], output_size, batch_size, input_size);
            cudaFree(d_errors_T);
            int weight_size = output_size * input_size;
            blocks = (weight_size + threads - 1) / threads;
            escalarDatos<<<blocks, threads>>>(pesos_gradientes[layer], 1.0f / batch_size, weight_size);
            blocks = (output_size + threads - 1) / threads;
            gradienteBias<<<blocks, threads>>>(capas_errores[layer], bias_gradientes[layer], batch_size, output_size);
            if (layer > 0) {
                float* d_error_prev = capas_errores[layer - 1];
                gridDim = dim3((input_size + 15) / 16, (batch_size + 15) / 16);
                multiplicarMatricesRapido<<<gridDim, blockDim>>>(capas_errores[layer], pesos[layer], d_error_prev, batch_size, output_size, input_size);
                float* d_relu_deriv;
                int prev_total = batch_size * input_size;
                CUDA_CHECK(cudaMalloc(&d_relu_deriv, prev_total * sizeof(float)));
                blocks = (prev_total + threads - 1) / threads;
                derivadaRELU<<<blocks, threads>>>(capas_output[layer - 1], d_relu_deriv, prev_total);
                multiplicarElementos<<<blocks, threads>>>(d_error_prev, d_relu_deriv, prev_total);
                cudaFree(d_relu_deriv);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void actualizarPesos() {
        int threads = 256;
        for (size_t layer = 0; layer < pesos.size(); layer++) {
            int input_size = capas[layer];
            int output_size = capas[layer + 1];
            int weight_size = input_size * output_size;
            int blocks = (weight_size + threads - 1) / threads;
            actualizarPesosGPU<<<blocks, threads>>>(pesos[layer], pesos_gradientes[layer], lr, weight_size);
            blocks = (output_size + threads - 1) / threads;
            actualizarPesosGPU<<<blocks, threads>>>(bias[layer], bias_gradientes[layer], lr, output_size);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float calculo_perdida(float* d_labels, int batch_size) {
        int output_size = capas.back();
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        float* d_losses;
        CUDA_CHECK(cudaMalloc(&d_losses, batch_size * sizeof(float)));
        perdidaCruzada<<<blocks, threads>>>(capas_activacion.back(), d_labels, d_losses, batch_size, output_size);
        vector<float> h_losses(batch_size);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses,
                             batch_size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        float total_loss = 0.0f;
        for (float loss : h_losses) {
            total_loss += loss;
        }
        cudaFree(d_losses);
        return total_loss / batch_size;
    }

    void learning(float* h_train_data, float* h_train_labels,
                  int num_samples, int input_size, int num_classes,
                  int epochs, float learning_rate, int batch_size) {
        lr = learning_rate;
        cout << "Entrenando con " << num_samples << " muestras..." << endl;
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epoch_loss = 0.0f;
            int num_batches = (num_samples + batch_size - 1) / batch_size;
            for (int batch = 0; batch < num_batches; batch++) {
                int start = batch * batch_size;
                int current_batch_size = min(batch_size, num_samples - start);
                float* d_batch_data;
                float* d_batch_labels;
                CUDA_CHECK(cudaMalloc(&d_batch_data,
                                     current_batch_size * input_size * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_batch_labels,
                                     current_batch_size * num_classes * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_batch_data,
                                     h_train_data + start * input_size,
                                     current_batch_size * input_size * sizeof(float),
                                     cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_batch_labels,
                                     h_train_labels + start * num_classes,
                                     current_batch_size * num_classes * sizeof(float),
                                     cudaMemcpyHostToDevice));
                feedForward(d_batch_data, current_batch_size);
                float loss = calculo_perdida(d_batch_labels, current_batch_size);
                backPropagation(d_batch_data, d_batch_labels, current_batch_size);
                actualizarPesos();
                epoch_loss += loss;
                cudaFree(d_batch_data);
                cudaFree(d_batch_labels);
            }
            if (epoch % 5 == 0) {
                cout << "Epoch " << epoch + 1 << "/" << epochs
                     << " - Loss: " << epoch_loss / num_batches << endl;
            }
        }
        cout << "Entrenamiento completado!" << endl;
    }

    float* predecir(float* h_test_data, int num_samples, int input_size) {
        float* d_test_data;
        CUDA_CHECK(cudaMalloc(&d_test_data, num_samples * input_size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_test_data, h_test_data,
                             num_samples * input_size * sizeof(float),
                             cudaMemcpyHostToDevice));
        feedForward(d_test_data, num_samples);
        int output_size = capas.back();
        float* h_predictions = new float[num_samples * output_size];
        CUDA_CHECK(cudaMemcpy(h_predictions, capas_activacion.back(),
                             num_samples * output_size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        cudaFree(d_test_data);
        return h_predictions;
    }

    float exactitud(float* h_predictions, float* h_labels,
                    int num_samples, int num_classes) {
        int correct = 0;
        for (int i = 0; i < num_samples; i++) {
            int pred_class = 0;
            int true_class = 0;
            float max_pred = h_predictions[i * num_classes];
            float max_true = h_labels[i * num_classes];
            for (int j = 1; j < num_classes; j++) {
                if (h_predictions[i * num_classes + j] > max_pred) {
                    max_pred = h_predictions[i * num_classes + j];
                    pred_class = j;
                }
                if (h_labels[i * num_classes + j] > max_true) {
                    max_true = h_labels[i * num_classes + j];
                    true_class = j;
                }
            }
            if (pred_class == true_class) correct++;
        }
        return (float)correct / num_samples;
    }
    void guardarPesos(const string& nombre_archivo) {
    ofstream file(nombre_archivo);
    if (!file.is_open()) {
        cout << "No se pudo abrir el archivo para guardar pesos" << endl;
        return;
    }

    file << "{\n";
    file << "  \"capas\": [";
    for (size_t i = 0; i < capas.size(); i++) {
        file << capas[i];
        if (i < capas.size() - 1) file << ", ";
    }
    file << "],\n";

    file << "  \"pesos\": [\n";

    for (size_t layer = 0; layer < pesos.size(); layer++) {

        int input = capas[layer];
        int output = capas[layer + 1];
        int size = input * output;

        vector<float> h_pesos(size);

        CUDA_CHECK(cudaMemcpy(h_pesos.data(), pesos[layer],
                              size * sizeof(float),
                              cudaMemcpyDeviceToHost));

        file << "    [";
        for (int i = 0; i < size; i++) {
            file << h_pesos[i];
            if (i < size - 1) file << ", ";
        }
        file << "]";
        if (layer < pesos.size() - 1) file << ",";
        file << "\n";
    }

    file << "  ],\n";

 
    file << "  \"bias\": [\n";

    for (size_t layer = 0; layer < bias.size(); layer++) {

        int output = capas[layer + 1];
        vector<float> h_bias(output);

        CUDA_CHECK(cudaMemcpy(h_bias.data(), bias[layer],
                              output * sizeof(float),
                              cudaMemcpyDeviceToHost));

        file << "    [";
        for (int j = 0; j < output; j++) {
            file << h_bias[j];
            if (j < output - 1) file << ", ";
        }
        file << "]";
        if (layer < bias.size() - 1) file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    cout << "Pesos guardados en JSON: " << nombre_archivo << endl;
}

};


