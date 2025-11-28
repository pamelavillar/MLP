#include "mlp_cuda.cu"
#include <fstream>
#include <vector>

using namespace std;

void toOneHot(const vector<int>& labels, float* one_hot, int num_samples, int num_classes) {
    for (int i = 0; i < num_samples * num_classes; i++) {
        one_hot[i] = 0.0f;
    }
    for (int i = 0; i < num_samples; i++) {
        one_hot[i * num_classes + labels[i]] = 1.0f;
    }
}

pair<vector<float>, vector<int>> load_cifar_batch(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir " << filename << endl;
        exit(1);
    }

    const int num_images = 10000;
    const int num_pixels = 3072;

    vector<float> images(num_images * num_pixels);
    vector<int> labels(num_images);

    for (int i = 0; i < num_images; i++) {
        unsigned char label;
        unsigned char pixels[num_pixels];

        file.read((char*)&label, 1);
        file.read((char*)pixels, num_pixels);

        labels[i] = label;
        for (int j = 0; j < num_pixels; j++) {
            images[i * num_pixels + j] = pixels[j] / 255.0f;
        }
    }
    file.close();
    return {images, labels};
}

int main() {
    cout << "Cargando CIFAR-10..." << endl;

    vector<float> all_train_images;
    vector<int> all_train_labels;

    vector<string> batch_files = {
        "cifar-10-batches-bin/data_batch_1.bin",
        "cifar-10-batches-bin/data_batch_2.bin",
        "cifar-10-batches-bin/data_batch_3.bin",
        "cifar-10-batches-bin/data_batch_4.bin",
        "cifar-10-batches-bin/data_batch_5.bin"
    };

    for (const string& file : batch_files) {
        auto [images, labels] = load_cifar_batch(file);
        all_train_images.insert(all_train_images.end(), images.begin(), images.end());
        all_train_labels.insert(all_train_labels.end(), labels.begin(), labels.end());
    }

    auto [test_images, test_labels_vec] = load_cifar_batch("cifar-10-batches-bin/test_batch.bin");

    int num_train = all_train_labels.size();
    int num_test = test_labels_vec.size();
    int num_classes = 10;
    int input_size = 3072;

    cout << "Datos cargados: " << num_train << " muestras de entrenamiento, "
         << num_test << " muestras de test" << endl;

    float* train_labels_oh = new float[num_train * num_classes];
    float* test_labels_oh = new float[num_test * num_classes];

    toOneHot(all_train_labels, train_labels_oh, num_train, num_classes);
    toOneHot(test_labels_vec, test_labels_oh, num_test, num_classes);

    cout << "\nEntrenando..." << endl;
    MLP mlp({3072, 512, 256, 128, 10});

    cout << "\nIniciando entrenamiento..." << endl;
    mlp.learning(all_train_images.data(), train_labels_oh,
                 num_train, input_size, num_classes,
                 100,
                 0.1f,
                 128);

    float* predictions = mlp.predecir(test_images.data(), num_test, input_size);
    float acc = mlp.exactitud(predictions, test_labels_oh, num_test, num_classes);

    cout << "\nAccuracy final: " << acc * 100 << "%" << endl;

    cout << "\nEjemplos de predicciones:" << endl;
    for (int i = 0; i < 10; i++) {
        int pred_class = 0;
        int true_class = 0;
        float max_pred = predictions[i * num_classes];

        for (int j = 1; j < num_classes; j++) {
            if (predictions[i * num_classes + j] > max_pred) {
                max_pred = predictions[i * num_classes + j];
                pred_class = j;
            }
            if (test_labels_oh[i * num_classes + j] > 0.5f) {
                true_class = j;
            }
        }

        cout << "Muestra " << i << " - Real: " << true_class
             << ", Predicho: " << pred_class
             << ", Confianza: " << max_pred * 100 << "%" << endl;
    }

    mlp.guardarPesos("pesos_entrenados.json");


    delete[] train_labels_oh;
    delete[] test_labels_oh;
    delete[] predictions;

    return 0;
}
