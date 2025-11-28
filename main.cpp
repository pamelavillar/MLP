#include <iostream>
//#include "mlp.cpp"
#include "mlp_threads.cpp"
//#include "mlp2.cpp"
#include <Eigen/Dense>

using namespace std;

//------------OBTENIENDO DATA ---------------------------------------/////

MatrixXd toOneHot(VectorXd& labels, int num_labels){
    MatrixXd one_hot {MatrixXd::Zero(labels.rows(), num_labels)};
    for (std::size_t i {}; i<labels.rows(); ++i){
        one_hot(i, static_cast<int>(labels(i))) = 1.0;
    }
    return one_hot;
}

pair<MatrixXd, VectorXd> load_all_batches() {

    vector<string> batch_files = {
        "data/data_batch_1.bin",
        "data/data_batch_2.bin",
        "data/data_batch_3.bin",
        "data/data_batch_4.bin",
        "data/data_batch_5.bin"
    };

    int total_images = 10000 * batch_files.size();

    MatrixXd pixeles_imagenes(total_images, 3072);
    VectorXd labels(total_images);

    int row = 0;

    for (const string &filename : batch_files) {

        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            cerr << "Error abriendo " << filename << endl;
            continue;
        }

        for (int i = 0; i < 10000; i++) {

            unsigned char label;
            unsigned char pixels[3072];

            file.read((char*)&label, 1);
            file.read((char*)pixels, 3072);

            for (int p = 0; p < 3072; ++p) {
                pixeles_imagenes(row, p) = pixels[p] / 255.0;
            }

            labels(row) = label;

            row++;
        }

        file.close();
    }

    return {pixeles_imagenes, labels};
}



pair<MatrixXd, VectorXd> load_test_batch() {
    ifstream file("data/test_batch.bin", ios::binary);
    if (!file.is_open()) {
        throw runtime_error("No se pudo abrir test_batch.bin");
    }

    const int num_images = 10000;
    const int num_pixels = 3072;

    MatrixXd pixeles_imagenes(num_images, num_pixels);
    VectorXd labels(num_images);

    for (int i = 0; i < num_images; i++) {
        unsigned char label;
        vector<unsigned char> pixels(num_pixels);

        file.read((char *)&label, 1);
        file.read((char *)pixels.data(), num_pixels);

        labels(i) = label;

       for (int j = 0; j < num_pixels; j++) {
            pixeles_imagenes(i, j) = double(pixels[j]) / 255.0;
        }
    }

    file.close();
    return {pixeles_imagenes, labels};
}

/////// ------------------MAIN------------------------------------/////

int main(){
    auto data_train = load_all_batches();
    auto data_test = load_test_batch();

    MatrixXd train_images = data_train.first;
    MatrixXd test_images  = data_test.first;
    VectorXd train_labels = data_train.second;
    VectorXd test_labels = data_test.second.cast<double>();

    MatrixXd train_labels_oh = toOneHot(train_labels, 10);
    MatrixXd test_labels_oh = toOneHot(test_labels, 10);

    MLP mlp({3072,512,256,128,64,10});

    mlp.learning(train_images,train_labels_oh, 50, 0.001, 128, true);

    MatrixXd pred = mlp.predecir(test_images);


    for (int i = 0; i < test_labels_oh.rows(); ++i){
        Eigen::Index actual_idx, pred_idx;
        test_labels_oh.row(i).maxCoeff(&actual_idx);
        pred.row(i).maxCoeff(&pred_idx);
        
        cout << "Actual: " << actual_idx << " Predecido: " << pred_idx 
            << " Probabilidades: " << pred.row(i) << endl;
    }
    cout<<"Accuracy: "<<mlp.exactitud(test_labels_oh, pred);

    return 0;
}   