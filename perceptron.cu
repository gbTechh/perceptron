
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const int CANT_PERCEPTRONS = 10;
const int PIXELS_PER_IMAGE = 3072;

struct ImageData {
  unsigned char label;
  unsigned char pixels[PIXELS_PER_IMAGE];
};

__global__ void actualizarPesosKernel(float *pesos, unsigned char *pixels,
                                      int *labels, float learning_rate,
                                      int num_images, int pixels_per_image) {
  int img_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (img_idx < num_images) {
    int d = labels[img_idx];

    float sum = pesos[0];
    for (int i = 0; i < pixels_per_image; i++) {
      float pixel = pixels[img_idx * pixels_per_image + i];
      float x = (pixel / 255.0f) * 2.0f - 1.0f;
      sum += pesos[i + 1] * x;
    }

    int y = (sum > 0) ? 1 : -1;

    if (y != d) {
      pesos[0] += learning_rate * (d - y) * 1.0f;

      for (int i = 0; i < pixels_per_image; i++) {
        float pixel = pixels[img_idx * pixels_per_image + i];
        float x = (pixel / 255.0f) * 2.0f - 1.0f;
        pesos[i + 1] += learning_rate * (d - y) * x;
      }
    }
  }
}

class PerceptronCUDA {
private:
  int label;
  float learning_rate;
  int pixels_per_image;

  float *d_pesos;
  unsigned char *d_pixels;
  int *d_labels;

  vector<float> h_pesos;
  vector<unsigned char> h_pixels;
  vector<int> h_labels;
  vector<int> indices;

public:
  PerceptronCUDA(int _label, float _learning_rate, int _pixels_per_image) {
    label = _label;
    learning_rate = _learning_rate;
    pixels_per_image = _pixels_per_image;

    h_pesos.resize(pixels_per_image + 1);
    for (int i = 0; i < pixels_per_image + 1; i++) {
      h_pesos[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.01f;
    }

    cudaMalloc(&d_pesos, (pixels_per_image + 1) * sizeof(float));
    cudaMemcpy(d_pesos, h_pesos.data(), (pixels_per_image + 1) * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  void cargarDatos(const vector<ImageData> &dataset) {
    h_pixels.resize(dataset.size() * pixels_per_image);
    h_labels.resize(dataset.size());
    indices.resize(dataset.size());

    for (int i = 0; i < dataset.size(); i++) {
      h_labels[i] = (dataset[i].label == label) ? 1 : -1;
      indices[i] = i;
      for (int j = 0; j < pixels_per_image; j++) {
        h_pixels[i * pixels_per_image + j] = dataset[i].pixels[j];
      }
    }

    cudaMalloc(&d_pixels,
               dataset.size() * pixels_per_image * sizeof(unsigned char));
    cudaMalloc(&d_labels, dataset.size() * sizeof(int));
  }

  void entrenamiento(int max_epocas = 20) {
    cout << "Entrenando perceptron " << label << " en GPU..." << endl;

    int num_images = h_labels.size();

    for (int epoca = 0; epoca < max_epocas; epoca++) {
      random_shuffle(indices.begin(), indices.end());

      vector<unsigned char> pixels_shuffled(num_images * pixels_per_image);
      vector<int> labels_shuffled(num_images);

      for (int i = 0; i < num_images; i++) {
        int orig_idx = indices[i];
        labels_shuffled[i] = h_labels[orig_idx];
        for (int j = 0; j < pixels_per_image; j++) {
          pixels_shuffled[i * pixels_per_image + j] =
              h_pixels[orig_idx * pixels_per_image + j];
        }
      }

      cudaMemcpy(d_pixels, pixels_shuffled.data(),
                 num_images * pixels_per_image * sizeof(unsigned char),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_labels, labels_shuffled.data(), num_images * sizeof(int),
                 cudaMemcpyHostToDevice);

      int blockSize = 256;
      int numBlocks = (num_images + blockSize - 1) / blockSize;

      actualizarPesosKernel<<<numBlocks, blockSize>>>(
          d_pesos, d_pixels, d_labels, learning_rate, num_images,
          pixels_per_image);
      cudaDeviceSynchronize();

      if (epoca % 5 == 0) {
        cudaMemcpy(h_pesos.data(), d_pesos,
                   (pixels_per_image + 1) * sizeof(float),
                   cudaMemcpyDeviceToHost);

        int aciertos = 0;
        for (int i = 0; i < num_images; i++) {
          float sum = h_pesos[0];
          for (int j = 0; j < pixels_per_image; j++) {
            float x =
                (h_pixels[i * pixels_per_image + j] / 255.0f) * 2.0f - 1.0f;
            sum += h_pesos[j + 1] * x;
          }
          int pred = (sum > 0) ? 1 : -1;
          if (pred == h_labels[i])
            aciertos++;
        }

        float precision = 100.0f * aciertos / num_images;
        cout << "  Epoca " << epoca << ": " << precision << "% correcto"
             << endl;
      }
    }

    cudaMemcpy(h_pesos.data(), d_pesos, (pixels_per_image + 1) * sizeof(float),
               cudaMemcpyDeviceToHost);
  }

  float predecir(const ImageData &imagen) {
    float sum = h_pesos[0];
    for (int i = 0; i < pixels_per_image; i++) {
      float x = (imagen.pixels[i] / 255.0f) * 2.0f - 1.0f;
      sum += h_pesos[i + 1] * x;
    }
    return sum;
  }

  ~PerceptronCUDA() {
    cudaFree(d_pesos);
    cudaFree(d_pixels);
    cudaFree(d_labels);
  }
};
vector<ImageData> load_all_batches() {
  vector<ImageData> all_data;
  vector<string> batch_files = {"./cifar-10-batches-bin/data_batch_1.bin",
                                "./cifar-10-batches-bin/data_batch_2.bin",
                                "./cifar-10-batches-bin/data_batch_3.bin",
                                "./cifar-10-batches-bin/data_batch_4.bin",
                                "./cifar-10-batches-bin/data_batch_5.bin"};

  for (const string &filename : batch_files) {
    ifstream file(filename, ios::binary);
    if (!file.is_open())
      continue;

    for (int i = 0; i < 10000; i++) {
      ImageData img;
      file.read((char *)&img.label, 1);
      file.read((char *)img.pixels, PIXELS_PER_IMAGE);
      all_data.push_back(img);
    }
    file.close();
  }
  return all_data;
}

vector<ImageData> load_test_batch() {
  vector<ImageData> test_data;
  ifstream file("./cifar-10-batches-bin/test_batch.bin", ios::binary);

  for (int i = 0; i < 10000; i++) {
    ImageData img;
    file.read((char *)&img.label, 1);
    file.read((char *)img.pixels, PIXELS_PER_IMAGE);
    test_data.push_back(img);
  }
  file.close();
  return test_data;
}

int main() {
  cout << "=== PERCEPTRON CON CUDA ===" << endl;

  auto start = chrono::high_resolution_clock::now();

  vector<ImageData> train_data = load_all_batches();
  vector<ImageData> test_data = load_test_batch();

  cout << "Datos entrenamiento: " << train_data.size() << endl;
  cout << "Datos prueba: " << test_data.size() << endl;

  vector<PerceptronCUDA> perceptrones;
  for (int i = 0; i < CANT_PERCEPTRONS; i++) {
    perceptrones.emplace_back(i, 0.0001f, PIXELS_PER_IMAGE);
    perceptrones[i].cargarDatos(train_data);
  }

  cout << "\n=== ENTRENAMIENTO ===" << endl;
  for (int i = 0; i < CANT_PERCEPTRONS; i++) {
    perceptrones[i].entrenamiento(20);
  }

  cout << "\n=== PRUEBA ===" << endl;
  int aciertos = 0;

  for (int i = 0; i < test_data.size(); i++) {
    int label_real = test_data[i].label;
    int pred_label = 0;
    float max_confianza = -1e9;

    for (int j = 0; j < CANT_PERCEPTRONS; j++) {
      float confianza = perceptrones[j].predecir(test_data[i]);
      if (confianza > max_confianza) {
        max_confianza = confianza;
        pred_label = j;
      }
    }

    if (pred_label == label_real)
      aciertos++;

    if (i < 10) {
      cout << "Imagen " << i << ": Real=" << label_real
           << ", Pred=" << pred_label
           << (pred_label == label_real ? " SI" : " NO") << endl;
    }
  }

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::seconds>(end - start);

  float precision = 100.0f * aciertos / test_data.size();
  cout << "\n=== RESULTADOS ===" << endl;
  cout << "Precision: " << aciertos << "/" << test_data.size() << " ("
       << precision << "%)" << endl;
  cout << "Tiempo total: " << duration.count() << " segundos" << endl;

  return 0;
}