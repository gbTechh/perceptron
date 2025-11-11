% %
    writefile perceptron2.cu
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

    using namespace std;

const int CANT_PERCEPTRONS = 10;

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__  \
           << ":" << __LINE__ << endl;                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

struct ImageData {
  unsigned char label;
  vector<unsigned char> pixels;
};

// ============ KERNEL SIMPLE SIN SHARED MEMORY ============
__global__ void perceptron_forward_simple(float *pesos, float *pixels_norm,
                                          float *output, int num_pixels) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Solo calcular si estamos dentro del rango
  if (idx < num_pixels) {
    output[idx] = pesos[idx + 1] * pixels_norm[idx];
  }
}
__global__ void perceptron_forward_reduce(float *pesos, float *pixels_norm,
                                          float *result, int num_pixels) {
  extern __shared__ float cache[];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
  float temp = 0.0f;

  // Cada hilo multiplica un peso por su pixel
  if (idx < num_pixels) {
    temp = pesos[idx + 1] * pixels_norm[idx];
  }
  cache[cacheIndex] = temp;

  __syncthreads();

  // Reducción en memoria compartida
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (cacheIndex < stride) {
      cache[cacheIndex] += cache[cacheIndex + stride];
    }
    __syncthreads();
  }

  // El hilo 0 de cada bloque escribe su suma parcial
  if (cacheIndex == 0) {
    atomicAdd(result, cache[0]);
  }
}
// ============ FUNCIONES DE CARGA ============
vector<ImageData> load_all_batches() {
  vector<ImageData> all_data;
  vector<string> batch_files = {"./cifar-10-batches-bin/data_batch_1.bin",
                                "./cifar-10-batches-bin/data_batch_2.bin",
                                "./cifar-10-batches-bin/data_batch_3.bin",
                                "./cifar-10-batches-bin/data_batch_4.bin",
                                "./cifar-10-batches-bin/data_batch_5.bin"};

  for (const string &filename : batch_files) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
      cerr << "No se pudo abrir: " << filename << endl;
      continue;
    }

    for (int i = 0; i < 10000; i++) {
      ImageData img;
      file.read((char *)&img.label, 1);
      img.pixels.resize(3072);
      file.read((char *)img.pixels.data(), 3072);
      all_data.push_back(img);
    }
    file.close();
  }
  cout << "Cargadas " << all_data.size() << " imágenes de entrenamiento"
       << endl;
  return all_data;
}

vector<ImageData> load_test_batch() {
  vector<ImageData> test_data;
  ifstream file("./cifar-10-batches-bin/test_batch.bin", ios::binary);

  if (!file.is_open()) {
    cerr << "No se pudo abrir test_batch.bin" << endl;
    return test_data;
  }

  for (int i = 0; i < 10000; i++) {
    ImageData img;
    file.read((char *)&img.label, 1);
    img.pixels.resize(3072);
    file.read((char *)img.pixels.data(), 3072);
    test_data.push_back(img);
  }
  file.close();
  cout << "Cargadas " << test_data.size() << " imágenes de test" << endl;
  return test_data;
}

// ============ CLASE PERCEPTRON ============
class Perceptron {
public:
  int label;
  int bias = 1;
  float n;
  int sizeX;
  vector<ImageData> dataset;

  // Memoria GPU
  float *d_pesos;
  float *d_pixels_norm;
  float *d_output;

public:
  vector<float> v_pesos;
  Perceptron(const Perceptron &) = delete;
  Perceptron &operator=(const Perceptron &) = delete;
  Perceptron(Perceptron &&other) noexcept {
    label = other.label;
    bias = other.bias;
    n = other.n;
    sizeX = other.sizeX;
    dataset = std::move(other.dataset);
    v_pesos = std::move(other.v_pesos);

    d_pesos = other.d_pesos;
    d_pixels_norm = other.d_pixels_norm;
    d_output = other.d_output;

    // Evitar que el destructor libere la memoria dos veces
    other.d_pesos = nullptr;
    other.d_pixels_norm = nullptr;
    other.d_output = nullptr;
  }

  Perceptron &operator=(Perceptron &&other) noexcept {
    if (this != &other) {
      cudaFree(d_pesos);
      cudaFree(d_pixels_norm);
      cudaFree(d_output);

      label = other.label;
      bias = other.bias;
      n = other.n;
      sizeX = other.sizeX;
      dataset = std::move(other.dataset);
      v_pesos = std::move(other.v_pesos);

      d_pesos = other.d_pesos;
      d_pixels_norm = other.d_pixels_norm;
      d_output = other.d_output;

      other.d_pesos = nullptr;
      other.d_pixels_norm = nullptr;
      other.d_output = nullptr;
    }
    return *this;
  }

  Perceptron(int _label, int _size, int _w, float _n,
             vector<ImageData> _dataset) {
    label = _label;
    sizeX = _size;
    dataset = _dataset;
    n = _n;

    v_pesos.resize(sizeX + 1, _w);

    // Reservar memoria GPU
    CUDA_CHECK(cudaMalloc(&d_pesos, (sizeX + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pixels_norm, sizeX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeX * sizeof(float)));

    // Copiar pesos iniciales
    CUDA_CHECK(cudaMemcpy(d_pesos, v_pesos.data(), (sizeX + 1) * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  ~Perceptron() {
    cudaFree(d_pesos);
    cudaFree(d_pixels_norm);
    cudaFree(d_output);
  }

  int fn(float value) { return (value > 0) ? 1 : -1; }

  float normalizacion(unsigned char pixel) {
    return (pixel / 255.0f) * 2.0f - 1.0f;
  }

  void updatePesos(int d, int y, int row) {
    for (int i = 0; i < v_pesos.size(); i++) {
      float x = 1;
      if (i > 0) {
        x = normalizacion(dataset[row].pixels[i - 1]);
      }
      v_pesos[i] = v_pesos[i] + n * (d - y) * x;
    }

    CUDA_CHECK(cudaMemcpy(d_pesos, v_pesos.data(), (sizeX + 1) * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  void entrenamiento(int epoca = 100) {
    while (epoca > 0) {
      // cout << "Epoca: " << epoca << endl;
      for (int i = 0; i < dataset.size(); i++) { // Solo 5 imágenes
        int d = (dataset[i].label == label) ? 1 : -1;

        // Normalizar pixels
        vector<float> pixels_norm(sizeX);
        for (int j = 0; j < sizeX; j++) {
          pixels_norm[j] = normalizacion(dataset[i].pixels[j]);
        }
        CUDA_CHECK(cudaMemcpy(d_pixels_norm, pixels_norm.data(),
                              sizeX * sizeof(float), cudaMemcpyHostToDevice));

        // Configurar ejecución del kernel
        int threads = 1024;
        int blocks = (sizeX + threads - 1) / threads;
        if (blocks > 65535)
          blocks = 65535;

        // cout << "Launching kernel with " << blocks << " blocks, " << threads
        // << " threads per block, sizeX=" << sizeX << endl;

        perceptron_forward_simple<<<blocks, threads>>>(d_pesos, d_pixels_norm,
                                                       d_output, sizeX);

        // Verificar error del kernel
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
          cerr << "Error kernel: " << cudaGetErrorString(kernelErr) << endl;
          cerr << "Blocks: " << blocks << ", Threads: " << threads << endl;
          continue;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        // Recuperar y sumar en CPU
        vector<float> output(sizeX);
        CUDA_CHECK(cudaMemcpy(output.data(), d_output, sizeX * sizeof(float),
                              cudaMemcpyDeviceToHost));

        float sum = bias * v_pesos[0];
        for (int j = 0; j < sizeX; j++) {
          sum += output[j];
        }

        int y = fn(sum);
        if (y != d) {
          updatePesos(d, y, i);
        }
      }
      epoca--;
    }
  }

  int predecir(const ImageData &imagen) {
    vector<float> pixels_norm(sizeX);
    for (int j = 0; j < sizeX; j++) {
      pixels_norm[j] = normalizacion(imagen.pixels[j]);
    }

    CUDA_CHECK(cudaMemcpy(d_pixels_norm, pixels_norm.data(),
                          sizeX * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (sizeX + threads - 1) / threads;
    if (blocks > 65535)
      blocks = 65535;

    perceptron_forward_simple<<<blocks, threads>>>(d_pesos, d_pixels_norm,
                                                   d_output, sizeX);

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
      cerr << "Error kernel predicción: " << cudaGetErrorString(kernelErr)
           << endl;
      return -1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> output(sizeX);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, sizeX * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float sum = bias * v_pesos[0];
    for (int j = 0; j < sizeX; j++) {
      sum += output[j];
    }

    return fn(sum);
  }
};

// ============ MAIN ============
int main() {
  cout << "=== PERCEPTRON CUDA ===" << endl;

  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  cout << "Dispositivos CUDA: " << deviceCount << endl;

  if (deviceCount == 0) {
    cerr << "No hay GPU CUDA" << endl;
    return 1;
  }

  CUDA_CHECK(cudaSetDevice(0));

  vector<ImageData> train_data = load_all_batches();
  vector<ImageData> test_data = load_test_batch();

  if (train_data.empty()) {
    cerr << "No hay datos de entrenamiento" << endl;
    return 1;
  }

  cout << "\nCreando " << CANT_PERCEPTRONS << " perceptrones..." << endl;
  vector<Perceptron> perceptrones;
  for (int label = 0; label < CANT_PERCEPTRONS; label++) {

    perceptrones.emplace_back(label, train_data[0].pixels.size(), 0, 0.01,
                              train_data);
  }

  cout << "\n=== ENTRENAMIENTO ===" << endl;
  for (auto &p : perceptrones) {
    p.entrenamiento(50); // Solo 2 épocas
  }

  cout << "\n--- PRUEBAS ---" << endl;
  int aciertos = 0;
  int total = test_data.size();

  for (int i = 0; i < total; i++) {
    ImageData &imagen = test_data[i];
    int label_real = imagen.label;

    float max_sum = -1e9;
    int pred_label = -1;

    for (int j = 0; j < CANT_PERCEPTRONS; j++) {
      int pred = perceptrones[j].predecir(imagen);
      if (pred == 1) {
        pred_label = j;
        break;
      }
    }

    if (pred_label == label_real)
      aciertos++;

    cout << "Imagen " << i << " - Real: " << (int)label_real
         << " Predicho: " << pred_label << endl;
  }

  float precision = (aciertos * 100.0f) / total;
  cout << "\n=== RESULTADOS ===" << endl;
  cout << "Aciertos: " << aciertos << "/" << total << " (" << precision << "%)"
       << endl;

  cout << "\n=== COMPLETADO ===" << endl;
  return 0;
}
