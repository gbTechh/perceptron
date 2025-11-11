% %
    writefile perceptron_optimized.cu
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

    using namespace std;

const int CANT_PERCEPTRONS = 10;
const int BATCH_SIZE = 100; // Procesar por lotes

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
__global__ void train_perceptrons_batch(float *all_weights,
                                        const float *pixels_norm_batch,
                                        const int *labels_batch, int num_pixels,
                                        int batch_size, float n,
                                        int num_classes) {

  int perceptron_idx = blockIdx.x; // un bloque por perceptrón
  int pixel_idx = threadIdx.x;

  if (perceptron_idx >= num_classes)
    return;

  float *weights = all_weights + perceptron_idx * (num_pixels + 1);

  for (int img = 0; img < batch_size; img++) {
    int d = (labels_batch[img] == perceptron_idx) ? 1 : -1;

    // cálculo de suma parcial por hilo
    float local_sum = 0.0f;
    for (int j = pixel_idx; j < num_pixels; j += blockDim.x) {
      local_sum += weights[j + 1] * pixels_norm_batch[img * num_pixels + j];
    }

    // reducir entre hilos
    __shared__ float total_sum;
    if (threadIdx.x == 0)
      total_sum = weights[0]; // bias
    __syncthreads();
    atomicAdd(&total_sum, local_sum);
    __syncthreads();

    int y = (total_sum > 0.0f) ? 1 : -1;
    if (y != d) {
      // actualización en paralelo
      if (threadIdx.x == 0)
        weights[0] += n * (d - y);
      for (int j = pixel_idx; j < num_pixels; j += blockDim.x) {
        float x = pixels_norm_batch[img * num_pixels + j];
        weights[j + 1] += n * (d - y) * x;
      }
    }
    __syncthreads();
  }
}

void entrenar_todos_perceptrones_en_gpu(vector<ImageData> &dataset,
                                        vector<float> &all_weights, int sizeX,
                                        float n) {
  const int num_classes = CANT_PERCEPTRONS;
  const int batch_size = BATCH_SIZE;
  const int epochs = 10; // número de pasadas por los datos

  float *d_all_weights, *d_pixels;
  int *d_labels;
  CUDA_CHECK(
      cudaMalloc(&d_all_weights, num_classes * (sizeX + 1) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pixels, batch_size * sizeX * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labels, batch_size * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_all_weights, all_weights.data(),
                        num_classes * (sizeX + 1) * sizeof(float),
                        cudaMemcpyHostToDevice));

  vector<float> pixels_norm_batch(batch_size * sizeX);
  vector<int> labels_batch(batch_size);

  for (int epoch = 1; epoch <= epochs; epoch++) {
    cout << "Época GPU " << epoch << "..." << endl;

    for (int start = 0; start < dataset.size(); start += batch_size) {
      int current_batch = min(batch_size, (int)dataset.size() - start);

      // preparar batch
      for (int i = 0; i < current_batch; i++) {
        labels_batch[i] = dataset[start + i].label;
        for (int j = 0; j < sizeX; j++) {
          pixels_norm_batch[i * sizeX + j] =
              (dataset[start + i].pixels[j] / 255.0f) * 2.0f - 1.0f;
        }
      }

      CUDA_CHECK(cudaMemcpy(d_pixels, pixels_norm_batch.data(),
                            current_batch * sizeX * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_labels, labels_batch.data(),
                            current_batch * sizeof(int),
                            cudaMemcpyHostToDevice));

      int threads = 256;
      int blocks = num_classes;
      train_perceptrons_batch<<<blocks, threads>>>(
          d_all_weights, d_pixels, d_labels, sizeX, current_batch, n,
          num_classes);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  CUDA_CHECK(cudaMemcpy(all_weights.data(), d_all_weights,
                        num_classes * (sizeX + 1) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_all_weights);
  cudaFree(d_pixels);
  cudaFree(d_labels);
}

// ============ KERNELS OPTIMIZADOS ============
__global__ void perceptron_forward_batch(float *pesos, float *pixels_norm_batch,
                                         float *output_batch, int num_pixels,
                                         int batch_size) {
  int img_idx = blockIdx.x;
  int pix_idx = threadIdx.x;

  if (img_idx < batch_size && pix_idx < num_pixels) {
    int offset = img_idx * num_pixels + pix_idx;
    output_batch[offset] = pesos[pix_idx + 1] * pixels_norm_batch[offset];
  }
}

__global__ void compute_sums(float *output_batch, float *sums, int num_pixels,
                             int batch_size, float bias_weight) {
  int img_idx = blockIdx.x;

  if (img_idx < batch_size) {
    float sum = bias_weight; // bias
    for (int i = 0; i < num_pixels; i++) {
      int idx = img_idx * num_pixels + i;
      sum += output_batch[idx];
    }
    sums[img_idx] = sum;
  }
}

// ============ FUNCIONES DE CARGA (igual que antes) ============
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

  for (int i = 0; i < 1000; i++) {
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

// ============ CLASE PERCEPTRON OPTIMIZADA ============
class Perceptron {
public:
  int label;
  float n;
  int sizeX;
  vector<ImageData> dataset;
  vector<float> prev_pesos;

  // Memoria GPU para procesamiento por lotes
  float *d_pesos;
  float *d_pixels_norm_batch;
  float *d_output_batch;
  float *d_sums;

public:
  vector<float> v_pesos;

  Perceptron(int _label, int _size, float _n, vector<ImageData> _dataset) {
    label = _label;
    sizeX = _size;
    dataset = _dataset;
    n = _n;

    v_pesos.resize(sizeX + 1);
    prev_pesos.resize(sizeX + 1);

    unsigned int seed = time(0) + label;
    srand(seed);
    for (int i = 0; i < sizeX + 1; i++) {
      v_pesos[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.01f;
    }

    // Reservar memoria para lotes
    CUDA_CHECK(cudaMalloc(&d_pesos, (sizeX + 1) * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_pixels_norm_batch, BATCH_SIZE * sizeX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_batch, BATCH_SIZE * sizeX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, BATCH_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pesos, v_pesos.data(), (sizeX + 1) * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  ~Perceptron() {
    cudaFree(d_pesos);
    cudaFree(d_pixels_norm_batch);
    cudaFree(d_output_batch);
    cudaFree(d_sums);
  }

  int fn(float value) { return (value > 0) ? 1 : -1; }

  float normalizacion(unsigned char pixel) {
    return (pixel / 255.0f) * 2.0f - 1.0f;
  }

  float calcular_cambio_pesos() {
    float max_diff = 0.0f;
    for (int i = 0; i < v_pesos.size(); i++) {
      float diff = fabs(v_pesos[i] - prev_pesos[i]);
      max_diff = max(max_diff, diff);
    }
    return max_diff;
  }

  void guardar_pesos() { prev_pesos = v_pesos; }

  void updatePesos(int d, int y, int row) {
    for (int i = 0; i < v_pesos.size(); i++) {
      float x = 1;
      if (i > 0) {
        x = normalizacion(dataset[row].pixels[i - 1]);
      }
      v_pesos[i] = v_pesos[i] + n * (d - y) * x;
    }

    // Actualizar pesos en GPU solo cuando sea necesario
    CUDA_CHECK(cudaMemcpy(d_pesos, v_pesos.data(), (sizeX + 1) * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  // ENTRENAMIENTO OPTIMIZADO CON PROCESAMIENTO POR LOTES
  void entrenamiento_optimizado(int max_epocas = 50,
                                float convergence_threshold = 1e-5,
                                int patience = 3) {
    cout << "Entrenando perceptrón para clase " << label << " (OPTIMIZADO)"
         << endl;

    int epoca = 1;
    int epocas_sin_mejora = 0;
    int mejor_aciertos = 0;

    while (epoca <= max_epocas) {
      guardar_pesos();

      int errores_epoca = 0;
      int total_procesadas = 0;

      // Procesar por lotes
      for (int batch_start = 0; batch_start < dataset.size();
           batch_start += BATCH_SIZE) {
        int current_batch_size =
            min(BATCH_SIZE, (int)dataset.size() - batch_start);
        if (current_batch_size <= 0)
          break;

        // Preparar lote en CPU
        vector<float> pixels_norm_batch(current_batch_size * sizeX);
        vector<int> labels_batch(current_batch_size);

        for (int i = 0; i < current_batch_size; i++) {
          int img_idx = batch_start + i;
          labels_batch[i] = (dataset[img_idx].label == label) ? 1 : -1;

          for (int j = 0; j < sizeX; j++) {
            pixels_norm_batch[i * sizeX + j] =
                normalizacion(dataset[img_idx].pixels[j]);
          }
        }

        // Copiar lote completo a GPU
        CUDA_CHECK(cudaMemcpy(d_pixels_norm_batch, pixels_norm_batch.data(),
                              current_batch_size * sizeX * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Ejecutar kernels
        dim3 blockDim(sizeX);
        dim3 gridDim(current_batch_size);

        perceptron_forward_batch<<<gridDim, blockDim>>>(
            d_pesos, d_pixels_norm_batch, d_output_batch, sizeX,
            current_batch_size);
        CUDA_CHECK(cudaGetLastError());

        compute_sums<<<current_batch_size, 1>>>(d_output_batch, d_sums, sizeX,
                                                current_batch_size, v_pesos[0]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Obtener resultados
        vector<float> sums(current_batch_size);
        CUDA_CHECK(cudaMemcpy(sums.data(), d_sums,
                              current_batch_size * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Actualizar pesos
        for (int i = 0; i < current_batch_size; i++) {
          int y = fn(sums[i]);
          if (y != labels_batch[i]) {
            updatePesos(labels_batch[i], y, batch_start + i);
            errores_epoca++;
          }
        }

        total_procesadas += current_batch_size;
        if (total_procesadas >= 1000)
          break; // Limitar para prueba rápida
      }

      // Lógica de early stopping (igual que antes)
      int aciertos_epoca = total_procesadas - errores_epoca;
      float precision_epoca = (aciertos_epoca * 100.0f) / total_procesadas;
      float cambio_pesos = calcular_cambio_pesos();
      bool convergencia_perfecta = (errores_epoca == 0);

      if (epoca % 5 == 0 || epoca == 1) {
        cout << "  Época " << epoca << " | Errores: " << errores_epoca << " ("
             << precision_epoca << "% correcto)"
             << " | Cambio pesos: " << cambio_pesos << endl;
      }

      if (aciertos_epoca > mejor_aciertos) {
        mejor_aciertos = aciertos_epoca;
        epocas_sin_mejora = 0;
      } else {
        epocas_sin_mejora++;
      }

      if (convergencia_perfecta) {
        cout << "  ✓ Convergencia perfecta en época " << epoca << endl;
        break;
      }

      if (cambio_pesos < convergence_threshold) {
        cout << "  ✓ Convergencia por pesos estables en época " << epoca
             << endl;
        break;
      }

      if (epocas_sin_mejora >= patience) {
        cout << "  ⚠ Early stopping en época " << epoca << endl;
        break;
      }

      epoca++;
    }

    if (epoca > max_epocas) {
      cout << "  ⏱ Alcanzado límite de épocas (" << max_epocas << ")" << endl;
    }
    cout << endl;
  }

  // Predicción optimizada
  float confidence(const ImageData &imagen) {
    // Para predicción individual, mantener simple
    float sum = v_pesos[0]; // bias
    for (int j = 0; j < sizeX; j++) {
      float x = normalizacion(imagen.pixels[j]);
      sum += v_pesos[j + 1] * x;
    }
    return sum;
  }
};

// ============ MAIN ============
int main() {
  cout << "=== PERCEPTRON CUDA OPTIMIZADO ===" << endl;

  // Verificar disponibilidad de GPU
  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  cout << "Dispositivos CUDA: " << deviceCount << endl;

  if (deviceCount == 0) {
    cerr << "No hay GPU CUDA disponible" << endl;
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(0));

  // Cargar dataset
  vector<ImageData> train_data = load_all_batches();
  vector<ImageData> test_data = load_test_batch();

  if (train_data.empty()) {
    cerr << "No se encontraron datos de entrenamiento" << endl;
    return 1;
  }

  int num_pixels = train_data[0].pixels.size();
  cout << "\nCargadas " << train_data.size() << " imágenes de entrenamiento"
       << endl;
  cout << "Cargadas " << test_data.size() << " imágenes de test" << endl;

  // Inicializar pesos de todos los perceptrones
  vector<float> all_weights(CANT_PERCEPTRONS * (num_pixels + 1));
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

  for (int i = 0; i < all_weights.size(); i++) {
    all_weights[i] = dist(gen);
  }

  cout << "\nCreando " << CANT_PERCEPTRONS << " perceptrones en GPU..." << endl;

  // ENTRENAMIENTO EN PARALELO
  cout << "\n=== ENTRENAMIENTO OPTIMIZADO (GPU) ===" << endl;
  entrenar_todos_perceptrones_en_gpu(train_data, all_weights, num_pixels,
                                     0.001);

  // EVALUACIÓN EN CPU (usando los pesos entrenados)
  cout << "\n=== EVALUACIÓN ===" << endl;
  int aciertos = 0;
  int total = test_data.size();
  vector<vector<int>> confusion(10, vector<int>(10, 0));
  vector<string> class_names = {"avión", "auto", "pájaro",  "gato",  "ciervo",
                                "perro", "rana", "caballo", "barco", "camión"};

  for (int i = 0; i < total; i++) {
    ImageData &img = test_data[i];
    int label_real = img.label;

    float max_conf = -1e9;
    int pred_label = -1;

    // Calcular salida para los 10 perceptrones
    for (int p = 0; p < CANT_PERCEPTRONS; p++) {
      float sum = all_weights[p * (num_pixels + 1)]; // bias
      for (int j = 0; j < num_pixels; j++) {
        float x = (img.pixels[j] / 255.0f) * 2.0f - 1.0f;
        sum += all_weights[p * (num_pixels + 1) + j + 1] * x;
      }
      if (sum > max_conf) {
        max_conf = sum;
        pred_label = p;
      }
    }

    confusion[label_real][pred_label]++;
    if (pred_label == label_real)
      aciertos++;
  }

  float precision = (aciertos * 100.0f) / total;
  cout << "Precisión total: " << aciertos << "/" << total << " (" << precision
       << "%)" << endl;

  cout << "\n=== MATRIZ DE CONFUSIÓN ===" << endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      cout << confusion[i][j] << " ";
    }
    cout << endl;
  }

  cout << "\n=== COMPLETADO ===" << endl;
  return 0;
}
