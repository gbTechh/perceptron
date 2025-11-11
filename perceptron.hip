#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const int CANT_PERCEPTRONS = 10;

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      std::cerr << "HIP error: " << hipGetErrorString(error) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

struct ImageData {
  unsigned char label;          // 0-9
  vector<unsigned char> pixels; // 3072 valores 0-255
};

// ============ KERNELS HIP VERIFICADOS ============
__global__ void kernel_forward_simple(const float *pesos,
                                      const unsigned char *pixels,
                                      float *resultado, int num_pixels,
                                      float bias) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float sum = pesos[0] * bias;
    for (int i = 0; i < num_pixels; i++) {
      float pixel_val = pixels[i];
      float x = (pixel_val / 255.0f) * 2.0f - 1.0f;
      sum += pesos[i + 1] * x;
    }
    resultado[0] = sum;
  }
}

__global__ void kernel_update_weights_simple(float *pesos,
                                             const unsigned char *pixels,
                                             float delta, int num_pixels,
                                             float bias) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    pesos[0] += delta * bias;
    for (int i = 0; i < num_pixels; i++) {
      float pixel_val = pixels[i];
      float x = (pixel_val / 255.0f) * 2.0f - 1.0f;
      pesos[i + 1] += delta * x;
    }
  }
}
__global__ void kernel_forward_debug(float *resultado) {
  resultado[0] = 1.0f; // Solo devuelve 1.0 para prueba
}

__global__ void kernel_update_debug(float *pesos) {
  pesos[0] += 0.1f; // Solo incrementa un poco el bias
}
// ============ FUNCIONES DE CARGA ============
vector<ImageData> load_all_batches() {
  vector<ImageData> all_data;
  vector<string> batch_files = {"data_batch_1.bin", "data_batch_2.bin",
                                "data_batch_3.bin", "data_batch_4.bin",
                                "data_batch_5.bin"};

  for (const string &filename : batch_files) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
      cerr << "Error: No se pudo abrir " << filename << endl;
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
  return all_data;
}

vector<ImageData> load_test_batch() {
  vector<ImageData> test_data;
  ifstream file("test_batch.bin", ios::binary);

  if (!file.is_open()) {
    cerr << "Error: No se pudo abrir test_batch.bin" << endl;
    exit(1);
  }

  for (int i = 0; i < 10000; i++) {
    ImageData img;
    file.read((char *)&img.label, 1);
    img.pixels.resize(3072);
    file.read((char *)img.pixels.data(), 3072);
    test_data.push_back(img);
  }
  file.close();
  return test_data;
}

// ============ CLASE PERCEPTRON CON VERIFICACIONES ============
class Perceptron {
private:
  int label;
  int bias = 1;
  float n;
  int sizeX;
  vector<ImageData> dataset;

  float *d_pesos;
  unsigned char *d_pixels_temp;
  float *d_resultado;

public:
  vector<float> v_pesos;

  Perceptron(int _label, int _size, int _w, float _n,
             vector<ImageData> _dataset) {
    label = _label;
    sizeX = _size;
    dataset = _dataset;
    n = _n;

    cout << "Inicializando Perceptron " << label << " con sizeX: " << sizeX
         << endl;

    // Verificar datos
    if (dataset.empty()) {
      cerr << "Error: Dataset vacío para perceptron " << label << endl;
      exit(1);
    }

    v_pesos.resize(sizeX + 1, _w);

    // Inicializar memoria GPU con verificaciones
    hipError_t err;

    err = hipMalloc(&d_pesos, (sizeX + 1) * sizeof(float));
    if (err != hipSuccess) {
      cerr << "Error hipMalloc d_pesos: " << hipGetErrorString(err) << endl;
      exit(1);
    }

    err = hipMalloc(&d_pixels_temp, sizeX * sizeof(unsigned char));
    if (err != hipSuccess) {
      cerr << "Error hipMalloc d_pixels_temp: " << hipGetErrorString(err)
           << endl;
      hipFree(d_pesos);
      exit(1);
    }

    err = hipMalloc(&d_resultado, sizeof(float));
    if (err != hipSuccess) {
      cerr << "Error hipMalloc d_resultado: " << hipGetErrorString(err) << endl;
      hipFree(d_pesos);
      hipFree(d_pixels_temp);
      exit(1);
    }

    // Copiar pesos iniciales
    HIP_CHECK(hipMemcpy(d_pesos, v_pesos.data(), (sizeX + 1) * sizeof(float),
                        hipMemcpyHostToDevice));

    cout << "Perceptron " << label << " inicializado correctamente" << endl;
  }

  ~Perceptron() {
    if (d_pesos)
      hipFree(d_pesos);
    if (d_pixels_temp)
      hipFree(d_pixels_temp);
    if (d_resultado)
      hipFree(d_resultado);
  }

  int fn(float value) { return (value > 0) ? 1 : -1; }

  void updatePesos(int d, int y, int row) {
    float delta = n * (d - y);

    // Verificar índices
    if (row < 0 || row >= dataset.size()) {
      cerr << "Error: Índice de fila inválido: " << row << endl;
      return;
    }

    // Copiar pixels a GPU
    HIP_CHECK(hipMemcpy(d_pixels_temp, dataset[row].pixels.data(),
                        sizeX * sizeof(unsigned char), hipMemcpyHostToDevice));

    // Lanzar kernel con sintaxis tradicional pero verificada
    cout << "Lanzando kernel update para perceptron " << label
         << ", delta: " << delta << ", sizeX: " << sizeX << endl;

    // VERIFICAR punteros antes de lanzar kernel
    if (d_pesos == nullptr || d_pixels_temp == nullptr) {
      cerr << "Error: Punteros GPU no inicializados" << endl;
      return;
    }

    kernel_update_debug<<<1, 1>>>(d_pesos);

    hipError_t kernelErr = hipGetLastError();
    if (kernelErr != hipSuccess) {
      cerr << "Error lanzando kernel: " << hipGetErrorString(kernelErr) << endl;
      return;
    }

    HIP_CHECK(hipDeviceSynchronize());

    // Copiar pesos actualizados
    HIP_CHECK(hipMemcpy(v_pesos.data(), d_pesos, (sizeX + 1) * sizeof(float),
                        hipMemcpyDeviceToHost));
  }

  void entrenamiento(int epoca = 100) {
    cout << "Entrenando perceptron " << label << " por " << epoca << " épocas"
         << endl;

    while (epoca > 0) {
      cout << "Epoca: " << epoca << endl;
      for (int i = 0; i < min(10, (int)dataset.size());
           i++) { // SOLO 10 imágenes
        int d = (dataset[i].label == label) ? 1 : -1;

        // Forward pass
        float sum_inicial = 0.0f;
        HIP_CHECK(hipMemcpy(d_resultado, &sum_inicial, sizeof(float),
                            hipMemcpyHostToDevice));

        HIP_CHECK(hipMemcpy(d_pixels_temp, dataset[i].pixels.data(),
                            sizeX * sizeof(unsigned char),
                            hipMemcpyHostToDevice));

        // Lanzar kernel forward
        kernel_forward_debug<<<1, 1>>>(d_resultado);

        hipError_t kernelErr = hipGetLastError();
        if (kernelErr != hipSuccess) {
          cerr << "Error kernel forward: " << hipGetErrorString(kernelErr)
               << endl;
          continue;
        }

        HIP_CHECK(hipDeviceSynchronize());

        float sum;
        HIP_CHECK(
            hipMemcpy(&sum, d_resultado, sizeof(float), hipMemcpyDeviceToHost));

        int y = fn(sum);
        if (y != d) {
          updatePesos(d, y, i);
        }
      }
      epoca--;
    }
  }

  int predecir(const ImageData &imagen) {
    float sum_inicial = 0.0f;
    HIP_CHECK(hipMemcpy(d_resultado, &sum_inicial, sizeof(float),
                        hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(d_pixels_temp, imagen.pixels.data(),
                        sizeX * sizeof(unsigned char), hipMemcpyHostToDevice));

    // Lanzar kernel prediction
    kernel_forward_debug<<<1, 1>>>(d_resultado);

    hipError_t kernelErr = hipGetLastError();
    if (kernelErr != hipSuccess) {
      cerr << "Error kernel predicción: " << hipGetErrorString(kernelErr)
           << endl;
      return -1;
    }

    HIP_CHECK(hipDeviceSynchronize());

    float sum;
    HIP_CHECK(
        hipMemcpy(&sum, d_resultado, sizeof(float), hipMemcpyDeviceToHost));

    return fn(sum);
  }
};

// ============ MAIN CON VERIFICACIONES ============
int main() {
  cout << "=== INICIANDO PERCEPTRON CIFAR-10 CON HIP ===" << endl;

  // Verificar HIP
  int deviceCount;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  cout << "Dispositivos HIP disponibles: " << deviceCount << endl;

  if (deviceCount == 0) {
    cerr << "ERROR: No se encontraron dispositivos HIP!" << endl;
    return 1;
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipInit(0));

  cout << "Cargando datos CIFAR-10..." << endl;
  vector<ImageData> train_data = load_all_batches();
  vector<ImageData> test_data = load_test_batch();

  cout << "Datos de entrenamiento: " << train_data.size() << " imágenes"
       << endl;
  cout << "Datos de prueba: " << test_data.size() << " imágenes" << endl;

  if (train_data.empty() || test_data.empty()) {
    cerr << "ERROR: No se pudieron cargar los datos!" << endl;
    return 1;
  }

  vector<Perceptron> perceptrones;
  cout << "Creando " << CANT_PERCEPTRONS << " perceptrones..." << endl;

  // Solo crear 1 perceptron para prueba
  perceptrones.push_back(
      Perceptron(0, train_data[0].pixels.size(), 0, 0.01, train_data));

  cout << "Entrenando (solo 1 perceptron para prueba)..." << endl;
  for (auto &p : perceptrones) {
    p.entrenamiento(3); // Solo 3 épocas para prueba
  }

  cout << "\n--- PRUEBAS ---" << endl;
  for (int i = 0; i < min(5, (int)test_data.size()); i++) {
    ImageData &imagen = test_data[i];
    int label_real = imagen.label;

    cout << "Imagen " << i << " - Label real: " << label_real << " (";

    for (int j = 0; j < perceptrones.size(); j++) {
      int pred = perceptrones[j].predecir(imagen);
      if (pred == 1) {
        cout << j << " ";
      }
    }
    cout << ")" << endl;
  }

  cout << "\n=== EJECUCIÓN COMPLETADA ===" << endl;
  return 0;
}