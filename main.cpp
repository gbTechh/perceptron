#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

const int CANT_PERCEPTRONS = 10;

struct ImageData {
  unsigned char label;
  vector<unsigned char> pixels;
};

vector<ImageData> load_all_batches() {
  vector<ImageData> all_data;
  vector<string> batch_files = {"data_batch_1.bin", "data_batch_2.bin",
                                "data_batch_3.bin", "data_batch_4.bin",
                                "data_batch_5.bin"};

  for (const string &filename : batch_files) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
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

class Perceptron {
private:
  int label;
  int bias = 1;
  float n;
  int sizeX;
  vector<ImageData> dataset;
  vector<float> prev_pesos; // Para detectar convergencia

public:
  vector<float> v_pesos;

  Perceptron(int _label, int _size, int _w, float _n,
             vector<ImageData> _dataset) {
    label = _label;
    sizeX = _size;
    dataset = _dataset;
    n = _n;

    v_pesos.resize(sizeX + 1);
    prev_pesos.resize(sizeX + 1);

    // Inicialización aleatoria pequeña
    for (int i = 0; i < sizeX + 1; i++) {
      v_pesos[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.01f;
    }
  }

  int fn(float value) {
    if (value > 0) {
      return 1;
    } else if (value < 0) {
      return -1;
    }
    return 0;
  }

  float normalizacion(unsigned char pixel) {
    return (pixel / 255.0f) * 2.0f - 1.0f;
  }

  // Calcular diferencia entre pesos actuales y anteriores
  float calcular_cambio_pesos() {
    float max_diff = 0.0f;
    for (int i = 0; i < v_pesos.size(); i++) {
      float diff = fabs(v_pesos[i] - prev_pesos[i]);
      max_diff = max(max_diff, diff);
    }
    return max_diff;
  }

  // Guardar pesos actuales
  void guardar_pesos() { prev_pesos = v_pesos; }

  void updatePesos(int d, int y, int row) {
    for (int i = 0; i < v_pesos.size(); i++) {
      float x = 1;
      if (i > 0) {
        x = normalizacion(dataset[row].pixels[i - 1]);
      }
      v_pesos[i] = v_pesos[i] + n * (d - y) * x;
    }
  }

  void entrenamiento(int max_epocas = 100, float convergence_threshold = 1e-5,
                     int patience = 3) {
    cout << "Entrenando perceptrón para clase " << label << endl;

    int epoca = 1;
    int epocas_sin_mejora = 0;
    int mejor_aciertos = 0;

    while (epoca <= max_epocas) {
      guardar_pesos();

      int errores_epoca = 0;

      // Entrenar una época completa
      for (int i = 0; i < dataset.size(); i++) {
        int d = (dataset[i].label == label) ? 1 : -1;

        float sum = 0;
        sum += bias * v_pesos[0];
        for (int j = 0; j < dataset[i].pixels.size(); j++) {
          unsigned char pix = dataset[i].pixels[j];
          float x = normalizacion(pix);
          sum += v_pesos[j + 1] * x;
        }

        int y = fn(sum);
        if (y != d) {
          updatePesos(d, y, i);
          errores_epoca++;
        }
      }

      int aciertos_epoca = dataset.size() - errores_epoca;
      float precision_epoca = (aciertos_epoca * 100.0f) / dataset.size();

      // CONDICIÓN 1: Cambio en pesos es muy pequeño
      float cambio_pesos = calcular_cambio_pesos();

      // CONDICIÓN 2: No hay errores (convergencia perfecta)
      bool convergencia_perfecta = (errores_epoca == 0);

      // CONDICIÓN 3: No hay mejora significativa (early stopping por patience)
      if (aciertos_epoca > mejor_aciertos) {
        mejor_aciertos = aciertos_epoca;
        epocas_sin_mejora = 0;
      } else {
        epocas_sin_mejora++;
      }

      // Mostrar progreso cada 5 épocas o si es la primera
      if (epoca % 5 == 0 || epoca == 1) {
        cout << "  Época " << epoca << " | Errores: " << errores_epoca << " ("
             << precision_epoca << "% correcto)"
             << " | Cambio pesos: " << cambio_pesos << endl;
      }

      // DETENER SI:
      // 1. Convergencia perfecta (sin errores)
      if (convergencia_perfecta) {
        cout << "  ✓ Convergencia perfecta en época " << epoca << " (0 errores)"
             << endl;
        break;
      }

      // 2. Cambio en pesos es insignificante
      if (cambio_pesos < convergence_threshold) {
        cout << "  ✓ Convergencia por pesos estables en época " << epoca
             << " (cambio: " << cambio_pesos << ")" << endl;
        break;
      }

      // 3. Sin mejora durante 'patience' épocas
      if (epocas_sin_mejora >= patience) {
        cout << "  ⚠ Early stopping en época " << epoca << " (sin mejora por "
             << patience << " épocas)" << endl;
        break;
      }

      epoca++;
    }

    if (epoca > max_epocas) {
      cout << "  ⏱ Alcanzado límite de épocas (" << max_epocas << ")" << endl;
    }

    cout << endl;
  }

  int predecir(const ImageData &imagen) {
    float sum = 0;
    sum += bias * v_pesos[0];
    for (int j = 0; j < imagen.pixels.size(); j++) {
      float x = normalizacion(imagen.pixels[j]);
      sum += v_pesos[j + 1] * x;
    }
    return fn(sum);
  }

  float confidence(const ImageData &imagen) {
    float sum = 0;
    sum += bias * v_pesos[0];
    for (int j = 0; j < imagen.pixels.size(); j++) {
      float x = normalizacion(imagen.pixels[j]);
      sum += v_pesos[j + 1] * x;
    }
    return sum; // Retornar valor sin activación
  }
};

int main() {
  cout << "=== PERCEPTRÓN CON EARLY STOPPING (MULTITHREAD) ===" << endl;

  vector<ImageData> train_data = load_all_batches();
  vector<ImageData> test_data = load_test_batch();

  cout << "Datos de entrenamiento: " << train_data.size() << endl;
  cout << "Datos de prueba: " << test_data.size() << endl;

  vector<Perceptron> perceptrones;
  perceptrones.reserve(CANT_PERCEPTRONS);
  for (int label = 0; label < CANT_PERCEPTRONS; label++) {
    perceptrones.emplace_back(label, train_data[0].pixels.size(), 0, 0.01f,
                              train_data);
  }

  cout << "\n=== ENTRENAMIENTO EN PARALELO ===" << endl;

  vector<thread> threads;
  for (int i = 0; i < CANT_PERCEPTRONS; i++) {
    threads.emplace_back(
        [&perceptrones, i]() { perceptrones[i].entrenamiento(100, 1e-5, 5); });
  }

  // Esperar a que todos los hilos terminen
  for (auto &t : threads) {
    t.join();
  }

  cout << "\n=== EVALUACIÓN ===" << endl;
  int aciertos = 0;
  int total = test_data.size();
  vector<vector<int>> confusion(10, vector<int>(10, 0));

  for (int i = 0; i < total; i++) {
    ImageData &imagen = test_data[i];
    int label_real = imagen.label;

    // Buscar perceptrón con mayor confianza
    float max_confidence = -1e9;
    int pred_label = -1;

    for (int j = 0; j < CANT_PERCEPTRONS; j++) {
      float conf = perceptrones[j].confidence(imagen);
      if (conf > max_confidence) {
        max_confidence = conf;
        pred_label = j;
      }
    }

    confusion[label_real][pred_label]++;
    if (pred_label == label_real)
      aciertos++;

    if (i < 20) {
      cout << "Imagen " << i << " - Real: " << label_real
           << " | Predicho: " << pred_label << " | "
           << (pred_label == label_real ? "✓" : "✗") << endl;
    }
  }

  float precision = (aciertos * 100.0f) / total;
  cout << "\n=== RESULTADOS ===" << endl;
  cout << "Precisión total: " << aciertos << "/" << total << " (" << precision
       << "%)" << endl;

  // Precisión por clase
  cout << "\nPrecisión por clase:" << endl;
  vector<string> class_names = {"avión", "auto", "pájaro",  "gato",  "ciervo",
                                "perro", "rana", "caballo", "barco", "camión"};

  for (int i = 0; i < 10; i++) {
    int total_class = 0;
    for (int j = 0; j < 10; j++)
      total_class += confusion[i][j];

    if (total_class > 0) {
      float class_acc = (confusion[i][i] * 100.0f) / total_class;
      cout << "  Clase " << i << " (" << class_names[i]
           << "): " << confusion[i][i] << "/" << total_class << " ("
           << class_acc << "%)" << endl;
    }
  }

  cout << "\n=== COMPLETADO ===" << endl;
  return 0;
}