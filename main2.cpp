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
  vector<float> pesos_anteriores;

public:
  vector<float> v_pesos;

  Perceptron(int _label, int _size, int _w, float _n,
             vector<ImageData> _dataset) {
    label = _label;
    sizeX = _size;
    dataset = _dataset;
    n = _n;

    v_pesos.resize(sizeX + 1);
    pesos_anteriores.resize(sizeX + 1);

    for (int i = 0; i < sizeX + 1; i++) {
      v_pesos[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.01f;
      pesos_anteriores[i] = v_pesos[i];
    }
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
  }

  float calcular_cambio_pesos() {
    float max_cambio = 0.0f;
    for (int i = 0; i < v_pesos.size(); i++) {
      float cambio = fabs(v_pesos[i] - pesos_anteriores[i]);
      if (cambio > max_cambio) {
        max_cambio = cambio;
      }
    }
    return max_cambio;
  }

  // <- NUEVA FUNCIÓN: Guardar pesos actuales para comparar después
  void guardar_pesos_anteriores() {
    for (int i = 0; i < v_pesos.size(); i++) {
      pesos_anteriores[i] = v_pesos[i];
    }
  }

  void entrenamiento(int max_epocas = 100) {
    int epoca_actual = max_epocas;
    const float UMBRAL_CAMBIO = n * 1000.0f;

    while (epoca_actual > 0) {
      cout << "Epoca: " << (max_epocas - epoca_actual + 1) << "\n";

      guardar_pesos_anteriores();
      bool hubo_cambio = false;

      for (int i = 0; i < dataset.size(); i++) {
        int d = (dataset[i].label == label) ? 1 : -1;

        float sum = bias * v_pesos[0];
        for (int j = 0; j < dataset[i].pixels.size(); j++) {
          float x = normalizacion(dataset[i].pixels[j]);
          sum += v_pesos[j + 1] * x;
        }

        int y = fn(sum);

        if (y != d) {
          updatePesos(d, y, i);
          hubo_cambio = true;
        }
      }

      float cambio_maximo = calcular_cambio_pesos();
      cout << "  Cambio maximo en pesos: " << cambio_maximo << endl;

      if (cambio_maximo > UMBRAL_CAMBIO) {
        cout << " Pesos cambiando demasiado...!" << endl;
      }

      if (!hubo_cambio) {
        cout << "Convergencia alcanzada. Entrenamiento detenido.\n";
        break;
      }

      if (cambio_maximo > 10.0f) {
        cout << "Pesos desbordados. Entrenamiento "
                "detenido.\n";
        break;
      }

      epoca_actual--;
    }
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
};

int main() {
  cout << "\n=== ENTRENAMIENTO CON THREADS Y CONTROL DE PESOS ===" << endl;
  vector<ImageData> train_data = load_all_batches();
  vector<ImageData> test_data = load_test_batch();
  cout << "Datos de entrenamiento: " << train_data.size() << endl;
  cout << "Datos de prueba: " << test_data.size() << endl;

  vector<Perceptron> perceptrones;
  for (int label = 0; label < CANT_PERCEPTRONS; label++) {
    perceptrones.push_back(
        Perceptron(label, train_data[0].pixels.size(), 0, 0.0001, train_data));
  }

  vector<thread> threads;

  for (int i = 0; i < CANT_PERCEPTRONS; i++) {
    threads.emplace_back([&perceptrones, i]() {
      cout << "Iniciando entrenamiento del perceptrón " << i << " en thread "
           << this_thread::get_id() << endl;
      perceptrones[i].entrenamiento(20);
      cout << "Perceptrón " << i << " completado" << endl;
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  cout << "Todos los threads han completado el entrenamiento" << endl;

  cout << "\n--- PRUEBAS ---" << endl;
  int aciertos = 0;
  int total = test_data.size();

  for (int i = 0; i < total; i++) {
    ImageData &imagen = test_data[i];
    int label_real = imagen.label;

    int pred_label = 0;
    float max_confianza = -1e9;

    for (int j = 0; j < CANT_PERCEPTRONS; j++) {

      float sum = perceptrones[j].v_pesos[0]; // bias
      for (int k = 0; k < imagen.pixels.size(); k++) {
        float x = (imagen.pixels[k] / 255.0f) * 2.0f - 1.0f;
        sum += perceptrones[j].v_pesos[k + 1] * x;
      }

      if (sum > max_confianza) {
        max_confianza = sum;
        pred_label = j;
      }
    }

    if (pred_label == label_real)
      aciertos++;

    if (i < 20) {
      cout << "Imagen " << i << " - Real: " << (int)label_real
           << " Predicho: " << pred_label << endl;
    }
  }

  float precision = (aciertos * 100.0f) / total;
  cout << "\n=== RESULTADOS ===" << endl;
  cout << "Aciertos: " << aciertos << "/" << total << " (" << precision << "%)"
       << endl;

  cout << "\n=== COMPLETADO ===" << endl;
  return 0;
}