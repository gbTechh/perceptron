#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const int CANT_PERCEPTRONS = 1;

struct ImageData {
  unsigned char label;          // 0-9
  vector<unsigned char> pixels; // 3072 valores 0-255
};

vector<ImageData> load_all_batches() {
  vector<ImageData> all_data;
  vector<string> batch_files = {"data_batch_1.bin", "data_batch_2.bin",
                                "data_batch_3.bin", "data_batch_4.bin",
                                "data_batch_5.bin"};

  for (const string &filename : batch_files) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
      continue; // O manejar error
    }

    for (int i = 0; i < 50000; i++) {
      ImageData img;

      // Leer label (1 byte)
      file.read((char *)&img.label, 1);

      // Leer pixels (3072 bytes)
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

public:
  vector<float> v_pesos;
  Perceptron(int _label, int _size, int _w, float _n,
             vector<ImageData> _dataset) {
    label = _label;
    sizeX = _size;
    dataset = _dataset;
    n = _n;

    v_pesos.resize(sizeX + 1);

    for (int i = 0; i < sizeX; i++) {
      v_pesos[i] = _w;
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

  void entrenamiento(int epoca = 100) {
    while (epoca > 0) {
      cout << "Epoca: " << epoca << "\n";
      bool hubo_cambio = false; // ← nuevo

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
          hubo_cambio = true; // ← si hubo ajuste
        }
      }

      // Si en toda la época no hubo ningún cambio → convergió
      if (!hubo_cambio) {
        cout << "Convergencia alcanzada. Entrenamiento detenido.\n";
        break;
      }

      epoca--;
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
  vector<ImageData> train_data = load_all_batches();
  vector<ImageData> test_data = load_test_batch();

  vector<Perceptron> perceptrones;
  for (int label = 0; label < CANT_PERCEPTRONS; label++) {
    perceptrones.push_back(
        Perceptron(label, train_data[0].pixels.size(), 0, 0.1, train_data));
  }

  // 1. PRIMERO ENTRENAR todos los perceptrones

  cout << "\n=== ENTRENAMIENTO ===" << endl;
  for (auto &p : perceptrones) {
    p.entrenamiento(20); // Solo 2 épocas
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

    if (i < 20) { // Solo mostrar primeras 20
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