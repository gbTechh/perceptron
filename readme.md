# Proyecto Perceptron

**Integrantes:**

- Rodrigo Silva
- Pamela Villar

## Ejecución

### Implementación en C++ (main.cpp)

- Perceptrón implementado en C++ utilizando threads para paralelización
- Tiempo de ejecución: menos de 2 minutos para 20 épocas
- Dataset: 50,000 muestras para entrenamiento y 10,000 para testeo
- Arquitectura: perceptrón simple (el mismo aprendido en clase)
- Resultados: entre 22-26% de aciertos
- Dataset: CIFAR10[https://www-cs-toronto-edu.translate.goog/~kriz/cifar-10-binary.tar.gz?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc]

### Implementación en CUDA (perceptron.cu)

- Misma lógica que la versión en C++ pero implementada en CUDA
- Tiempo de ejecución: ligeramente mayor que la versión con threads
- Entorno de prueba: Google Colab
- Resultados: porcentaje de aciertos ligeramente inferior (menor al 20%)
- **Observaciones:**
  - Aún estamos aprendiendo a utilizar CUDA
  - No entendemos completamente por qué hay mayor tiempo de ejecución
  - Tampoco comprendemos la razón del menor porcentaje de aciertos

Los dataset ya estan en el proyecto, para ejecutar solo hay que compilar el main.cpp y perceptron.cu por separado
