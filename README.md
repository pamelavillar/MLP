# Implementaciones de MLP en Threads y CUDA

Este repositorio contiene dos implementaciones de un Perceptrón Multicapa (MLP): una basada en threads (CPU) y otra acelerada mediante CUDA (GPU). Ambas permiten comparar rendimiento y estructura interna del modelo.

---

## 1. Implementación en Threads (CPU)

La implementación en threads paraleliza operaciones del MLP usando múltiples hilos de CPU.

### Archivos incluidos
- `main.cpp`
- `mlp_threads.cpp`
- `Makefile`

### Compilación y ejecución
bash
make
./main

## 2. Implementación en CUDA (GPU)

La versión CUDA utiliza la GPU para acelerar los cálculos del MLP.

### Archivos incluidos
- `main.cu`
- `mlp.cu`

---

##  Cómo ejecutar la versión CUDA en Google Colab

### 1. Verificar GPU disponible
!nvidia-smi
### 2. Instalar CUDA Toolkit

!apt-get install -y nvidia-cuda-toolkit

### 3. Descargar dataset CIFAR-10

!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

!tar -xzf cifar-10-binary.tar.gz

!ls cifar-10-batches-bin/



### 4. Crear archivos cuda
%%writefile mlp_cuda.cu

// aquí el contenido de mlp.cu

%%writefile main.cu

// aquí el contenido de main.cu

### 5. Compilar usando NVCC

!nvcc -o mlp_cuda main.cu -O3 -arch=sm_75 --std=c++17

### 6. Ejecutar programa
!./mlp_cuda

