# ============================
#   Makefile for MLP Project
# ============================

CXX = g++
CXXFLAGS = -std=c++17 -O2 -I/opt/homebrew/include/eigen3

# Busca TODOS los .cpp del proyecto
SRC = $(wildcard *.cpp)

# Genera automáticamente un .o por cada .cpp
OBJ = $(SRC:.cpp=.o)

# Nombre del ejecutable
TARGET = main

# Regla principal
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET)

# Regla para compilar cada .cpp a .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Limpiar archivos generados
clean:
	rm -f *.o $(TARGET)

# Limpiar todo
distclean: clean
	rm -f $(TARGET)