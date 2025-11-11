CXX = hipcc
TARGET = perceptron_cifar10
SOURCES = perceptron.hip

HIP_PATH ?= /opt/rocm
ROCM_PATH ?= /opt/rocm

# Flags CORREGIDOS para AMD
CXXFLAGS = -O2 -std=c++17 -Wall -I$(HIP_PATH)/include -D__HIP_PLATFORM_AMD__
LDFLAGS = -L$(HIP_PATH)/lib -lamdhip64

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean