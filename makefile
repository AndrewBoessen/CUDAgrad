# Compiler
CC = gcc
NVCC = nvcc

# Compiler flags
CFLAGS = -Wall -Wextra -std=c99
NVCCFLAGS = -arch=sm_89 -O3

# Include directories
INC_DIRS = -I./autograd/

# Source files
C_SOURCES = autograd/nn_demo.c autograd/moons_demo.c autograd/engine.c autograd/data.c
CU_SOURCES = autograd/gd.cu autograd/nn.cu

# Object files
C_OBJECTS = $(C_SOURCES:.c=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# Demos
NN = nn_demo
MOONS = moons_demo

# Executable
EXECUTABLES = $(NN) $(MOONS)

# Build rules
all: $(EXECUTABLES)

$(NN): $(C_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(CU_OBJECTS) $(filter-out autograd/moons_demo.o,$(C_OBJECTS)) -o $@

$(MOONS): $(C_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(CU_OBJECTS) $(filter-out autograd/nn_demo.o,$(C_OBJECTS)) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC_DIRS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INC_DIRS) -c $< -o $@

clean:
	rm -f $(EXECUTABLES) $(C_OBJECTS) $(CU_OBJECTS)
