# Compiler
CC = gcc
NVCC = nvcc

# Compiler flags
CFLAGS = -Wall -Wextra -std=c99
NVCCFLAGS = -arch=sm_35

# Include directories
INC_DIRS = -I./autograd/

# Source files
C_SOURCES = autograd/nn_demo.c autograd/engine.c
CU_SOURCES = autograd/gd.cu autograd/nn.cu

# Object files
C_OBJECTS = $(C_SOURCES:.c=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# Executable
EXECUTABLE = nn_demo

# Build rules
all: $(EXECUTABLE)

$(EXECUTABLE): $(C_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(CU_OBJECTS) $(C_OBJECTS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC_DIRS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(INC_DIRS) -c $< -o $@

clean:
	rm -f $(EXECUTABLE) $(C_OBJECTS) $(CU_OBJECTS)


