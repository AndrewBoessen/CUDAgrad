CC = nvcc
APP = autograd
ENG = engine

all: $(ENG)_cpu $(ENG)_gpu

$(ENG)_cpu : ./$(APP)/$(ENG).c ./$(APP)/gd.cu ./$(APP)/main.c
	$(CC) $^ -o $@.out

$(ENG)_gpu : ./$(APP)/$(ENG).c ./$(APP)/gd.cu ./$(APP)/main.c
	$(CC) $^ -D CUDA -o $@.out

clean:
	rm -f $(ENG)*.out


