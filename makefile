CC = nvcc
FLAGS = -diag-suppress=20039
APP = autograd
ENG = engine

all: $(ENG)_cpu $(ENG)_gpu

$(ENG)_cpu : ./$(APP)/$(ENG).c ./$(APP)/gd.cu ./$(APP)/main.c
	$(CC) $(FALGS) $^ -o $@.out

$(ENG)_gpu : ./$(APP)/$(ENG).c ./$(APP)/gd.cu ./$(APP)/main.c
	$(CC) $(FLAGS) $^ -D CUDA -o $@.out

clean:
	rm -f $(ENG)*.out


