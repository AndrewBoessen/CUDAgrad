CC = nvcc
APP = autograd
ENG = engine
FLAGS = -g -G -O0
all: $(ENG)

$(ENG) : ./$(APP)/$(ENG).c ./$(APP)/gd.cu ./$(APP)/main.c
	$(CC) $(FLAGS) $^ -o $@.out

clean:
	rm -f $(ENG)*.out


