CC = nvcc
APP = autograd
ENG = engine
FLAGS = -g -G -O0
all: $(ENG)

$(ENG) : ./$(APP)/gd.cu ./$(APP)/$(ENG).c ./$(APP)/main.c
	$(CC) $(FLAGS) $^ -o $@.out

clean:
	rm -f $(ENG)*.out


