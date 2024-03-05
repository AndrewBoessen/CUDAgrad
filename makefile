CC = nvcc
FLAGS = -Wall -Werror -std=c99
APP = autograd
ENG = engine

all: $(ENG)

$(ENG) : ./$(APP)/$(ENG).c ./$(APP)/gd.cu ./$(APP)/main.c
	$(CC) $^ -o $@.out

clean:
	rm -f $(ENG).out


