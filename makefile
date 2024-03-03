CC = gcc
FLAGS = -Wall -Werror -std=c99
APP = autograd
ENG = engine

all: $(ENG)

$(ENG) : ./$(APP)/$(ENG).c ./$(APP)/main.c
	$(CC) $(FLAGS) $^ -o $@.out

clean:
	rm -f $(ENG).out


