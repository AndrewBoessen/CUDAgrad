CC = nvcc
APP = autograd
ENG = engine

all: $(ENG)

$(ENG) : ./$(APP)/$(ENG).c ./$(APP)/gd.cu ./$(APP)/main.c
	$(CC) $^ -o $@.out

clean:
	rm -f $(ENG)*.out


