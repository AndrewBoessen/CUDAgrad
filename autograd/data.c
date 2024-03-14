#include "data.h"

int load_data(const char *filename, Entry dataset[]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return -1;
    }

    int count = 0;
    char line[100]; // Assuming maximum line length of 100 characters

    while (fgets(line, sizeof(line), file) && count < MAX_ENTRIES) {
        if (sscanf(line, "%f,%f,%d", &dataset[count].x, &dataset[count].y, &dataset[count].label) == 3) {
            count++;
        } else {
            fprintf(stderr, "Error reading line %d from file %s\n", count + 1, filename);
        }
    }

    fclose(file);
    return count;
}