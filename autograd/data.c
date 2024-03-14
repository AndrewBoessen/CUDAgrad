#include "data.h"

/**
 * Parses a line from the CSV file and populates an Entry struct.
 * Returns 1 on success, 0 on failure.
 */
int parse_line(char* line, Entry* entry) {
    char* token;
    char* saveptr;
    int i = 0;
    float values[3];

    token = strtok_r(line, ",", &saveptr);
    while (token != NULL && i < 3) {
        values[i++] = atof(token);
        token = strtok_r(NULL, ",", &saveptr);
    }

    if (i != 3) {
        return 0;
    }

    entry->x = values[0];
    entry->y = values[1];
    entry->label = (int)values[2];

    return 1;
}

/**
 * Loads data from a CSV file into an array of Entry structs.
 * Returns the number of entries loaded, or -1 on error.
 */
int load_data(const char* filename, Entry** entries) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Failed to open file %s\n", filename);
        return -1;
    }

    char line[256];
    int count = 0;
    *entries = (Entry*)malloc(sizeof(Entry) * MAX_ENTRIES);

    while (fgets(line, sizeof(line), file)) {
        if (count >= MAX_ENTRIES) {
            fprintf(stderr, "Error: Maximum number of entries reached\n");
            break;
        }

        if (!parse_line(line, &(*entries)[count])) {
            fprintf(stderr, "Error: Failed to parse line: %s\n", line);
            continue;
        }

        count++;
    }

    fclose(file);
    return count;
}