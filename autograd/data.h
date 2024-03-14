/*
 * Header for loading data from external file
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_DIR = './data/'
#define MAX_ENTRIES 1000

/**
 * Struct for data points in the dataset.
 */
typedef struct {
    float x;
    float y;
    int label;
} Entry;

int parse_line(char* line, Entry* entry);

int load_data(const char* filename, Entry** entries);
