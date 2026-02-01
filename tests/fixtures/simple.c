/* Simple C file for testing function extraction and allocation analysis */

#include <stdlib.h>
#include <string.h>

/* Simple allocation wrapper */
char* create_buffer(size_t n) {
    char* buf = malloc(n + 1);
    if (!buf) return NULL;
    buf[n] = '\0';
    return buf;
}

/* Allocation with size from struct field */
struct Buffer {
    size_t capacity;
    size_t length;
    char* data;
};

struct Buffer* buffer_new(size_t initial_capacity) {
    struct Buffer* buf = malloc(sizeof(struct Buffer));
    if (!buf) return NULL;

    buf->data = malloc(initial_capacity);
    if (!buf->data) {
        free(buf);
        return NULL;
    }

    buf->capacity = initial_capacity;
    buf->length = 0;
    return buf;
}

void buffer_free(struct Buffer* buf) {
    if (buf) {
        free(buf->data);
        free(buf);
    }
}

/* Reallocation */
int buffer_grow(struct Buffer* buf, size_t new_capacity) {
    if (new_capacity <= buf->capacity) {
        return 0;
    }

    char* new_data = realloc(buf->data, new_capacity);
    if (!new_data) {
        return -1;
    }

    buf->data = new_data;
    buf->capacity = new_capacity;
    return 0;
}

/* String duplication */
char* my_strdup(const char* s) {
    size_t len = strlen(s);
    char* copy = malloc(len + 1);
    if (copy) {
        memcpy(copy, s, len + 1);
    }
    return copy;
}

/* No allocation */
int buffer_length(const struct Buffer* buf) {
    return buf ? buf->length : 0;
}

/* Conditional allocation */
char* maybe_allocate(int should_allocate, size_t size) {
    if (should_allocate) {
        return malloc(size);
    }
    return NULL;
}

/* Allocation stored to output parameter */
int create_pair(char** out_first, char** out_second, size_t size) {
    *out_first = malloc(size);
    if (!*out_first) return -1;

    *out_second = malloc(size);
    if (!*out_second) {
        free(*out_first);
        *out_first = NULL;
        return -1;
    }

    return 0;
}
