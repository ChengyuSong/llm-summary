/* Test file for recursive function handling */

#include <stdlib.h>

/* Simple direct recursion */
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

/* Recursive with allocation */
struct Node {
    int value;
    struct Node* left;
    struct Node* right;
};

struct Node* create_node(int value) {
    struct Node* node = malloc(sizeof(struct Node));
    if (node) {
        node->value = value;
        node->left = NULL;
        node->right = NULL;
    }
    return node;
}

struct Node* build_tree(int* values, int count) {
    if (count == 0) return NULL;

    int mid = count / 2;
    struct Node* node = create_node(values[mid]);
    if (!node) return NULL;

    node->left = build_tree(values, mid);
    node->right = build_tree(values + mid + 1, count - mid - 1);

    return node;
}

void free_tree(struct Node* node) {
    if (node) {
        free_tree(node->left);
        free_tree(node->right);
        free(node);
    }
}

/* Mutual recursion */
int is_even(int n);
int is_odd(int n);

int is_even(int n) {
    if (n == 0) return 1;
    return is_odd(n - 1);
}

int is_odd(int n) {
    if (n == 0) return 0;
    return is_even(n - 1);
}

/* Recursive allocation with accumulator */
char* build_string_recursive(const char* parts[], int count, int current, char* acc) {
    if (current >= count) {
        return acc;
    }

    size_t acc_len = acc ? strlen(acc) : 0;
    size_t part_len = strlen(parts[current]);
    size_t new_len = acc_len + part_len + 1;

    char* new_acc = realloc(acc, new_len);
    if (!new_acc) {
        free(acc);
        return NULL;
    }

    if (acc_len == 0) {
        new_acc[0] = '\0';
    }
    strcat(new_acc, parts[current]);

    return build_string_recursive(parts, count, current + 1, new_acc);
}

char* build_string(const char* parts[], int count) {
    return build_string_recursive(parts, count, 0, NULL);
}
