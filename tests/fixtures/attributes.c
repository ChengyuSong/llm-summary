/* Test file for function attribute detection. */

#include <stdlib.h>

/* Constructor attribute - called before main() */
__attribute__((constructor))
void init_subsystem(void) {
    /* Initialize subsystem */
}

/* Destructor attribute - called after main() */
__attribute__((destructor))
void cleanup_subsystem(void) {
    /* Cleanup subsystem */
}

/* Weak symbol - can be overridden by another definition */
__attribute__((weak))
void default_handler(int sig) {
    /* Default signal handler */
}

/* Section placement - .init.text */
__attribute__((section(".init.text")))
void early_init(void) {
    /* Early initialization */
}

/* Regular function (no special attributes) */
void regular_function(int x) {
    /* Nothing special */
}

/* Another regular function */
int compute(int a, int b) {
    return a + b;
}
