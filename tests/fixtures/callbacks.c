/* Test file for indirect call analysis */

#include <stdlib.h>

/* Callback type definitions */
typedef void (*event_handler_t)(void* data, int event_type);
typedef int (*comparator_t)(const void* a, const void* b);
typedef void* (*allocator_t)(size_t size);

/* Callback implementations */
void log_event(void* data, int event_type) {
    /* Log the event */
}

void handle_click(void* data, int event_type) {
    /* Handle click event */
}

void handle_keypress(void* data, int event_type) {
    /* Handle keypress event */
}

int compare_int(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

int compare_string(const void* a, const void* b) {
    return strcmp(*(char**)a, *(char**)b);
}

void* default_alloc(size_t size) {
    return malloc(size);
}

void* zeroed_alloc(size_t size) {
    return calloc(1, size);
}

/* Event system using callbacks */
struct EventHandler {
    event_handler_t on_event;
    void* user_data;
};

void dispatch_event(struct EventHandler* handler, int event_type) {
    if (handler && handler->on_event) {
        handler->on_event(handler->user_data, event_type);  /* Indirect call */
    }
}

void register_handler(struct EventHandler* handler, event_handler_t callback) {
    handler->on_event = callback;  /* Address taken */
}

/* Array of function pointers */
static event_handler_t event_handlers[10];
static int handler_count = 0;

void add_handler(event_handler_t handler) {
    if (handler_count < 10) {
        event_handlers[handler_count++] = handler;  /* Address taken */
    }
}

void dispatch_all(int event_type) {
    for (int i = 0; i < handler_count; i++) {
        event_handlers[i](NULL, event_type);  /* Indirect call through array */
    }
}

/* Custom allocator pattern */
struct Allocator {
    allocator_t alloc;
    void (*free)(void*);
};

void* allocator_alloc(struct Allocator* a, size_t size) {
    return a->alloc(size);  /* Indirect call to allocator */
}

struct Allocator* create_default_allocator(void) {
    struct Allocator* a = malloc(sizeof(struct Allocator));
    if (a) {
        a->alloc = default_alloc;  /* Address taken */
        a->free = free;
    }
    return a;
}

/* Example usage */
void example_usage(void) {
    struct EventHandler handler;
    register_handler(&handler, log_event);  /* Address taken */
    dispatch_event(&handler, 42);

    add_handler(handle_click);    /* Address taken */
    add_handler(handle_keypress); /* Address taken */
    dispatch_all(1);

    struct Allocator* alloc = create_default_allocator();
    void* mem = allocator_alloc(alloc, 100);  /* Uses indirect call */
    free(mem);
    free(alloc);
}
