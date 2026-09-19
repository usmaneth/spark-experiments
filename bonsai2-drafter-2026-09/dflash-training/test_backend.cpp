#include "llama.h"
#include "llama-ext.h"
#include <cstdio>

int main() {
    llama_backend_init();
    printf("llama backend initialized successfully!\n");
    llama_backend_free();
    return 0;
}
