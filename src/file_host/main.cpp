#include "endpoints.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2 || argv[1][0] == '\0') {
        std::fprintf(stderr, "Usage: %s <root_directory>\n", argv[0]);
        return 1;
    }

    try {
        httplib::Server server;
        register_endpoints(server, argv[1]);
        return server.listen("127.0.0.1", 8081) ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
