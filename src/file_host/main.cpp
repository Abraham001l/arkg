#include "endpoints.h"
#include <cstdio>
#include <charconv>
#include <string_view>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3 || argv[1][0] == '\0') {
        std::fprintf(stderr, "Usage: %s <root_directory> <port>\n", argv[0]);
        return 1;
    }

    try {
        int port;
        const std::string_view value(argv[2]);
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), port);
        if (error != std::errc{} || end != value.data() + value.size() || port < 1 || port > 65535) {
            throw std::runtime_error("Port must be an integer between 1 and 65535");
        }

        httplib::Server server;
        register_endpoints(server, argv[1]);
        return server.listen("127.0.0.1", port) ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
