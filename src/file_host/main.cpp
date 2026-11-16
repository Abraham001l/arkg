#include "endpoints.h"

int main() {
    httplib::Server server;
    register_endpoints(server);
    return server.listen("127.0.0.1", 8081) ? 0 : 1;
}
