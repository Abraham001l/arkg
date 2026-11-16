#include "endpoints.h"
#include <nlohmann/json.hpp>

void register_endpoints(httplib::Server& server) {

    server.Get("/crawl_dir", [](const httplib::Request& req, httplib::Response& res) {
        const string path = req.get_param_value("path");

        vector<string> doc_paths;
        try {
            crawl_dir(path, doc_paths);
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
            return;
        }

        res.set_content(nlohmann::json(doc_paths).dump(), "application/json");
    });

    server.Get("/get_file", [](const httplib::Request& req, httplib::Response& res) {
        const string path = req.get_param_value("path");

        vector<char> file_contents;
        try {
            get_file(path, file_contents);
            res.set_content(file_contents.data(), file_contents.size(), "application/octet-stream");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
        }
    });
}

void crawl_dir(const string& path, vector<string>& doc_paths) {
    for (const auto& entry : filesystem::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
            doc_paths.push_back(entry.path().string());
        }
    }
}

void get_file(const string& path, vector<char>& file_contents) {
    ifstream file(path, ios::binary);
    if (!file) {
        throw runtime_error("Failed to open file: " + path);
    }
    file_contents.assign(istreambuf_iterator<char>(file), istreambuf_iterator<char>());
}
