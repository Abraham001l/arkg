#include "endpoints.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <unistd.h>

namespace {
filesystem::path resolve_path(const filesystem::path& root, const string& path, bool must_exist = true) {
    if (path.find('\0') != string::npos) {
        throw runtime_error("Invalid path");
    }
    const auto resolved = must_exist ? filesystem::canonical(root / path)
                                     : filesystem::weakly_canonical(root / path);
    const auto relative = resolved.lexically_relative(root);
    if (relative.empty() || *relative.begin() == "..") {
        throw runtime_error("Path is outside the file host root");
    }
    return resolved;
}

}

void register_endpoints(httplib::Server& server, const string& root_path) {
    const auto root = filesystem::canonical(root_path);
    if (!filesystem::is_directory(root)) {
        throw runtime_error("The file host root must be a directory");
    }

    server.set_payload_max_length(numeric_limits<size_t>::max());

    server.Post("/create_directory", [root](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto options = nlohmann::json::parse(req.body);
            auto directory = resolve_path(root, options.at("path").get<string>(), false);
            if (options.value("unique", false)) {
                if (directory == root) throw runtime_error("A unique directory needs a name within the root");
                filesystem::create_directories(directory.parent_path());
                string pattern = directory.string() + "-XXXXXX";
                if (!mkdtemp(pattern.data())) throw runtime_error("Failed to create directory");
                directory = pattern;
            } else {
                filesystem::create_directories(directory);
            }
            res.set_content(nlohmann::json{{"path", directory.lexically_relative(root).generic_string()}}.dump(),
                             "application/json");
        } catch (const exception& error) {
            res.status = 500;
            res.set_content(error.what(), "text/plain");
        }
    });

    server.Post("/save_file", [root](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto path = resolve_path(root, req.get_param_value("path"), false);
            save_file(path, req.body);
            res.set_content(nlohmann::json{{"byte_length", req.body.size()}}.dump(), "application/json");
        } catch (const exception& error) {
            res.status = 500;
            res.set_content(error.what(), "text/plain");
        }
    });

    server.Get("/crawl_dir", [root](const httplib::Request& req, httplib::Response& res) {
        const string path = req.get_param_value("path");

        vector<string> doc_paths;
        try {
            crawl_dir(resolve_path(root, path).string(), doc_paths, req.get_param_values("exclude"));
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
            return;
        }

        res.set_content(nlohmann::json(doc_paths).dump(), "application/json");
    });

    server.Get("/get_file", [root](const httplib::Request& req, httplib::Response& res) {
        const string path = req.get_param_value("path");

        vector<char> file_contents;
        try {
            const auto file = resolve_path(root, path);
            if (!filesystem::is_regular_file(file)) {
                throw runtime_error("Path must refer to a regular file");
            }
            get_file(file.string(), file_contents);
            res.set_content(file_contents.empty() ? "" : file_contents.data(),
                            file_contents.size(), "application/octet-stream");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
        }
    });
}

void crawl_dir(const string& path, vector<string>& doc_paths, const vector<string>& excluded_directories) {
    for (auto it = filesystem::recursive_directory_iterator(path);
         it != filesystem::recursive_directory_iterator(); ++it) {
        const auto& entry = *it;
        if (entry.is_directory() && find(excluded_directories.begin(), excluded_directories.end(),
                                        entry.path().filename().string()) != excluded_directories.end()) {
            it.disable_recursion_pending();
            continue;
        }
        if (filesystem::is_regular_file(entry.symlink_status())) {
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

void save_file(const filesystem::path& path, const string& bytes) {
    string temporary = path.string() + ".tmp-XXXXXX";
    const int fd = mkstemp(temporary.data());
    if (fd == -1) throw runtime_error("Failed to create temporary file");
    const auto remove_temporary = [](const char* name) { unlink(name); };
    const unique_ptr<const char, decltype(remove_temporary)> cleanup(temporary.c_str(), remove_temporary);
    FILE* file = fdopen(fd, "wb");
    if (!file) {
        close(fd);
        throw runtime_error("Failed to open temporary file");
    }
    const size_t written = fwrite(bytes.data(), 1, bytes.size(), file);
    const int status = fclose(file);
    if (written != bytes.size() || status != 0) throw runtime_error("Failed to write file");
    // Publish the completed file atomically, rejecting any existing destination.
    if (link(temporary.c_str(), path.c_str()) != 0) {
        throw runtime_error("Failed to save file; destination may already exist");
    }
}
