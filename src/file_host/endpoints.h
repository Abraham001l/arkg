#pragma once
#include <httplib.h>
#include <filesystem>
using namespace std;

void register_endpoints(httplib::Server& server, const string& root_path);

void crawl_dir(const string& path, vector<string>& doc_paths,
               const vector<string>& excluded_directories = {});

void get_file(const string& path, vector<char>& file_contents);

// Publishes a complete file and rejects an existing destination.
void save_file(const filesystem::path& path, const string& bytes);
