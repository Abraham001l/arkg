#pragma once
#include <httplib.h>
#include <filesystem>
using namespace std;

void register_endpoints(httplib::Server& server);

void crawl_dir(const string& path, vector<string>& doc_paths);

void get_file(const string& path, vector<char>& file_contents);