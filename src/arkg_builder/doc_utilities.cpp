#include "doc_utilities.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <string_view>
#include <unordered_map>
#include <unistd.h>

namespace {
int mpi_count(std::uint64_t count) {
    if (count > static_cast<std::uint64_t>(numeric_limits<int>::max())) {
        throw length_error("Document payload exceeds the MPI message count limit");
    }
    return static_cast<int>(count);
}

void read_text_file(string_view bytes, docs_container& global_docs) {
    if (bytes.empty()) return;

    global_docs.doc_mem.insert(global_docs.doc_mem.end(), bytes.begin(), bytes.end());
    global_docs.doc_sizes.push_back(bytes.size());
}

void read_pdf_file(string_view bytes, docs_container& global_docs) {
    if (bytes.empty()) return;

    // pdftotext opens a local copy of the downloaded bytes.
    char path[] = "/tmp/arkg-pdf-XXXXXX";
    const int fd = mkstemp(path);
    if (fd == -1) {
        perror("Error creating temporary PDF");
        return;
    }
    const auto remove_file = [](const char* name) { unlink(name); };
    const unique_ptr<char, decltype(remove_file)> temporary_file(path, remove_file);

    FILE* file = fdopen(fd, "wb");
    if (!file) {
        close(fd);
        fprintf(stderr, "Error opening temporary PDF\n");
        return;
    }
    const size_t written = fwrite(bytes.data(), 1, bytes.size(), file);
    const int close_status = fclose(file);
    if (written != bytes.size() || close_status != 0) {
        fprintf(stderr, "Error writing temporary PDF\n");
        return;
    }

    const string command = string("pdftotext -enc UTF-8 -nopgbrk ") + path + " -";
    const auto close_pipe = [](FILE* stream) { pclose(stream); };
    unique_ptr<FILE, decltype(close_pipe)> pipe(popen(command.c_str(), "r"), close_pipe);
    if (!pipe) {
        fprintf(stderr, "Error starting pdftotext\n");
        return;
    }

    string text;
    array<char, 4096> buffer;
    while (const size_t count = fread(buffer.data(), 1, buffer.size(), pipe.get())) {
        text.append(buffer.data(), count);
    }
    const bool read_failed = ferror(pipe.get()) != 0;
    const int status = pclose(pipe.release());
    if (read_failed || status != 0) {
        fprintf(stderr, "Error extracting PDF text\n");
        return;
    }

    if (text.find_first_not_of(" \t\r\n\f\v") != string::npos) {
        read_text_file(text, global_docs);
    }
}
}

bool pick_doc_retriever(const string& doc_path, int my_rank, int& master_rank) {
    // check if the folder` exists
    error_code ec;
    bool found = filesystem::exists(doc_path, ec);

    // getting leader rank
    int candidate_rank = (found && !ec) ? my_rank : numeric_limits<int>::max();

    int owner_rank;
    MPI_Allreduce(
        &candidate_rank, 
        &owner_rank, 
        1, 
        MPI_INT, 
        MPI_MIN, 
        MPI_COMM_WORLD);

    master_rank = owner_rank;

    return owner_rank != numeric_limits<int>::max();
}

void populate_docs_container(httplib::Client& client, docs_container& global_docs,
                             const vector<string>& doc_paths) {
    using FileParser = void (*)(string_view, docs_container&);
    static const unordered_map<string, FileParser> readers{
        {".txt", read_text_file},
        {".pdf", read_pdf_file},
    };

    global_docs.doc_mem.reserve(one_GB);
    global_docs.doc_sizes.reserve(doc_paths.size());
    global_docs.doc_paths.reserve(doc_paths.size());

    for (const string& path : doc_paths) {
        // find appropiate reader for the file extension
        const auto reader = readers.find(filesystem::path(path).extension().string());
        if (reader == readers.end()) continue;

        // download the file from the server
        const httplib::Params params{{"path", path}};
        const auto response = client.Get("/get_file", params);
        if (!response || response->status != 200) {
            const string error = response ? "HTTP " + to_string(response->status)
                                          : httplib::to_string(response.error());
            fprintf(stderr, "Failed to download %s: %s\n", path.c_str(), error.c_str());
            continue;
        }

        // read the file into the global_docs container
        const size_t doc_count = global_docs.doc_sizes.size();
        reader->second(response->body, global_docs);
        if (global_docs.doc_sizes.size() != doc_count) {
            global_docs.doc_paths.push_back(path);
        }
    }
}

void calc_offsets(docs_container& d, size_t world_size) {
    const size_t total_bytes = d.doc_mem.size();
    const size_t rank_count = min(world_size, d.doc_sizes.size());
    d.worker_offsets.clear();
    d.worker_offsets.reserve(world_size);

    size_t bytes_remaining = total_bytes;
    size_t buckets_left = rank_count;
    size_t current_bucket_size = 0;
    size_t current_doc_offset = 0;
    size_t items_left = d.doc_sizes.size();

    for (size_t i = 0; i < d.doc_sizes.size(); ++i) {
        const size_t doc_size = d.doc_sizes[i];

        const double target = static_cast<double>(bytes_remaining) / static_cast<double>(buckets_left);
        const double size_if_added = static_cast<double>(current_bucket_size + doc_size);
        const double overshoot = size_if_added - target;
        const double undershoot = target - static_cast<double>(current_bucket_size);

        // Split now if each remaining worker needs one of the remaining documents.
        if (current_bucket_size > 0 &&
            buckets_left > 1 &&
            (items_left == buckets_left - 1 || overshoot > undershoot)) {
                // finalize the current bucket and move to the next one
                d.worker_offsets.push_back(current_doc_offset - current_bucket_size);
                bytes_remaining -= current_bucket_size;
                --buckets_left;

                // new bucket takes overflow of last
                current_bucket_size = doc_size;
        } else {
            current_bucket_size += doc_size;
        }

        current_doc_offset += doc_size;
        --items_left;
    }

    // finalize the last bucket
    if (d.worker_offsets.size() < rank_count) {
        d.worker_offsets.push_back(total_bytes - current_bucket_size);
    }

    // every rank gets an offset. repeated end offsets represent empty work.
    d.worker_offsets.resize(world_size, total_bytes);
}

void distribute_docs_to_ranks(const docs_container& global_docs, docs_container& local_docs,
                              int world_size, int my_rank) {
    size_t next_doc = 0;
    for (int rank = 0; rank < world_size; ++rank) {
        // getting raw text byte range for this rank
        const size_t start = global_docs.worker_offsets[rank];
        const size_t end = rank + 1 == world_size ? global_docs.doc_mem.size() : global_docs.worker_offsets[rank + 1];
        const auto text = span<const char>(global_docs.doc_mem).subspan(start, end - start);

        // looping through doc_sizes to find range
        const size_t first_doc = next_doc;
        size_t remaining = text.size();
        while (remaining != 0) {
            remaining -= global_docs.doc_sizes[next_doc++];
        }
        const auto sizes = span<const size_t>(global_docs.doc_sizes).subspan(first_doc, next_doc - first_doc);

        // Copy the master's shard into its local vectors so it can release global_docs.
        if (rank == my_rank) {
            local_docs.doc_mem.assign(text.begin(), text.end());
            local_docs.doc_sizes.assign(sizes.begin(), sizes.end());
            local_docs.worker_offsets.clear();
            continue;
        }

        // an empty rank only receives the header
        // header contains byte count for raw text and doc count for sizes array
        const std::array<std::uint64_t, 2> header{text.size(), sizes.size()};
        const int byte_count = mpi_count(header[0]);
        const int doc_count = mpi_count(header[1]);
        MPI_Send(header.data(), 2, MPI_UINT64_T, rank, size_tag, MPI_COMM_WORLD);
        if (byte_count == 0) continue;

        // send the sizes array and the raw text to the rank
        const vector<std::uint64_t> wire_sizes(sizes.begin(), sizes.end());
        MPI_Send(wire_sizes.data(), doc_count, MPI_UINT64_T, rank, doc_sizes_tag, MPI_COMM_WORLD);
        MPI_Send(text.data(), byte_count, MPI_CHAR, rank, data_tag, MPI_COMM_WORLD);
    }
}

void receive_docs_from_master(docs_container& local_docs, int master_rank) {
    // get header
    std::array<std::uint64_t, 2> header{};
    MPI_Recv(header.data(), 2, MPI_UINT64_T, master_rank, size_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const int byte_count = mpi_count(header[0]);
    const int doc_count = mpi_count(header[1]);

    local_docs.doc_mem.resize(byte_count);
    local_docs.worker_offsets.clear();
    if (byte_count == 0) {
        local_docs.doc_sizes.clear();
        return;
    }

    vector<std::uint64_t> wire_sizes(doc_count);
    MPI_Recv(wire_sizes.data(), doc_count, MPI_UINT64_T, master_rank, doc_sizes_tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_docs.doc_sizes.assign(wire_sizes.begin(), wire_sizes.end());

    MPI_Recv(local_docs.doc_mem.data(), byte_count, MPI_CHAR, master_rank, data_tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void crawl_dir(httplib::Client& client, const string& path, vector<string>& doc_paths) {
    const httplib::Params params{{"path", path}};
    const auto res = client.Get("/crawl_dir", params);

    if (!res || res->status != 200) {
        throw runtime_error("Failed to crawl directory: " + path);
    }

    doc_paths = nlohmann::json::parse(res->body).get<vector<string>>();
}
