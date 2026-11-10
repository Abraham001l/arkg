#include "doc_utilities.h"

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

void populate_docs_container(docs_container& d, char* data_ptr, const vector<string>& doc_paths) {
    vector<size_t> temp_doc_sizes;
    size_t cur_total_size = 0;

    for (const string& path : doc_paths) {
        // opening file & reading in contents
        ifstream file(path, ios::binary | ios::ate);

        if (!file.is_open()) {
            printf("Error opening file: %s\n", path.c_str());
            continue;
        }

        streamsize new_content_size = file.tellg();

        if (new_content_size <= 0) {
            printf("Error reading file: %s\n", path.c_str());
            continue;
        }

        file.seekg(0, ios::beg);
        file.read(data_ptr + d.doc_mem_offset + cur_total_size, new_content_size);

        temp_doc_sizes.push_back(new_content_size);
        cur_total_size += new_content_size;
    }

    // copying in doc_size's vector
    d.doc_sizes_offset = d.doc_mem_offset + cur_total_size;
    memcpy(data_ptr + d.doc_sizes_offset, temp_doc_sizes.data(), temp_doc_sizes.size() * sizeof(size_t));

    // assigning spans
    d.doc_mem = span<char>(data_ptr + d.doc_mem_offset, cur_total_size);
    d.doc_sizes = span<size_t>(reinterpret_cast<size_t*>(data_ptr + d.doc_sizes_offset), temp_doc_sizes.size());

    // copying doc_sizes_offset & doc_mem_offset into memory
    memcpy(data_ptr + sizeof(size_t), &d.doc_sizes_offset, sizeof(size_t));
    memcpy(data_ptr, &d.doc_mem_offset, sizeof(size_t));
}

void calc_rank_offsets(docs_container& d, char* data_ptr, size_t world_size) {
    const size_t total_bytes = d.doc_mem.size();
    const size_t rank_count = min(world_size, d.doc_sizes.size());
    vector<size_t> temp_rank_offsets;
    temp_rank_offsets.reserve(world_size);

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

        if (current_bucket_size > 0 &&
            buckets_left > 1 &&
            items_left >= buckets_left &&
            overshoot > undershoot) {
                // finalize the current bucket and move to the next one
                temp_rank_offsets.push_back(current_doc_offset - current_bucket_size);
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
    if (temp_rank_offsets.size() < rank_count) {
        temp_rank_offsets.push_back(total_bytes-current_bucket_size);
    }

    // Every rank gets an offset. Repeated end offsets represent empty work.
    temp_rank_offsets.resize(world_size, total_bytes);
    // assigning rank_offsets_offset
    d.rank_offsets_offset = d.doc_sizes_offset + d.doc_sizes.size() * sizeof(size_t);

    // copying in rank_offsets vector & assigning span
    memcpy(data_ptr + d.rank_offsets_offset, temp_rank_offsets.data(), temp_rank_offsets.size() * sizeof(size_t));
    d.rank_offsets = span<size_t>(reinterpret_cast<size_t*>(data_ptr + d.rank_offsets_offset), temp_rank_offsets.size());

    // assigning rank_offsets_offset into memory
    memcpy(data_ptr + sizeof(size_t)*2, &d.rank_offsets_offset, sizeof(size_t));
}

void distribute_docs_to_ranks(docs_container& d, char* data_ptr, size_t world_size, int my_rank) {
    // iterating through rank_offsets and sending the appropriate data to each rank
    for (int rank = 0; rank < world_size; ++rank) {
        if (rank == my_rank) {
            // this rank already has the data, no need to send
            continue;
        }
        
        // calculating the start and end offsets for the current rank
        size_t start_offset = d.rank_offsets[rank];
        size_t end_offset = (rank == d.rank_offsets.size() - 1) ? d.doc_sizes_offset - d.doc_mem_offset : d.rank_offsets[rank + 1];
        size_t data_size = end_offset - start_offset;

        // sending the data in two parts: first the size, then the actual data
        MPI_Send(
            &data_size, 
            1, 
            MPI_UINT64_T, 
            rank, 
            size_tag, 
            MPI_COMM_WORLD);

        // The size message is sufficient for ranks with no assigned bytes.
        if (data_size == 0) continue;

        MPI_Send(
            data_ptr + d.doc_mem_offset + start_offset, 
            data_size, 
            MPI_CHAR, 
            rank, 
            data_tag, 
            MPI_COMM_WORLD);
    }
}

void receive_docs_from_master(work_container& w, int master_rank) {
    // receiving the data from the master rank
    MPI_Recv(
        &w.work_mem_size, 
        1, 
        MPI_UINT64_T, 
        master_rank, 
        size_tag, 
        MPI_COMM_WORLD, 
        MPI_STATUS_IGNORE);

    if (w.work_mem_size == 0) {
        w.work_mem = span<char>();
        w.owned_memory.reset();
        return;
    }

    w.owned_memory = make_unique_for_overwrite<char[]>(w.work_mem_size);
    w.work_mem = span<char>(w.owned_memory.get(), w.work_mem_size);
    MPI_Recv(
        w.work_mem.data(), 
        w.work_mem_size, 
        MPI_CHAR, 
        master_rank, 
        data_tag, 
        MPI_COMM_WORLD, 
        MPI_STATUS_IGNORE);
}

void fill_work_container_for_master_rank(work_container& w, docs_container& d, char* data_ptr, int my_rank) {
    // calculating the total size of the work memory
    size_t start_offset = d.rank_offsets[my_rank];
    size_t end_offset = (my_rank == d.rank_offsets.size() - 1) ? d.doc_sizes_offset - d.doc_mem_offset : d.rank_offsets[my_rank + 1];
    size_t total_size = end_offset - start_offset;
    w.work_mem_size = total_size;

    if (total_size == 0) {
        w.work_mem = span<char>();
        w.owned_memory.reset();
        return;
    }

    // allocating memory for the work memory
    w.owned_memory = make_unique_for_overwrite<char[]>(total_size);
    w.work_mem = span<char>(w.owned_memory.get(), total_size);

    // copying the relevant portion of the doc memory into the work memory
    memcpy(w.work_mem.data(), data_ptr + d.doc_mem_offset + start_offset, total_size);
}

void free_docs_container(docs_container& d) {
    // These spans do not own memory; main's unique_ptr frees the backing buffer.
    d.doc_mem = span<char>();
    d.doc_sizes = span<size_t>();
    d.rank_offsets = span<size_t>();
}
