#include "common.h"
#include <mpi.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <set>
#include <algorithm>

struct Bin {
    int rank;
    int index;
    std::vector<particle_t*> particles;
    std::set<int> ghost_to;                 // stores rank indices of ranks that use this bin as a ghost bin
    std::vector<Bin*> neighbors;            // stores bins that are neighbors to use this (used in calculation)

    // Constructor to initialize member vectors
    Bin() 
        : rank(0), 
          index(0),
          particles(),
          ghost_to(),
          neighbors() {} 
};

static int num_bins_one_side;                           // number of bins in one row/column of the sim env
static Bin** grid;                                      // the grid of all the bins
static std::vector<Bin*> own_rank_bins;                 // the bins that belong to this rank
static std::unordered_set<Bin*> own_rank_ghost_bins;    // the bins that are ghost bins to this rank
static std::vector<Bin*> ghost_to_other_rank_bins;      // the bins that are ghost bins to another rank

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {

    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// Computes which bin a particle should go into
int compute_bin_index(particle_t* p) {
    return floor(p->x / cutoff) + floor(p->y / cutoff) * num_bins_one_side;
}

// Find which rank a particle's bin is owned by
int get_particle_rank(particle_t* p) {
    return grid[compute_bin_index(p)]->rank;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    // Instantiate the grid of bins
    num_bins_one_side = ceil(size / cutoff);
    int num_bins = num_bins_one_side * num_bins_one_side;

    grid = new Bin*[num_bins];
    for (int i = 0; i < num_bins; ++i) {
        grid[i] = new Bin();
    }
    
    // Assign particles to their initial bins
    for (int i = 0; i < num_parts; i++) {
        particle_t* part = parts + i;
        int bin_idx = compute_bin_index(part);
        grid[bin_idx]->particles.push_back(part);
    }

    // Assign bins for each rank
    // Split grid into chunk rows, divide the chunk rows as evenly as possible between all of the ranks
    int chunk_rows = ceil(sqrt(num_procs));     // number of chunk rows
    int chunk_cols[chunk_rows];                 // number of chunks in each chunk-row
    int chunk_row_dims[chunk_rows];             // row dimension of each chunk-row

    for (int chunk_row = 0; chunk_row < chunk_rows; ++chunk_row) {
        chunk_cols[chunk_row] = num_procs / chunk_rows;
        chunk_row_dims[chunk_row] = num_bins_one_side / chunk_rows;
    }
    for (int chunk_row = 0; chunk_row < num_procs % chunk_rows; ++chunk_row) {
        chunk_cols[chunk_row]++;
    }
    for (int i = 0; i < num_bins_one_side % chunk_rows; i++) {
        chunk_row_dims[i]++;
    }

    // Keep track of how many chunks are in each chunk row 
    std::vector<int> chunk_col_dims[chunk_rows];
    for (int i = 0; i < chunk_rows; i++) {
        chunk_col_dims[i].resize(chunk_cols[i]);
        int remainder_grids_in_row = num_bins_one_side % chunk_cols[i];
        for (int j = 0; j < chunk_col_dims[i].size(); j++) {
            chunk_col_dims[i][j] = num_bins_one_side / chunk_cols[i];
            if (j < remainder_grids_in_row) {
                chunk_col_dims[i][j] += 1;
            }
        }
    }
    
    // Figure out which bins are in chunks that have been assigned to this rank, and properly set their fields
    // and add them to this rank's list of bins to keep track of
    int chunk_row_idx = 0;
    int chunk_col_idx = 0;
    int bin_row_offset = 0;
    int bin_col_offset = 0;
    int rank_inc = 0;
    for (int chunk_row_idx = 0; chunk_row_idx < chunk_rows; chunk_row_idx++) {
        for (int chunk_col_idx = 0; chunk_col_idx < chunk_cols[chunk_row_idx]; chunk_col_idx++) {
            for (int grid_row_idx = 0; grid_row_idx < chunk_row_dims[chunk_row_idx]; grid_row_idx++) {
                for (int grid_col_idx = 0; grid_col_idx < chunk_col_dims[chunk_row_idx][chunk_col_idx]; grid_col_idx++) {

                    // identify bin that belongs to this rank
                    int target_bin_idx = (grid_row_idx + bin_row_offset) * num_bins_one_side + grid_col_idx + bin_col_offset;
                    Bin* target_bin = grid[target_bin_idx];

                    // properly set the bin's fields
                    target_bin->rank = rank_inc;
                    target_bin->index = target_bin_idx;

                    // add the bin to this rank's bins to keep track of
                    if (rank == rank_inc) {
                        own_rank_bins.push_back(target_bin);
                    }
                }
            }
            bin_col_offset += chunk_col_dims[chunk_row_idx][chunk_col_idx];
            rank_inc++;
        }
        bin_col_offset = 0;
        bin_row_offset += chunk_row_dims[chunk_row_idx];
    }
    

    // Figure out neighbors for each bin and assign them in the neighbors field
    for (int bin_idx = 0; bin_idx < num_bins; bin_idx++) {
        int col_idx = bin_idx % num_bins_one_side;
        int row_idx = bin_idx / num_bins_one_side;
        int row_min = fmax(0, row_idx - 1);
        int row_max = fmin(num_bins_one_side - 1, row_idx + 1);
        int col_min = fmax(0, col_idx - 1);
        int col_max = fmin(num_bins_one_side - 1, col_idx + 1);

        if (col_idx < num_bins_one_side - 1) {
            for (int row = row_min; row <= row_max; row++) {
                int neighbor_idx = col_idx + 1 + row * num_bins_one_side;
                
                grid[bin_idx]->neighbors.push_back(grid[neighbor_idx]);
                grid[neighbor_idx]->neighbors.push_back(grid[bin_idx]);
            }
        }
        if (row_idx > 0) {
            int neighbor_idx = bin_idx - num_bins_one_side;
            grid[bin_idx]->neighbors.push_back(grid[neighbor_idx]);
            grid[neighbor_idx]->neighbors.push_back(grid[bin_idx]);
        }
    }

    // Figure out which bins are not in the same rank as their neighbors and thus ghost
    for (int bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
        Bin* target_bin = grid[bin_idx];
        for (Bin* neighbor_bin: target_bin->neighbors) {
            if (target_bin->rank != neighbor_bin->rank) {
                target_bin->ghost_to.insert(neighbor_bin->rank);

                own_rank_ghost_bins.insert(neighbor_bin);  
            }
        }
    }

    // Figure out which of this rank's bins are ghost bins to bins in other ranks
    for (Bin* target_bin: own_rank_bins) {
        if (target_bin->ghost_to.size() > 0) {
            ghost_to_other_rank_bins.push_back(target_bin);
        }
    }
    
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function

    // For each grid cell object calculate forces between all particles
        // For particles within current grid cell
            // Intercel force 
                // Apply force
            // Intracel force
                // For neighbor particle
                    // Apply force
    
    // Compute foces on each particle in each bin
    for (Bin* target_bin: own_rank_bins) {
        for (particle_t* part: target_bin->particles) {    
            part->ax = part->ay = 0;

            // forces from particles in the same bin
            for (particle_t* other_part: target_bin->particles) {
                apply_force(*part, *other_part);                
            }

            // forces from particles in neighboring bins
            for (Bin* neighbor_bin: target_bin->neighbors){
                for (particle_t* neighbor_part: neighbor_bin->particles) {
                    apply_force(*part, *neighbor_part);
                    
                }
            }
        }
    }

    // Save particles to send (by p.ID), for each neighbor rank
    std::unordered_set<u_int64_t> send_particle_ids[num_procs];

    // For each bin
        // For each particle
            // Move particle
              // Update particle to its new bin 
            // After move, check to see if we need to send particle to another rank / take in any moved particles
            // that now belong to this rank
              // In implementation, tally up changes (amount + particle_ids for each neighbor rank) for MPI calls
            // 1. If particle moving into another chunk (owned by another rank)
            // 2. If particle started in a bin that is in ghost regions of other rank(s)
            // 3. If particle moved into a bin that is in ghost regions of other rank(s)
            // All independent checks, because these can all apply to the same particle 

    // Case 2: Because it started in a bin that is already a ghost bin to other ranks, 
    // all particles that started in this grid needs to be communicated
    // regardless of movement
    for (Bin* target_bin: ghost_to_other_rank_bins) {
        for (particle_t* part: target_bin->particles){
            for (int neighbor_rank: target_bin->ghost_to) {
                send_particle_ids[neighbor_rank].insert(part->id);
            }
        }
    }

    // A vector of ids of particles owned by this rank before all rebinning. Later used for grid updates.
    std::vector<int> own_rank_part_ids_premove;

    // Move the particles and check if the particle's rank changed
    // If so, this rank needs to communicate this new particle info with another rank
    for (Bin* target_bin: own_rank_bins) {
        for (particle_t* part: target_bin->particles) {
            own_rank_part_ids_premove.push_back(part->id);
            move(*part, size);

            int new_rank = get_particle_rank(part);
            if (rank != new_rank) {
                send_particle_ids[new_rank].insert(part->id);
            }
            Bin* new_part_bin = grid[compute_bin_index(part)];
            if (new_part_bin->ghost_to.size() > 0) { // Case 3
                for (int neighbor_rank: new_part_bin->ghost_to) {
                    send_particle_ids[neighbor_rank].insert(part->id);
                }
            }
        }
    }
    
    // Clear the particles in this rank's bin and ghost bins, so we can re-fill
    for (Bin* target_bin: own_rank_bins) {
        target_bin->particles.clear();
    }
    for (Bin* target_bin: own_rank_ghost_bins) {
        target_bin->particles.clear();
    }

    // Can update interally without needing to communicate with other ranks
    // if still in this rank or ghost bins
    for (int part_id: own_rank_part_ids_premove) {
        particle_t* part = parts + (part_id - 1);
        
        
        Bin* target_bin = grid[compute_bin_index(part)];
        if (target_bin->rank == rank || own_rank_ghost_bins.find(target_bin) != own_rank_ghost_bins.end()) {
            grid[compute_bin_index(part)]->particles.push_back(part);
        }
    }


  	// Communication section
    // For each other rank that has been affected
        // Put particles meant for that rank into buffer, send to that rank
        // Each rank based on the amount of particles to receive, allocate buffer, receive particles
        // Update local particles in parts

    // First exchange the amount of particles to send/receive
    int* send_particle_counts;
    int* receive_particle_counts;

    MPI_Alloc_mem(num_procs * sizeof(int), MPI_INFO_NULL, &send_particle_counts);
    MPI_Alloc_mem(num_procs * sizeof(int), MPI_INFO_NULL, &receive_particle_counts);
    for (int target_rank = 0; target_rank < num_procs; target_rank++) {
        if (target_rank == rank) { // No need to send to self
            send_particle_counts[target_rank] = 0;
            send_particle_ids[target_rank].clear();
        } else {
            send_particle_counts[target_rank] = send_particle_ids[target_rank].size();
        }
    }

    // Use all_to_all, getting the amount of particles to receive from each rank
    MPI_Alltoall(send_particle_counts, 1, MPI_INT, receive_particle_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // Prepare send and receive buffers
    std::vector<particle_t> send_buffers[num_procs];    // indices = rank to send to
    std::vector<particle_t> receive_buffers[num_procs]; // indices = rank to receive from

    // Fill send buffers, resize receive buffers based on how many particles are being sent from each rank
    for (int rank_i = 0; rank_i < num_procs; ++rank_i) {
        for (int part_id: send_particle_ids[rank_i]) {
            send_buffers[rank_i].push_back(parts[part_id - 1]);
        }
        receive_buffers[rank_i].resize(receive_particle_counts[rank_i]);
    }

    // Use Isend/Irecv for async send/receives on each rank
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> receive_requests;

    int num_sends = 0;
    int num_receives = 0;
    for (int rank_i = 0; rank_i < num_procs; ++rank_i) {
        if (send_particle_counts[rank_i] > 0) {
            MPI_Request send_request;
            MPI_Isend(&send_buffers[rank_i][0], send_buffers[rank_i].size(), PARTICLE, rank_i, 0, MPI_COMM_WORLD, &send_request);
            send_requests.push_back(send_request);
            num_sends++;
        }
        if (receive_particle_counts[rank_i] > 0) {
            MPI_Request receive_request;
            MPI_Irecv(&receive_buffers[rank_i][0], receive_buffers[rank_i].size(), PARTICLE, rank_i, 0, MPI_COMM_WORLD, &receive_request);
            receive_requests.push_back(receive_request);
            num_receives++;
        }
    }

    // Wait for all of the sends and receives to finish before further computation
    MPI_Status send_statuses[num_sends];
    MPI_Status receive_statuses[num_receives];
    
    if (send_requests.size() > 0) {
        MPI_Waitall(num_sends, &send_requests[0], send_statuses);
    }
    if (receive_requests.size() > 0) {
        MPI_Waitall(num_receives, &receive_requests[0], receive_statuses);
    }
    
    // For each received particle:
        // update received particles info to local parts array
        // place into designated bin 
    for (int rank_i = 0; rank_i < num_procs; ++rank_i) {
        for (particle_t part: receive_buffers[rank_i]) {
            parts[part.id - 1] = part;
            int bin_idx = compute_bin_index(&part);
            grid[bin_idx]->particles.push_back(&parts[part.id - 1]);
        }
    }

    MPI_Free_mem(send_particle_counts);
    MPI_Free_mem(receive_particle_counts);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // Vectors for gathering particles from all the ranks
    std::vector<particle_t> send_parts;  
    std::vector<particle_t> receive_parts(num_parts);

    // Add particles from own rank to be sent for gathering
    for (int i = 0; i < own_rank_bins.size(); i += 1) {
        std::vector<particle_t*> target_bin_parts = own_rank_bins[i]->particles;
        for (int j = 0; j < target_bin_parts.size(); j += 1) {
            send_parts.push_back(*target_bin_parts[j]);
        }
    }

    // Use variable gather due to particle count varying for each rank
    // Create arrays for specifying count and offset of where in the array of particles for each rank
    int* receive_counts;
    int* offsets;
    if (rank == 0) {
        receive_counts = new int[num_procs];
        offsets = new int[num_procs];
    }

    // Use gather to populate array for specifying count of particles for each offset
    int send_count = send_parts.size();
    MPI_Gather(&send_count, 1, MPI_INT, receive_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Build up offsets array
    if (rank == 0) {
        offsets[0] = 0;
        for (int i = 1; i < num_procs; i += 1) {
            offsets[i] = offsets[i - 1] + receive_counts[i - 1];
        }
    }

    // Variable gather of particles from ranks
    MPI_Gatherv(&send_parts[0], send_parts.size(), PARTICLE, &receive_parts[0], receive_counts, offsets, PARTICLE, 0, MPI_COMM_WORLD);


    // Populate master rank parts array with sorted array of particles
    if (rank == 0) {
        for (int i = 0; i < num_parts; i += 1) {
            particle_t target_part = receive_parts[i];
            parts[target_part.id - 1] = target_part;
        }
        delete receive_counts;
        delete offsets;
    }

}