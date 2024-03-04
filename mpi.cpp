#include "common.h"
#include <mpi.h>
#include <cmath> 
#include <cstddef> 
#include <cstdlib>
#include <algorithm>
MPI_Datatype MPI_PARTICLE_T;
// MPI_Type_contiguous(sizeof(particle_t), MPI_BYTE, &MPI_PARTICLE_T);
// MPI_Type_commit(&MPI_PARTICLE_T);
void initialize_MPI_datatypes() {
    MPI_Type_contiguous(sizeof(particle_t), MPI_BYTE, &MPI_PARTICLE_T);
    MPI_Type_commit(&MPI_PARTICLE_T);
}
// int compare_id(const void* particle_a, const void* particle_b) {
//     const particle_t* pa = static_cast<const particle_t*>(particle_a);
//     const particle_t* pb = static_cast<const particle_t*>(particle_b);
//     return (pa->id < pb->id) ? -1 : ((pa->id > pb->id) ? 1 : 0);
// }
const int MAX_PARTS_PER_BIN = 4;



int num_bins_one_side;
int num_bins;

typedef struct Bin {
    int num_particles;
    particle_t* members[MAX_PARTS_PER_BIN];
    int neighbors[9];
} Bin;

Bin* grid;

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

// computes which bin a particle should go into
static inline int compute_bin_index(particle_t* part) {
    int bin_x = static_cast<int> (part->x / cutoff);
    int bin_y = static_cast<int> (part->y / cutoff);
    return bin_x + bin_y * num_bins_one_side;
}

// computes the total force on one particle using its own bin and neighboring bins
static inline void total_force_one_particle(Bin* own_bin, particle_t* part) {
    particle_t* other_part;
    int* neighbor_indices = own_bin->neighbors;
    for (int k = 0; k < 9; k++) {
        int i = neighbor_indices[k];
        if (i <= 0) { break; }

        Bin* bin = &grid[i-1];

        if (bin->num_particles > 0) {
            for (int j = 0; j < MAX_PARTS_PER_BIN; j++) {
                other_part = bin->members[j];
                if (other_part == NULL) continue;
                if (part == other_part) continue; 
                apply_force(*part, *other_part);
            }
        }
    }
}

// void init_simulation(particle_t* parts, int num_parts, double size) {
//     // You can use this space to initialize static, global data objects
//     // that you may need. This function will be called once before the
//     // algorithm begins. Do not do any particle simulation here

//     // instantiate the bins

// }


// void simulate_one_step(particle_t* parts, int num_parts, double size) {

// }

// Put any static global variables here that you will use throughout the simulation.

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    // particle binning
    num_bins_one_side = static_cast<int>(ceil(size / cutoff));
    num_bins = num_bins_one_side * num_bins_one_side;
    grid = (Bin*)calloc(num_bins, sizeof(Bin));

    // assign particles to their initial bins
    for (int i = 0; i < num_parts; i++) {
        particle_t* part = &parts[i];

        int bin_idx = compute_bin_index(part);
        Bin* bin = &grid[bin_idx];

        for (int part_idx = 0; part_idx < MAX_PARTS_PER_BIN; part_idx++) {
            if (bin->members[part_idx] == NULL) {
                bin->members[part_idx] = part;
                bin->num_particles++;
                break;
            }
        }
    }

    // figure out neighbors for each bin
    for (int i = 0; i < num_bins; ++i) {
        Bin* target_bin = &grid[i];
        int count = 0;
        for (int dx = -1; dx <= 1; ++dx) { 
            for (int dy = -1; dy <= 1; ++dy) { 
                int neighbor_idx = i + dx + dy * num_bins_one_side; 
                if (neighbor_idx >= 0 && neighbor_idx < num_bins) {
                    target_bin->neighbors[count] = neighbor_idx+1;
                    count++;
                }
            }
        }
    }
    // MPI_Datatype MPI_PARTICLE_T;
    MPI_Type_contiguous(sizeof(particle_t), MPI_BYTE, &MPI_PARTICLE_T);
    MPI_Type_commit(&MPI_PARTICLE_T);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    // rebin particles after they move
    for (int bin_idx = 0; bin_idx < num_bins; bin_idx++) {
        Bin* bin = &grid[bin_idx];
        if (bin->num_particles == 0) continue;
        for (int part_idx = 0; part_idx < MAX_PARTS_PER_BIN; part_idx++) {
            particle_t* part = bin->members[part_idx];
            if (part == NULL) continue;
            int new_idx = compute_bin_index(part);
            
            // don't do anything if bin doesn't change
            if (bin_idx == new_idx) continue;

            // move particle to new bin and erase it from new bin
            Bin* new_bin = &grid[new_idx];
            for (int new_bin_part_idx = 0; new_bin_part_idx < MAX_PARTS_PER_BIN; new_bin_part_idx++) {
                if (new_bin->members[new_bin_part_idx] == NULL) {

                    new_bin->num_particles++;
                    new_bin->members[new_bin_part_idx] = bin->members[part_idx];

                    bin->num_particles--;
                    bin->members[part_idx] = NULL;
                    break;
                }
            }
        }
    }

    // compute forces on each particle, faster to do by bin
    for(int bin_idx = 0; bin_idx < num_bins; bin_idx++) {
        Bin* own_bin = &grid[bin_idx];
        if (0 != own_bin->num_particles) {
            for (int part_idx = 0; part_idx < MAX_PARTS_PER_BIN; part_idx++) {
                particle_t* part = own_bin->members[part_idx];
                if (part == NULL) continue;
                part->ax = part->ay = 0;
                total_force_one_particle(own_bin, part);
            }
        }
    }

    // move particles after calculating force
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // allocate memory to store all particles on the master process
    particle_t* all_parts = NULL;
    // row format 
    if (rank == 0) { 
        all_parts = (particle_t *)malloc(num_parts * num_procs * sizeof(particle_t));
    }

    MPI_Gather(parts, num_parts, MPI_PARTICLE_T, all_parts, num_parts, MPI_PARTICLE_T, 0, MPI_COMM_WORLD);
    if (rank == 0) {

        // Sort the gathered particles by particle ID
        std::sort(all_parts, all_parts + num_parts * num_procs,
                  [](const particle_t& a, const particle_t& b) {
                      return a.id < b.id;
                  });

        // copy back to original
        std::copy(all_parts, all_parts + num_parts * num_procs, parts);

        // Free the memory allocated for all_parts
        free(all_parts);
    // if (rank == 0) {
    //     qsort(all_parts, num_parts * num_procs, sizeof(particle_t), compare_id);

    //     for (int i = 0; i < num_parts * num_procs; i++) {
    //         parts[i] = all_parts[i];
    //     }

    //     free(all_parts);
    }
    
}