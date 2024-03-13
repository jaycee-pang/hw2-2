#include "common.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <set>
#include <vector>

using namespace std;

static int bins_total;
static int bins_per_axis;
static double bin_size;

static int local_num_particles;
static int local_start_idx;
static int local_end_idx;

static int base_particles_per_thread;
static int remainder_particles;

std::vector<std::vector<std::set<int>>> grid;

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs)
{
   
    double minimum_bin_size = cutoff + 0.000001;
    bins_per_axis = floor(size / minimum_bin_size);
    bins_total = bins_per_axis * bins_per_axis;
    bin_size = size / bins_per_axis;

   

    // Initialize the grid
    grid = std::vector<std::vector<std::set<int>>>(bins_per_axis, std::vector<std::set<int>>(bins_per_axis, std::set<int>()));
    for (int i = 0; i < num_parts; i++) {
        int x_idx = floor(parts[i].x / bin_size);
        int y_idx = floor(parts[i].y / bin_size);
        grid[y_idx][x_idx].insert(i);
    }

    // Find particles per thread stuff
    base_particles_per_thread = floor(num_parts / num_procs);
    remainder_particles = num_parts % num_procs;
    if (rank < remainder_particles) {
        local_num_particles = base_particles_per_thread + 1;
        local_start_idx = rank * local_num_particles;
    } else {
        local_num_particles = base_particles_per_thread;
        local_start_idx = rank * local_num_particles + remainder_particles;
    }
    local_end_idx = local_start_idx + local_num_particles;

  
}

void apply_force(particle_t& particle, particle_t& neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size)
{
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs)
{
    // Apply forces to this rank's particles
    for (int i = local_start_idx; i < local_end_idx; i++) {
        // Reset acceleration
        parts[i].ax = parts[i].ay = 0;

        // Find which bin a particle is in
        int y_idx = floor(parts[i].y / bin_size);
        int x_idx = floor(parts[i].x / bin_size);

        // Look at the 3x3 grid of bins around this bin
        for (int j = max(0, y_idx - 1); j < min(bins_per_axis, y_idx + 2); j++) {
            for (int k = max(0, x_idx - 1); k < min(bins_per_axis, x_idx + 2); k++) {
                for (std::set<int>::iterator it1 = grid[j][k].begin(); it1 != grid[j][k].end(); it1++) {
                    apply_force(parts[i], parts[*it1]);
                }
            }
        }
    }

    // Move this rank's particles
    for (int i = local_start_idx; i < local_end_idx; i++) {
        int old_x_idx = floor(parts[i].x / bin_size);
        int old_y_idx = floor(parts[i].y / bin_size);
        move(parts[i], size);
        int new_x_idx = floor(parts[i].x / bin_size);
        int new_y_idx = floor(parts[i].y / bin_size);
        if (new_x_idx != old_x_idx || new_y_idx != old_y_idx) {
            grid[old_y_idx][old_x_idx].erase(i);
            grid[new_y_idx][new_x_idx].insert(i);
        }
    }

    // Calculate how many particles we should receive from each process, and their start/end indexes
    std::vector<int> recvcounts(num_procs, 0);
    std::vector<int> displs(num_procs, 0);
    int sum_displs = 0;
    for (int i = 0; i < num_procs; ++i) {
        recvcounts[i] = (i < remainder_particles) ? base_particles_per_thread + 1 : base_particles_per_thread;
        displs[i] = sum_displs;
        sum_displs += recvcounts[i];
    }

    // Gather all particles from all processes
    particle_t* all_particles = new particle_t[num_parts];
    MPI_Allgatherv(parts + local_start_idx, local_num_particles, PARTICLE, all_particles, recvcounts.data(), displs.data(), PARTICLE, MPI_COMM_WORLD);

    // Check which particles have moved to a different bin
    for (int i = 0; i < num_parts; i++) {
        int old_x_idx = floor(parts[i].x / bin_size);
        int old_y_idx = floor(parts[i].y / bin_size);
        int new_x_idx = floor(all_particles[i].x / bin_size);
        int new_y_idx = floor(all_particles[i].y / bin_size);
        if (new_x_idx != old_x_idx || new_y_idx != old_y_idx) {
            grid[old_y_idx][old_x_idx].erase(i);
            grid[new_y_idx][new_x_idx].insert(i);
        }
    }

    // Rewrite particle positions and clear memory
    memcpy(parts, all_particles, num_parts * sizeof(particle_t));
    delete[] all_particles;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs)
{
    // pass 
    return;
}

