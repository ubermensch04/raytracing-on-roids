// CUDA render kernel
// Key concept: instead of a loop over pixels on one CPU thread,
// the GPU launches thousands of threads simultaneously — one per pixel.
// Each thread runs render_kernel() independently.

#include "render_cuda.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>   // GPU random number generator
#include <cstdio>

using namespace rt;

// Crash immediately with a useful message if any CUDA call fails.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
//  Device helper functions
//
//  __device__ means "runs on the GPU, callable only from GPU code."
//  The logic is identical to render_seq.cpp — only the RNG is different.
//  curandState holds per-thread random state (like thread_local in OpenMP).
// ─────────────────────────────────────────────────────────────────────────────

__device__ static float drng(curandState& s) {
    return curand_uniform(&s);   // returns float in (0, 1]
}

__device__ static vec3 d_random_unit_vector(curandState& s) {
    while (true) {
        vec3  p     = { drng(s)*2-1, drng(s)*2-1, drng(s)*2-1 };
        float lensq = p.length_sq();
        if (lensq > 1e-8f && lensq <= 1.0f)
            return p / sqrtf(lensq);
    }
}

__device__ static vec3 d_random_in_unit_disk(curandState& s) {
    while (true) {
        vec3 p = { drng(s)*2-1, drng(s)*2-1, 0.0f };
        if (p.length_sq() < 1.0f) return p;
    }
}

// ── scatter: same physics as CPU version, __device__ so GPU can call it ──────
__device__ static bool d_scatter(const RtMaterial& mat, const RtRay& r_in,
                                  const HitRecord& rec, vec3& attenuation,
                                  RtRay& scattered, curandState& s)
{
    if (mat.type == MatType::LAMBERTIAN) {
        vec3 dir = rec.normal + d_random_unit_vector(s);
        if (dir.near_zero()) dir = rec.normal;
        scattered   = { rec.p, dir };
        attenuation = mat.albedo;
        return true;
    }
    if (mat.type == MatType::METAL) {
        vec3 reflected = r_in.dir - rec.normal * 2.0f * r_in.dir.dot(rec.normal);
        reflected   = reflected.normalize() + d_random_unit_vector(s) * mat.fuzz;
        scattered   = { rec.p, reflected };
        attenuation = mat.albedo;
        return reflected.dot(rec.normal) > 0.0f;
    }
    // DIELECTRIC
    attenuation    = {1.0f, 1.0f, 1.0f};
    float ri       = rec.front_face ? (1.0f / mat.ref_idx) : mat.ref_idx;
    vec3  unit_dir = r_in.dir.normalize();
    float cos_t    = fminf(-unit_dir.dot(rec.normal), 1.0f);
    float sin_t    = sqrtf(1.0f - cos_t * cos_t);
    float r0       = (1.0f - ri) / (1.0f + ri); r0 = r0 * r0;
    float refl     = r0 + (1.0f - r0) * powf(1.0f - cos_t, 5.0f);
    vec3 dir;
    if (ri * sin_t > 1.0f || refl > drng(s)) {
        dir = unit_dir - rec.normal * 2.0f * unit_dir.dot(rec.normal);
    } else {
        vec3 r_perp     = (unit_dir + rec.normal * cos_t) * ri;
        vec3 r_parallel = rec.normal * (-sqrtf(fabsf(1.0f - r_perp.length_sq())));
        dir = r_perp + r_parallel;
    }
    scattered = { rec.p, dir };
    return true;
}

// On CPU we recurse (call ray_color inside ray_color). CUDA has a stack size
// limit per thread that makes deep recursion crash. The iterative version
// accumulates the attenuation product in a loop instead.
__device__ static vec3 d_ray_color(RtRay r,
                                    const Sphere* spheres, int n_spheres,
                                    const RtMaterial* mats,
                                    int max_depth, curandState& s)
{
    vec3 attenuation = {1, 1, 1};
    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;
        float     t_max   = RT_INF;
        int       hit_idx = -1;

        for (int i = 0; i < n_spheres; i++) {
            HitRecord tmp;
            if (hit_sphere(spheres[i], r, 0.001f, t_max, tmp)) {
                t_max = tmp.t; rec = tmp; hit_idx = i;
            }
        }

        if (hit_idx < 0) {
            float t   = 0.5f * (r.dir.normalize().y + 1.0f);
            vec3  sky = vec3{1,1,1}*(1-t) + vec3{0.5f,0.7f,1.0f}*t;
            return attenuation * sky;
        }

        vec3 new_atten; RtRay scattered;
        if (!d_scatter(mats[rec.mat_id], r, rec, new_atten, scattered, s))
            return {0, 0, 0};

        attenuation = attenuation * new_atten;
        r = scattered;
    }
    return {0, 0, 0};
}

//  Kernel 1: initialise one curandState per pixel
//
//  GPU threads are identified by blockIdx and threadIdx.
//  A "block" is a group of threads (here 16×16 = 256).
//  We map 2D block/thread indices to a pixel (x, y).
__global__ static void init_rng_kernel(curandState* states, int W, int H,
                                        unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;  // threads beyond image edge do nothing
    // Each thread gets a unique sequence by using its pixel index as sequence number.
    curand_init(seed, y * W + x, 0, &states[y * W + x]);
}

//  Kernel 2: render — one thread computes one pixel
//
//  __global__ means "entry point callable from CPU, runs on GPU."
//  The CPU calls this with <<<blocks, threads>>> to launch all threads at once.
__global__ static void render_kernel(unsigned char* fb, int W, int H,
                                      RtCamera cam,
                                      const Sphere* spheres, int n_spheres,
                                      const RtMaterial* mats,
                                      int samples, int max_depth,
                                      curandState* rng_states)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int         idx       = y * W + x;
    curandState local_rng = rng_states[idx]; // copy to registers — faster

    vec3 col = {0, 0, 0};
    for (int s = 0; s < samples; s++) {
        RtRay r = cam.get_ray(x + drng(local_rng), y + drng(local_rng));
        if (cam.defocus_angle > 0.0f) {
            vec3   disk   = d_random_in_unit_disk(local_rng);
            point3 origin = cam.origin
                          + cam.defocus_disk_u * disk.x
                          + cam.defocus_disk_v * disk.y;
            r = { origin, r.at(1.0f) - origin };
        }
        col += d_ray_color(r, spheres, n_spheres, mats, max_depth, local_rng);
    }

    col = col / (float)samples;
    col.x = sqrtf(fmaxf(col.x, 0.0f));  // gamma correction
    col.y = sqrtf(fmaxf(col.y, 0.0f));
    col.z = sqrtf(fmaxf(col.z, 0.0f));

    fb[idx*4+0] = (unsigned char)(255.99f * fminf(col.x, 1.0f));
    fb[idx*4+1] = (unsigned char)(255.99f * fminf(col.y, 1.0f));
    fb[idx*4+2] = (unsigned char)(255.99f * fminf(col.z, 1.0f));
    fb[idx*4+3] = 255;

    rng_states[idx] = local_rng; // write RNG state back to device memory
}

void render_cuda(unsigned char* fb, int W, int H,
                 const RtCamera& cam, const Scene& scene,
                 int samples, int max_depth)
{
    // Step 1: upload scene geometry to GPU memory
    Sphere*     d_spheres;
    RtMaterial* d_mats;
    CUDA_CHECK(cudaMalloc(&d_spheres, scene.n_spheres * sizeof(Sphere)));
    CUDA_CHECK(cudaMalloc(&d_mats,    scene.n_mats    * sizeof(RtMaterial)));
    CUDA_CHECK(cudaMemcpy(d_spheres, scene.spheres,
                          scene.n_spheres * sizeof(Sphere),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mats,    scene.mats,
                          scene.n_mats    * sizeof(RtMaterial), cudaMemcpyHostToDevice));

    // Step 2: allocate GPU framebuffer (GPU writes here, we copy back later)
    unsigned char* d_fb;
    CUDA_CHECK(cudaMalloc(&d_fb, W * H * 4));

    // Step 3: allocate per-pixel RNG states on GPU
    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_rng, (size_t)W * H * sizeof(curandState)));

    // Step 4: configure thread layout
    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16);  

    // Step 5: initialise RNG (must finish before render kernel reads it)
    init_rng_kernel<<<blocks, threads>>>(d_rng, W, H, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());  // wait for GPU to finish

    // Step 6: render — all pixels in parallel
    render_kernel<<<blocks, threads>>>(d_fb, W, H, cam,
                                       d_spheres, scene.n_spheres, d_mats,
                                       samples, max_depth, d_rng);
    CUDA_CHECK(cudaDeviceSynchronize());  // wait before reading result

    // Step 7: copy rendered pixels back to CPU so Raylib can display them
    CUDA_CHECK(cudaMemcpy(fb, d_fb, W * H * 4, cudaMemcpyDeviceToHost));

    // Step 8: free all GPU memory
    cudaFree(d_spheres);
    cudaFree(d_mats);
    cudaFree(d_fb);
    cudaFree(d_rng);
}
