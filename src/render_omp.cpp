// OpenMP parallel render — one thread per pixel block
#include "render_omp.h"
#include <omp.h>
#include <random>
#include <cmath>

using namespace rt;

// ── thread-local RNG: each OpenMP thread gets its own independent state ──────
// Without this, threads racing on a shared rand() produce data races and
// garbage pixels. thread_local guarantees one copy per thread.
static thread_local std::mt19937 tl_rng(std::random_device{}());
static thread_local std::uniform_real_distribution<float> tl_dist(0.0f, 1.0f);
static float rng_f() { return tl_dist(tl_rng); }

static vec3 random_unit_vector() {
    while (true) {
        vec3  p     = { rng_f()*2-1, rng_f()*2-1, rng_f()*2-1 };
        float lensq = p.length_sq();
        if (lensq > 1e-8f && lensq <= 1.0f)
            return p / sqrtf(lensq);
    }
}

static vec3 random_in_unit_disk() {
    while (true) {
        vec3 p = { rng_f()*2-1, rng_f()*2-1, 0.0f };
        if (p.length_sq() < 1.0f) return p;
    }
}

// ── scatter ───────────────────────────────────────────────────────────────────
static bool scatter(const RtMaterial& mat, const RtRay& r_in, const HitRecord& rec,
                    vec3& attenuation, RtRay& scattered)
{
    if (mat.type == MatType::LAMBERTIAN) {
        vec3 dir = rec.normal + random_unit_vector();
        if (dir.near_zero()) dir = rec.normal;
        scattered   = { rec.p, dir };
        attenuation = mat.albedo;
        return true;
    }
    if (mat.type == MatType::METAL) {
        vec3 reflected = r_in.dir - rec.normal * 2.0f * r_in.dir.dot(rec.normal);
        reflected = reflected.normalize() + random_unit_vector() * mat.fuzz;
        scattered   = { rec.p, reflected };
        attenuation = mat.albedo;
        return reflected.dot(rec.normal) > 0.0f;
    }
    // DIELECTRIC
    attenuation     = {1.0f, 1.0f, 1.0f};
    float ri        = rec.front_face ? (1.0f / mat.ref_idx) : mat.ref_idx;
    vec3  unit_dir  = r_in.dir.normalize();
    float cos_theta = fminf(-unit_dir.dot(rec.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    float r0 = (1.0f - ri) / (1.0f + ri);
    r0 = r0 * r0;
    float reflectance = r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);

    vec3 dir;
    if (ri * sin_theta > 1.0f || reflectance > rng_f()) {
        dir = unit_dir - rec.normal * 2.0f * unit_dir.dot(rec.normal);
    } else {
        vec3 r_perp     = (unit_dir + rec.normal * cos_theta) * ri;
        vec3 r_parallel = rec.normal * (-sqrtf(fabsf(1.0f - r_perp.length_sq())));
        dir = r_perp + r_parallel;
    }
    scattered = { rec.p, dir };
    return true;
}

// ── ray_color (iterative) ────────────────────────────────────────────────────
static vec3 ray_color(RtRay r, const Scene& scene, int max_depth) {
    vec3 attenuation = {1.0f, 1.0f, 1.0f};
    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;
        float     t_max   = RT_INF;
        int       hit_idx = -1;

        for (int i = 0; i < scene.n_spheres; i++) {
            HitRecord tmp;
            if (hit_sphere(scene.spheres[i], r, 0.001f, t_max, tmp)) {
                t_max = tmp.t; rec = tmp; hit_idx = i;
            }
        }

        if (hit_idx < 0) {
            float t   = 0.5f * (r.dir.normalize().y + 1.0f);
            vec3  sky = vec3{1,1,1} * (1-t) + vec3{0.5f,0.7f,1.0f} * t;
            return attenuation * sky;
        }

        vec3  new_atten;
        RtRay scattered;
        if (!scatter(scene.mats[rec.mat_id], r, rec, new_atten, scattered))
            return {0,0,0};

        attenuation = attenuation * new_atten;
        r = scattered;
    }
    return {0,0,0};
}

// ─────────────────────────────────────────────────────────────────────────────
//  The actual parallel render
//
//  parallel for     → split iterations across all available threads
//  collapse(2)      → flatten the j/i loops into one pool of W*H pixels
//                     so any thread can grab any pixel
//  schedule(dynamic, 4) → idle threads grab 4 pixels at a time on demand;
//                         handles uneven per-pixel cost (centre pixels hit
//                         more spheres than corner pixels)
// ─────────────────────────────────────────────────────────────────────────────
void render_omp(unsigned char* fb, int W, int H,
                const RtCamera& cam, const Scene& scene,
                int samples, int max_depth)
{
    #pragma omp parallel for schedule(dynamic, 4) collapse(2)
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            vec3 col = {0,0,0};
            for (int s = 0; s < samples; s++) {
                RtRay r = cam.get_ray(i + rng_f(), j + rng_f());
                if (cam.defocus_angle > 0.0f) {
                    vec3   disk   = random_in_unit_disk();
                    point3 origin = cam.origin
                                  + cam.defocus_disk_u * disk.x
                                  + cam.defocus_disk_v * disk.y;
                    r = { origin, r.at(1.0f) - origin };
                }
                col += ray_color(r, scene, max_depth);
            }
            col = col / (float)samples;

            col.x = sqrtf(fmaxf(col.x, 0.0f));
            col.y = sqrtf(fmaxf(col.y, 0.0f));
            col.z = sqrtf(fmaxf(col.z, 0.0f));

            int idx     = (j * W + i) * 4;
            fb[idx + 0] = (unsigned char)(255.99f * fminf(col.x, 1.0f));
            fb[idx + 1] = (unsigned char)(255.99f * fminf(col.y, 1.0f));
            fb[idx + 2] = (unsigned char)(255.99f * fminf(col.z, 1.0f));
            fb[idx + 3] = 255;
        }
    }
}
