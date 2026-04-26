//Contains ALL shared classes: vec3, ray, camera, sphere, material, hitrecord
#pragma once
#include <cmath>
#include <cstdlib>

// __host__ __device__ are built-in nvcc keywords — no header needed.
// When a plain C++ compiler processes this header, define them away.
#ifndef __CUDACC__
  #define __host__
  #define __device__
#endif

// Everything lives in namespace rt to avoid clashing with Raylib's
// Ray, Material, Camera, and PI definitions.
namespace rt {

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
constexpr float RT_PI  = 3.14159265358979323846f;
constexpr float RT_INF = 1e30f;

inline float degrees_to_radians(float d) { return d * RT_PI / 180.0f; }

// ─────────────────────────────────────────────
//  vec3
// ─────────────────────────────────────────────
class vec3 {
public:
    float x, y, z;

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ vec3 operator+(const vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
    __host__ __device__ vec3 operator-(const vec3& b) const { return {x-b.x, y-b.y, z-b.z}; }
    __host__ __device__ vec3 operator-()              const { return {-x, -y, -z}; }
    __host__ __device__ vec3 operator*(float t)       const { return {x*t,   y*t,   z*t};   }
    __host__ __device__ vec3 operator*(const vec3& b) const { return {x*b.x, y*b.y, z*b.z}; }
    __host__ __device__ vec3 operator/(float t)       const { return *this * (1.0f/t); }

    __host__ __device__ vec3& operator+=(const vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }

    __host__ __device__ float dot(const vec3& b)  const { return x*b.x + y*b.y + z*b.z; }
    __host__ __device__ float length_sq()         const { return dot(*this); }
    __host__ __device__ float length()            const { return sqrtf(length_sq()); }
    __host__ __device__ vec3  normalize()         const { return *this / length(); }

    __host__ __device__ vec3 cross(const vec3& b) const {
        return { y*b.z - z*b.y,
                 z*b.x - x*b.z,
                 x*b.y - y*b.x };
    }

    __host__ __device__ bool near_zero() const {
        const float s = 1e-8f;
        return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
    }

    static vec3 random() {
        auto r = []{ return (float)rand() / ((float)RAND_MAX + 1.0f); };
        return { r(), r(), r() };
    }
    static vec3 random(float mn, float mx) {
        auto r = [&]{ return mn + (mx-mn) * ((float)rand() / ((float)RAND_MAX + 1.0f)); };
        return { r(), r(), r() };
    }
};

using point3 = vec3;

inline __host__ __device__ vec3 operator*(float t, const vec3& v) { return v * t; }

// ─────────────────────────────────────────────
//  RtRay  (named RtRay to avoid clash with raylib::Ray)
// ─────────────────────────────────────────────
struct RtRay {
    point3 origin;
    vec3   dir;
    __host__ __device__ point3 at(float t) const { return origin + dir * t; }
};

// ─────────────────────────────────────────────
//  RtMaterial  (flat struct — CUDA safe, no vtable)
// ─────────────────────────────────────────────
enum class MatType { LAMBERTIAN, METAL, DIELECTRIC };

struct RtMaterial {
    MatType type    = MatType::LAMBERTIAN;
    vec3    albedo  = {0.5f, 0.5f, 0.5f};
    float   fuzz    = 0.0f;
    float   ref_idx = 1.5f;
};

// ─────────────────────────────────────────────
//  HitRecord
// ─────────────────────────────────────────────
struct HitRecord {
    float  t;
    point3 p;
    vec3   normal;
    int    mat_id;
    bool   front_face;

    __host__ __device__ void set_face_normal(const RtRay& r, const vec3& outward_normal) {
        front_face = r.dir.dot(outward_normal) < 0.0f;
        normal     = front_face ? outward_normal : -outward_normal;
    }
};

// ─────────────────────────────────────────────
//  Sphere
// ─────────────────────────────────────────────
struct Sphere {
    point3 center;
    float  radius;
    int    mat_id;
};

__host__ __device__
inline bool hit_sphere(const Sphere& s, const RtRay& r,
                       float t_min, float t_max, HitRecord& rec)
{
    vec3  oc   = s.center - r.origin;
    float a    = r.dir.dot(r.dir);
    float h    = r.dir.dot(oc);
    float c    = oc.dot(oc) - s.radius * s.radius;
    float disc = h*h - a*c;
    if (disc < 0.0f) return false;

    float sqrtd = sqrtf(disc);
    float root  = (h - sqrtd) / a;
    if (root <= t_min || root >= t_max) {
        root = (h + sqrtd) / a;
        if (root <= t_min || root >= t_max) return false;
    }

    rec.t      = root;
    rec.p      = r.at(root);
    rec.mat_id = s.mat_id;
    vec3 outward_normal = (rec.p - s.center) / s.radius;
    rec.set_face_normal(r, outward_normal);
    return true;
}

// ─────────────────────────────────────────────
//  Scene
// ─────────────────────────────────────────────
constexpr int MAX_SPHERES   = 512;
constexpr int MAX_MATERIALS = 512;

struct Scene {
    Sphere     spheres[MAX_SPHERES];
    RtMaterial mats[MAX_MATERIALS];
    int        n_spheres = 0;
    int        n_mats    = 0;

    int add_material(const RtMaterial& m) { mats[n_mats] = m; return n_mats++; }
    void add_sphere(point3 center, float radius, int mat_id) {
        spheres[n_spheres++] = { center, radius, mat_id };
    }
};

// ─────────────────────────────────────────────
//  RtCamera  (named RtCamera to avoid clash with raylib::Camera)
// ─────────────────────────────────────────────
struct RtCamera {
    point3 lookfrom      = {13, 2, 3};
    point3 lookat        = {0,  0, 0};
    vec3   vup           = {0,  1, 0};
    float  vfov          = 20.0f;
    float  defocus_angle = 0.6f;
    float  focus_dist    = 10.0f;

    // derived — filled by init()
    point3 origin;
    point3 pixel00_loc;
    vec3   pixel_delta_u;
    vec3   pixel_delta_v;
    vec3   defocus_disk_u;
    vec3   defocus_disk_v;
    vec3   u, v, w;

    void init(int image_width, int image_height) {
        origin = lookfrom;

        float theta = degrees_to_radians(vfov);
        float h     = tanf(theta / 2.0f);
        float vp_h  = 2.0f * h * focus_dist;
        float vp_w  = vp_h * ((float)image_width / (float)image_height);

        w = (lookfrom - lookat).normalize();
        u = vup.cross(w).normalize();
        v = w.cross(u);

        vec3 vp_u = u * vp_w;
        vec3 vp_v = (-v) * vp_h;

        pixel_delta_u = vp_u / (float)image_width;
        pixel_delta_v = vp_v / (float)image_height;

        point3 vp_upper_left = origin - w * focus_dist - vp_u * 0.5f - vp_v * 0.5f;
        pixel00_loc = vp_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5f;

        float dr   = focus_dist * tanf(degrees_to_radians(defocus_angle / 2.0f));
        defocus_disk_u = u * dr;
        defocus_disk_v = v * dr;
    }

    __host__ __device__
    RtRay get_ray(float px, float py) const {
        point3 pixel_sample = pixel00_loc
                            + pixel_delta_u * px
                            + pixel_delta_v * py;
        return RtRay{ origin, pixel_sample - origin };
    }
};

// ─────────────────────────────────────────────
//  Scene builder
// ─────────────────────────────────────────────
inline Scene build_final_scene() {
    Scene scene;

    int gnd = scene.add_material({ MatType::LAMBERTIAN, {0.5f,0.5f,0.5f} });
    scene.add_sphere({0,-1000,0}, 1000.0f, gnd);

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose = (float)rand() / (float)RAND_MAX;
            point3 center = {
                a + 0.9f * ((float)rand()/(float)RAND_MAX),
                0.2f,
                b + 0.9f * ((float)rand()/(float)RAND_MAX)
            };
            if ((center - point3{4.0f,0.2f,0.0f}).length() <= 0.9f) continue;

            RtMaterial m;
            if (choose < 0.8f) {
                m.type   = MatType::LAMBERTIAN;
                m.albedo = vec3::random() * vec3::random();
            } else if (choose < 0.95f) {
                m.type   = MatType::METAL;
                m.albedo = vec3::random(0.5f, 1.0f);
                m.fuzz   = ((float)rand()/(float)RAND_MAX) * 0.5f;
            } else {
                m.type    = MatType::DIELECTRIC;
                m.ref_idx = 1.5f;
            }
            scene.add_sphere(center, 0.2f, scene.add_material(m));
        }
    }

    scene.add_sphere({0,1,0},  1.0f, scene.add_material({MatType::DIELECTRIC, {}, 0.0f, 1.5f}));
    scene.add_sphere({-4,1,0}, 1.0f, scene.add_material({MatType::LAMBERTIAN, {0.4f,0.2f,0.1f}}));
    scene.add_sphere({4,1,0},  1.0f, scene.add_material({MatType::METAL, {0.7f,0.6f,0.5f}, 0.0f}));

    return scene;
}

} // namespace rt
