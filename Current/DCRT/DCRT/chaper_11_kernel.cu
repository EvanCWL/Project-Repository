#include "kernels.cuh"

#include <float.h>
#include "device_launch_parameters.h"
#include "texturegpu.cuh"
#include "hitable_list.cuh"
#include "camera.cuh"
#include "math.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "cuda_utils.cuh"
#include "managed_ptr.cuh"
#include "world.cuh"

__device__ Vec3 color_11(const Ray& r, Hitable** hitable_objects,
    curandState* local_rand_state) {
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < 50; ++i) {
        HitRecord rec;
        if ((*hitable_objects)->hit(cur_ray, 0.001f, 10.0f, rec)) {
            Ray scattered;
            Vec3 attenuation;

            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,
                local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return Vec3(0.0f, 0.0f, 0.0f);
            }

        }
        else {
            Vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);

            Vec3 c = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);

            return cur_attenuation * c;
        }
    }

    return Vec3(0.0f, 0.0f, 0.0f);
}

__global__ void chapter_11_kernel(TextureGPU* tex, Camera camera,
    Hitable** hitable_objects,
    curandState* rand_state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    size_t w = tex->get_width();
    size_t h = tex->get_height();

    if ((x >= w || (y >= h)))
        return;

    int pixel_index = y * w + x;
    curandState* local_rand_state = &rand_state[pixel_index];

    Vec3 col(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < camera.get_ns(); ++s) {
        float u = float(x + curand_uniform(local_rand_state)) / float(w);
        float v = float(h - y + curand_uniform(local_rand_state)) / float(h);

        Ray ray = camera.get_ray(u, v, local_rand_state);

        col += color_11(ray, hitable_objects, local_rand_state);
    }

    rand_state[pixel_index] = *local_rand_state;

    col /= float(camera.get_ns());
    Uint8 r = sqrt(col.r()) * 255.99f;
    Uint8 g = sqrt(col.g()) * 255.99f;
    Uint8 b = sqrt(col.b()) * 255.99f;

    tex->set_rgb(x, y, r, g, b);
}
