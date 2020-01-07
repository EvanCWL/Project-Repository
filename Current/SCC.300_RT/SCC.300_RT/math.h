#pragma once

#include <curand_kernel.h>

#include "vec3.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

namespace Math {
    __device__ vec3 reflect(const vec3& v, const vec3& n);

    __device__ vec3 random_in_unit_sphere(curandState* local_rand_state);

    __device__ vec3 random_in_unit_disk(curandState* local_rand_state);

    __device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt,
        vec3& refracted);

    __device__ float schlick(float cosine, float ref_idx);

	__device__ float schlick(float cosine, float ref_idx) {
		float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
	}

	__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
		vec3 uv = unit_vector(v);
		float dt = dot(uv, n);
		float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
		if (discriminant > 0) {
			refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
			return true;
		}
		else
			return false;
	}

	__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
		vec3 p;
		do {
			p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
		} while (p.squared_length() >= 1.0f);
		return p;
	}

	__device__ vec3 reflect(const vec3& v, const vec3& n) {
		return v - 2.0f * dot(v, n) * n;
	}

	__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
		vec3 p;
		do {
			p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
		} while (dot(p, p) >= 1.0f);
		return p;
	}

};