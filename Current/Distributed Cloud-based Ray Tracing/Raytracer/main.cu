#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "device_launch_parameters.h"
#include <texture_fetch_functions.h>
#include <SDL.h>
#include <Windows.h>
#include <gl/GL.h>

using namespace std;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.1f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int width, int height, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
			new lambertian(vec3(0.5, 0.5, 0.5)));
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = RND;
				vec3 center(a + RND, 0.2, b + RND);
				if (choose_mat < 0.8f) {
					d_list[i++] = new sphere(center, 0.2,
						new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2,
						new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
		*rand_state = local_rand_state;
		*d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.1;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			30.0,
			float(width) / float(height),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete* d_world;
	delete* d_camera;
}

void saveImage(int width, int height, vec3* fb, string output) {
	ofstream img(output);
	img << "P3\n" << width << " " << height << "\n255\n";
	for (int j = height - 1; j >= 0; j--) {
		for (int i = 0; i < width; i++) {
			size_t pixel_index = j * width + i;
			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());
			img << ir << " " << ig << " " << ib << "\n";
		}
	}
}

int main(int argc, char* argv[]) {
	int width = 1920;
	int height = 1080;
	int ns = 10;
	int tx = 22;
	int ty = 22;
	bool running = true, updateDisplay = true;
	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Texture* texture;

	std::cerr << "Rendering a " << width << "x" << height << " image with " << ns << " samples per pixel ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";


	int num_pixels = width * height;
	size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	// allocate random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState* d_rand_state2;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// make our world of hitables & the camera
	hitable** d_list;
	int num_hitables = 22 * 22 + 1 + 3;
	checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
	camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
	create_world << <1, 1 >> > (d_list, d_world, d_camera, width, height, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Save image
	//saveImage(width, height, fb, "output.ppm");

	// Output FB as Image

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cout << "[Error] Failed to initialise SDL2";
		return 1;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	window = SDL_CreateWindow("Ray Tracing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
	if (window == NULL) {
		std::cout << SDL_GetError();
		return 1;
	}

	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	if (renderer == NULL) {
		std::cout << SDL_GetError();
		return 1;
	}

	texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);
	if (texture == NULL) {
		std::cout << SDL_GetError();
		return 1;
	}

	/*
	SDL_GLContext context = SDL_GL_CreateContext(window);
	if (context == NULL) {
		std::cout << SDL_GetError();
		return 1;
	}


	for (int j = height - 1; j >= 0; j--) {
		for (int i = 0; i < width; i++) {
			size_t pixel_index = j * width + i;
			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());
			SDL_SetRenderDrawColor(renderer, ir, ig, ib, 255);
			SDL_Rect rectangle;
			rectangle.x = i;
			rectangle.y = j;
			rectangle.w = 50;
			rectangle.h = 50;
			SDL_RenderFillRect(renderer, &rectangle);
			SDL_RenderPresent(renderer);
		}
	}
	*/

	//Render Scene
	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (width, height, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render << <blocks, threads >> > (fb, width, height, ns, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	int j = 0;
	int i = 0;
	while (running) {
		// Poll for user input events
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			switch (event.type) {
			case SDL_QUIT: {
				running = false;
			}
						 break;
			case SDL_WINDOWEVENT: {
				if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
					width = (uint32_t)event.window.data1;
					height = (uint32_t)event.window.data2;
				}
			}
								break;
								// ... Handle other events here. **IMPLEMENTATION OF MOUSE DETECTIONS**
			}
		}

		size_t pixel_index = j * width + i;
		int ir = int(255.99 * fb[pixel_index].r());
		int ig = int(255.99 * fb[pixel_index].g());
		int ib = int(255.99 * fb[pixel_index].b());
		SDL_SetRenderDrawColor(renderer, ir, ig, ib, 255);
		SDL_Rect rectangle;
		rectangle.x = i;
		rectangle.y = height - (j + 1);
		rectangle.w = 1;
		rectangle.h = 1;
		SDL_RenderFillRect(renderer, &rectangle);
		SDL_RenderPresent(renderer);
		if (j == height) {
			j = 0;
		}
		else {
			if (i < width) {
				i++;
			}
			else {
				i = 0;
				j++;
			}
		}
	}

	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));


	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	cudaDeviceReset();

	return 0;
}