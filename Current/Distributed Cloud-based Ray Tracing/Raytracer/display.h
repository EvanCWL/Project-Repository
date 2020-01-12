#pragma once

#include "SDL.h"
#include <string>
#include <iostream>
#include <thread>
class display {
public:
	display() {}
	display(const std::string& title, int width, int height);
	void closeDisplay();
	void update(vec3* fb);
	int get_width() { return width; }
	int get_height() { return height; }
	bool get_status() { return running; }
	void clear_render() const { SDL_RenderClear(renderer); }
	void present_render() const { SDL_RenderPresent(renderer); }
	void close() { running = false; }
	SDL_Window* get_window() { return window; }
	SDL_Renderer* get_renderer() { return renderer; }

	SDL_Window* window;
	SDL_Renderer* renderer;
	//SDL_Texture* texture;

	int max_thread = 2;

	int width = 0;
	int height = 0;

	bool running = false;

};

display::display(const std::string& title, int width, int height) :width(width),height(height)
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cout << "[Error] Failed to initialise SDL2";
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
	if (window == NULL) {
		std::cout << SDL_GetError();
	}

	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	if (renderer == NULL) {
		std::cout << SDL_GetError();
	}
	SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
	running = true;
}

inline void display::closeDisplay() {
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	close();
}

inline void display::update(vec3* fb)
{
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
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
		}
	}
	SDL_RenderPresent(renderer);
}
