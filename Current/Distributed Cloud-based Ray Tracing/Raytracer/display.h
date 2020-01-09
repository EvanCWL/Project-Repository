#pragma once

#include "SDL.h"
#include <string>
#include <iostream>

class display {
public:
	display() {}
	display(const std::string& title, int width, int height);
	void closeDisplay();
	__device__ void draw(int x, int y, int r, int g, int b);
	int get_width() { return width; }
	int get_height() { return height; }
	bool get_status() { return running; }
	void clear_render() const { SDL_RenderClear(renderer); }
	void present_render() const { SDL_RenderPresent(renderer); }
	void copy_to_renderer(SDL_Renderer* renderer) {
		SDL_RenderCopy(renderer, texture, NULL, NULL);
	}
	void close() { running = false; }
	SDL_Window* get_window() { return window; }
	SDL_Renderer* get_renderer() { return renderer; }

	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Texture* texture;

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

	texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);
	if (texture == NULL) {
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

__device__ inline void display::draw(int x, int y, int r, int g, int b)
{
	SDL_SetRenderDrawColor(renderer, r, g, b, 255);
	SDL_Rect rec;
	rec.x = x;
	rec.y = height - (y + 1);
	rec.w = 1;
	rec.h = 1;
	SDL_RenderFillRect(renderer, &rec);
	SDL_RenderPresent(renderer);
}
