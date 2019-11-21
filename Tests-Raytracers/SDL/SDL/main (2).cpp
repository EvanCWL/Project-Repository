#include <iostream>
#include <SDL.h>
#include <SDL_image.h>



int main(int argc, char* argv[]) {
	int width = 1280, height = 720;
	bool running = true;

	SDL_Init(SDL_INIT_VIDEO);
	
	SDL_Window* window = SDL_CreateWindow("SDL2", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
	if (window == NULL) {
		std::cout << "Could not create window: " << SDL_GetError();
		return 1;
	}
	/*
	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	SDL_RWops* rwop = SDL_RWFromFile("D:/Chua's File/Project-Repository/Distributed Cloud-based Raytracing/Raytracing/Raytracing/frames/output.bmp", "rb");
	SDL_Surface* surface = IMG_LoadBMP_RW(rwop);
	if (!surface) {
		printf("IMG_LoadBMP_RW: %s\n", IMG_GetError());
		// handle error
	}
	SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
	SDL_FreeSurface(surface);
	*/
	while (running) {
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
			}
		}
		//SDL_RenderClear(renderer);
		//SDL_RenderCopy(renderer, texture, NULL, NULL);
		//SDL_RenderPresent(renderer);
	}
	//SDL_DestroyTexture(texture);
	//SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);

	SDL_Quit();

	return 0;
}