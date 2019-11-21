#define STB_IMAGE_IMPLEMENTATION

#include <Windows.h>
#include <stdint.h>

#include <gl/GL.h>
#include <SDL.h>
#include <iostream>

// SDL2 state
static SDL_Window *global_window;
static SDL_GLContext global_gl_context;
static bool global_running;

// Window size
static uint32_t global_window_width;
static uint32_t global_window_height;

float texCoords[] = {
	0.0f, 0.0f,  // lower-left corner  
	1.0f, 0.0f,  // lower-right corner
	0.5f, 1.0f   // top-center corner
};

int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {

	// Initialise SDL2
	if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		OutputDebugString("[Error] Failed to initialise SDL2");
		return 1;
	}

	// Create the SDL2 window
	global_window = SDL_CreateWindow("SDL2", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL);
	if (!global_window) {
		OutputDebugString("[Error] Failed to create SDL2 window");
		return 1;
	}

	// Select the version of OpenGL you want to use.. This just uses 3.3
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

	// Choose the profile version you want.. This just selects the core profile
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	// Create the OpenGL context
	global_gl_context = SDL_GL_CreateContext(global_window);
	if (!global_gl_context) {
		OutputDebugString("[Error] Failed to create OpenGL context");
		return 1;
	}

	global_running = true;
	while (global_running) {
		// Poll for user input events
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			switch (event.type) {
				case SDL_QUIT: { global_running = false; } break;
				case SDL_WINDOWEVENT: {
					if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
						global_window_width  = (uint32_t) event.window.data1;
						global_window_height = (uint32_t) event.window.data2;
					}
				}
				break;
				// ... Handle other events here.
			}
		}

		// Set viewport to draw to entire window
		glViewport(0, 0, global_window_width, global_window_height);

		// Clear the screen color
		glClearColor(1.0f, 1.0f, 1.0f, 1.5f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//Test render
		

		// Swap the window buffers to display screen contents
		SDL_GL_SwapWindow(global_window);

	}
	
	// Delete OpenGL context and destroy the window
	SDL_GL_DeleteContext(global_gl_context);
	SDL_DestroyWindow(global_window);

	// Quit SDL
	SDL_Quit();

	return 0;
}