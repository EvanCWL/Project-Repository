#include <iostream>
#include <stdexcept>
#include <string>
#include <SDL.h>

class Window {
public:
    friend class Viewport;

    Window(const std::string& title, size_t width, size_t height);
    ~Window();

    size_t get_width() const { return width_; }
    size_t get_height() const { return height_; }
    float get_aspect_ratio() const { return float(width_) / height_; }

    SDL_Renderer* get_renderer() const { return renderer_; }

    void clear_render() const { SDL_RenderClear(renderer_); }
    void present_render() const { SDL_RenderPresent(renderer_); }

    bool should_quit() const { return quit_; }
    void close() { quit_ = true; }

    float get_delta_time() const { return delta_time_; }
    float get_fps() const { return 1.0f / delta_time_; }
    void update_delta_time();

private:
    SDL_Window* window_;
    SDL_Renderer* renderer_;

    float delta_time_ = 0.0f;
    size_t last_frame_ = 0;

    size_t width_ = 0;
    size_t height_ = 0;

    bool quit_ = true;
};


Window::Window(const std::string& title, size_t width, size_t height)
    : width_(width), height_(height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        throw std::runtime_error(SDL_GetError());
    }

    if (!SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1")) {
        std::cout << "Warning: Linear texture filtering not enabled!" << std::endl;
    }

    window_ = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED, width_, height_,
        SDL_WINDOW_SHOWN);

    if (!window_) {
        throw std::runtime_error(SDL_GetError());
    }

    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);

    if (!renderer_) {
        throw std::runtime_error(SDL_GetError());
    }

    SDL_SetRenderDrawColor(renderer_, 0xFF, 0xFF, 0xFF, 0xFF);

    quit_ = false;
}

Window::~Window() {
    SDL_DestroyRenderer(renderer_);
    SDL_DestroyWindow(window_);
}

void Window::update_delta_time() {
    size_t current_frame = SDL_GetTicks();
    delta_time_ = float(current_frame - last_frame_) / 1000.0f;
    last_frame_ = current_frame;
}
