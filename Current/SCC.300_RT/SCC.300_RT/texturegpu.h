#pragma once
#include "SDL.h"
#include "device_launch_parameters.h"
#include <type_traits>
#include "cuda_utils.h"


class TextureGPU {
public:
    TextureGPU(SDL_Renderer* renderer, size_t width, size_t height,
        float scale_factor = 1.0f);

    ~TextureGPU();

    __host__ __device__ size_t get_width() const { return width_; }
    __host__ __device__ size_t get_height() const { return height_; }

    __host__ __device__ size_t get_n_pixels() const { return width_ * height_; }

    __device__ Uint32& access(int x, int y) { return d_pixels_[y * width_ + x]; }

    __device__ void set_rgb(int x, int y, Uint8 r, Uint8 g, Uint8 b) {
        set_rgba(x, y, r, g, b, 0xFF);
    }

    __device__ void set_rgba(int x, int y, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
        access(x, y) = (r >> fmt_->Rloss) << fmt_->Rshift |
            (g >> fmt_->Gloss) << fmt_->Gshift |
            (b >> fmt_->Bloss) << fmt_->Bshift |
            ((a >> fmt_->Aloss) << fmt_->Ashift & fmt_->Amask);
    }

    void copy_to_cpu();
    void copy_to_renderer(SDL_Renderer* renderer) {
        SDL_RenderCopy(renderer, tex_, NULL, NULL);
    }

private:
    SDL_Texture* tex_ = NULL;
    Uint32* d_pixels_ = NULL;

    size_t width_ = 0;
    size_t height_ = 0;

    size_t size_in_bytes_ = 0;

    SDL_PixelFormat* fmt_ = NULL;
};



TextureGPU::TextureGPU(SDL_Renderer* renderer, size_t width, size_t height,
    float scale_factor)
    : width_(width* scale_factor), height_(height* scale_factor) {

    tex_ = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING, width_, height_);

    fmt_ = cuda_malloc_managed<SDL_PixelFormat>(sizeof(SDL_PixelFormat));
    cudaCheckErr(cudaMemcpy(fmt_, SDL_AllocFormat(SDL_PIXELFORMAT_RGBA32),
        sizeof(SDL_PixelFormat), cudaMemcpyHostToDevice));

    size_in_bytes_ = width_ * height_ * fmt_->BytesPerPixel;

    d_pixels_ = cuda_malloc<Uint32>(size_in_bytes_);
}

TextureGPU::~TextureGPU() {
    cudaCheckErr(cudaFree(d_pixels_));
    cudaCheckErr(cudaFree(fmt_));
    SDL_DestroyTexture(tex_);
}

void TextureGPU::copy_to_cpu() {
    Uint32* h_pixels;
    int pitch = 0;

    cudaDeviceSynchronize();

    SDL_LockTexture(tex_, NULL, (void**)&h_pixels, &pitch);
    cudaCheckErr(
        cudaMemcpy(h_pixels, d_pixels_, size_in_bytes_, cudaMemcpyDeviceToHost));

    SDL_UnlockTexture(tex_);
}