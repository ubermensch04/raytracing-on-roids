//raylib window, mode switching, and HUD
#include "raylib.h"
#include "scene.h"
#include "render_seq.h"
#include "render_omp.h"
#include "render_cuda.h"

int main() {
    const int W          = 800;
    const int H          = 450;
    const int SAMPLES    = 50;
    const int MAX_DEPTH  = 10;

    InitWindow(W, H, "Parallel Ray Tracer");
    SetTargetFPS(60);

    unsigned char* fb = new unsigned char[W * H * 4]();

    Image     img    = { fb, W, H, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
    Texture2D screen = LoadTextureFromImage(img);

    rt::Scene    scene = rt::build_final_scene();
    rt::RtCamera cam;
    cam.init(W, H);

    const char* mode_names[] = { "Sequential", "OpenMP", "CUDA" };
    int    mode    = 0;
    double last_ms = 0.0;
    double seq_ms  = 1.0;
    bool   dirty   = true;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ONE))   { mode = 0; dirty = true; }
        if (IsKeyPressed(KEY_TWO))   { mode = 1; dirty = true; }
        if (IsKeyPressed(KEY_THREE)) { mode = 2; dirty = true; }
        if (IsKeyPressed(KEY_R))     { dirty = true; }

        if (dirty) {
            dirty = false;
            double t0 = GetTime();

            if (mode == 0) render_sequential(fb, W, H, cam, scene, SAMPLES, MAX_DEPTH);
            if (mode == 1) render_omp       (fb, W, H, cam, scene, SAMPLES, MAX_DEPTH);
            if (mode == 2) render_cuda      (fb, W, H, cam, scene, SAMPLES, MAX_DEPTH);

            last_ms = (GetTime() - t0) * 1000.0;
            if (mode == 0) seq_ms = last_ms;

            UpdateTexture(screen, fb);
        }

        BeginDrawing();
            DrawTexture(screen, 0, 0, WHITE);
            DrawRectangle(0, 0, 320, 110, { 0, 0, 0, 160 });
            DrawText(TextFormat("Mode: %s  [1/2/3]",     mode_names[mode]), 8, 8,  18, YELLOW);
            DrawText(TextFormat("Render: %.0f ms",        last_ms),          8, 30, 18, WHITE);
            DrawText(TextFormat("Speedup: %.1fx",         seq_ms/last_ms),   8, 52, 18, LIME);
            DrawText(TextFormat("SPP: %d  Depth: %d",    SAMPLES,MAX_DEPTH), 8, 74, 18, WHITE);
            DrawText("R = re-render",                                         8, 96, 16, DARKGRAY);
        EndDrawing();
    }

    UnloadTexture(screen);
    delete[] fb;
    CloseWindow();
    return 0;
}
