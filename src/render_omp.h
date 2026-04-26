#pragma once
#include "scene.h"

void render_omp(unsigned char* fb, int W, int H,
                const rt::RtCamera& cam, const rt::Scene& scene,
                int samples, int max_depth);
