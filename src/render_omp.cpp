//OpenMP render loop — Phase 2 (currently delegates to sequential)
#include "render_omp.h"
#include "render_seq.h"

void render_omp(unsigned char* fb, int W, int H,
                const rt::RtCamera& cam, const rt::Scene& scene,
                int samples, int max_depth)
{
    render_sequential(fb, W, H, cam, scene, samples, max_depth);
}
