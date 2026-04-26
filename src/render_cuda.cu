//CUDA kernel — Phase 3 (currently delegates to sequential)
#include "render_cuda.h"
#include "render_seq.h"

void render_cuda(unsigned char* fb, int W, int H,
                 const rt::RtCamera& cam, const rt::Scene& scene,
                 int samples, int max_depth)
{
    render_sequential(fb, W, H, cam, scene, samples, max_depth);
}
