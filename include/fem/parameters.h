#ifndef FEM_PARAMETERS_H
#define FEM_PARAMETERS_H

namespace params {
    const int n_x = 10;
    const int n_y = 10;
    const int n_z = 10;
    const int window_width = 800;
    const int window_height = 800;
    const int sub_steps = 100;
    const int quads_per_cube = 4;
    const float dx = 1. / 32.;
    const float time_step = 4e-3;
    const float dt = time_step / sub_steps;
    const float damping = 0.95;
    const float g = 9.8;
    const float m = 1.0;
    const float youngs_modulus = 1e6;
    const float poisson_ratio = 0.;
    const float mu = youngs_modulus / (2 * (1 + poisson_ratio));
    const float lambda = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio));
}

#endif // FEM_PARAMETERS_H