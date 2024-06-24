import numpy as np
import taichi as ti
import warp as wp
import time

wp.init()
ti.init()

N_x = 10
N_y = 10
dx = 1 / 64
h = 4e-3
substepping = 100
g = 9.8
youngs = 1e6
poisson = 0.0

N_points = N_x * N_y
N_edges = (N_x - 1) * N_y + (N_y - 1) * N_x + (N_x - 1) * (N_y - 1)
N_triangles = 2 * (N_x - 1) * (N_y - 1)
x_cpu = np.zeros((N_points, 2), dtype=np.float32)
triangles_cpu = np.zeros((N_triangles, 3), dtype=np.uint32)
edges_cpu = np.zeros((N_edges, 2), dtype=np.uint32)
v = wp.empty(N_points, dtype=wp.vec2f, device="cuda")
f = wp.empty(N_points, dtype=wp.vec2f, device="cuda")
A = wp.empty(N_triangles, dtype=wp.float32)
Dm_invs = wp.empty(N_triangles, dtype=wp.mat22f)

mu = wp.constant(youngs / (2 * (1 + poisson)))
la = wp.constant(youngs * poisson / ((1 + poisson) * (1 - 2 * poisson)))
dh = wp.constant(h / substepping)

def ij2index(i: wp.uint8, j: wp.uint8): return i * N_y + j

def setUp():
    # triangles
    for i in range(N_x - 1):
        for j in range(N_y - 1):
            index = (i * (N_y - 1) + j) * 2
            triangles_cpu[index][0] = ij2index(i, j)
            triangles_cpu[index][1] = ij2index(i + 1, j)
            triangles_cpu[index][2] = ij2index(i, j + 1)

            index += 1
            triangles_cpu[index][0] = ij2index(i, j + 1)
            triangles_cpu[index][1] = ij2index(i + 1, j + 1)
            triangles_cpu[index][2] = ij2index(i + 1, j)

    # edges
    eid_base = 0
    for i in range(N_x-1):
        for j in range(N_y):
            eid = eid_base+i*N_y+j
            edges_cpu[eid] = [ij2index(i, j), ij2index(i+1, j)]

    eid_base += (N_x-1)*N_y
    for i in range(N_x):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges_cpu[eid] = [ij2index(i, j), ij2index(i, j+1)]

    eid_base += N_x*(N_y-1)
    for i in range(N_x-1):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges_cpu[eid] = [ij2index(i+1, j), ij2index(i, j+1)]

    for i in range(N_x):
        for j in range(N_y):
            index = ij2index(i, j)
            x_cpu[index] = [(i - N_x // 2) * dx + 0.5, (j - N_y // 2) * dx + 0.5]

@wp.kernel
def compute_Dm_invs(Dm_invs_: wp.array(dtype=wp.mat22f), x_: wp.array(dtype=wp.vec2f), triangles_: wp.array(dtype=wp.vec3i), A_: wp.array(dtype=wp.float32)): # type:ignore
    tid = wp.tid()
    a = triangles_[tid][0]
    b = triangles_[tid][1]
    c = triangles_[tid][2]
    Dm = wp.mat22f(x_[b] - x_[a], x_[c] - x_[a])
    Dm_invs_[tid] = wp.inverse(Dm)
    A_[tid] = 0.5 * wp.abs(wp.determinant(Dm))

@wp.kernel
def init_f(f_: wp.array(dtype=wp.vec2f)): # type:ignore
    tid = wp.tid()
    f_[tid] = wp.vec2f(0., 0.)

@wp.kernel
def compute_force(Dm_invs_: wp.array(dtype=wp.mat22f), x_: wp.array(dtype=wp.vec2f), triangles_: wp.array(dtype=wp.vec3i), f_: wp.array(dtype=wp.vec2f), A_: wp.array(dtype=wp.float32)): # type:ignore
    tid = wp.tid()
    a = triangles_[tid][0]
    b = triangles_[tid][1]
    c = triangles_[tid][2]
    Ds = wp.mat22f(x_[b] - x_[a], x_[c] - x_[a])
    F = Ds @ Dm_invs_[tid]
    E = 0.5 * (wp.transpose(F) @ F - wp.identity(n=2, dtype=wp.float32))
    P = F @ (2. * mu * E)
    grad = A_[tid] * P @ wp.transpose(Dm_invs_[tid])
    grad = wp.transpose(grad)
    wp.atomic_add(f_, b, grad[0])
    wp.atomic_add(f_, c, grad[1])
    wp.atomic_add(f_, a, -grad[0] - grad[1])
    
@wp.kernel
def update(x_: wp.array(dtype=wp.vec2f), v_: wp.array(dtype=wp.vec2f), f_: wp.array(dtype=wp.vec2f)): # type:ignore
    tid = wp.tid()
    acc = -f_[tid] / 1.0 + wp.vec2f(0., -g)
    v_[tid] += acc * dh
    x_[tid] += v_[tid] * dh
    v_[tid] *= wp.exp(-dh * 1.)

    # boundary conditions
    if x_[tid][1] < 0:
        x_[tid][1] = 0.
        v_[tid][1] = 0.
        v_[tid][0] *= 0.9

gui = ti.GUI('Deformable Simulation', (800, 800))
setUp()
x = wp.array(x_cpu, dtype=wp.vec2f, device="cuda")
triangles = wp.array(triangles_cpu, dtype=wp.vec3i, device="cuda")
wp.launch(
    kernel=compute_Dm_invs,
    dim=N_triangles,
    inputs=[Dm_invs, x, triangles, A]
)
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.EXIT:
            exit()

    for _ in range(substepping):
        wp.launch(kernel=init_f, dim=N_points, inputs=[f])
        wp.launch( kernel=compute_force, dim=N_triangles, inputs=[Dm_invs, x, triangles, f, A])
        wp.launch(kernel=update, dim=N_points, inputs=[x, v, f])

    # render
    x_cpu = x.numpy()
    for i in range(N_edges):
        a, b = edges_cpu[i][0], edges_cpu[i][1]
        gui.line(x_cpu[a], x_cpu[b], radius=1, color=0xFFFF00)
    gui.show()
