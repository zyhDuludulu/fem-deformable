import numpy as np
import taichi as ti
import time

ti.init()

n_x = 10
n_y = 10
n_z = 10
dx = 1 / 64
h = 4e-3
substep = 100
dt = h / substep
g = 9.8
youngs = 1e7
poisson = 0.0
mu = ti.field(ti.f32, ())
la = ti.field(dtype=ti.f32, shape=())

n_points = n_x * n_y * n_z
n_quads = 5 * (n_x - 1) * (n_y - 1) * (n_z - 1)
x = ti.Vector.field(3, dtype=ti.f32, shape=n_points)
quads = ti.Vector.field(4, dtype=ti.i32, shape=n_quads)
v = ti.Vector.field(3, dtype=ti.f32, shape=n_points)
f = ti.Vector.field(3, dtype=ti.f32, shape=n_points)
A = ti.field(dtype=ti.f32, shape=n_quads)
Dm_invs = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_quads)

@ti.func
def ijk2index(i, j, k): return i * n_y * n_z + j * n_z + k

@ti.kernel
def init_material():
    mu[None] = youngs / (2 * (1 + poisson))
    la[None] = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))

@ti.kernel
def initPointPos():
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                index = ijk2index(i, j, k)
                x[index] = ti.Vector([(i - n_x // 2) * dx, (j - n_y // 2) * dx, (k - n_z // 2) * dx])

@ti.kernel
def init_quads():
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            for k in range(n_z - 1):
                index = (i * (n_y - 1) * (n_z - 1) + j * (n_z - 1) + k) * 5
                quads[index][0] = ijk2index(i, j, k)
                quads[index][1] = ijk2index(i + 1, j, k)
                quads[index][2] = ijk2index(i + 1, j + 1, k)
                quads[index][3] = ijk2index(i + 1, j, k + 1)

                index += 1
                quads[index][0] = ijk2index(i, j, k)
                quads[index][1] = ijk2index(i, j + 1, k)
                quads[index][2] = ijk2index(i + 1, j + 1, k)
                quads[index][3] = ijk2index(i, j + 1, k + 1)

                index += 1
                quads[index][0] = ijk2index(i, j, k)
                quads[index][1] = ijk2index(i, j, k + 1)
                quads[index][2] = ijk2index(i, j + 1, k + 1)
                quads[index][3] = ijk2index(i + 1, j, k + 1)

                index += 1
                quads[index][0] = ijk2index(i + 1, j + 1, k + 1)
                quads[index][1] = ijk2index(i, j + 1, k + 1)
                quads[index][2] = ijk2index(i + 1, j + 1, k)
                quads[index][3] = ijk2index(i + 1, j, k + 1)

                index += 1
                quads[index][0] = ijk2index(i + 1, j + 1, k + 1)
                quads[index][1] = ijk2index(i + 1, j, k)
                quads[index][2] = ijk2index(i, j + 1, k)
                quads[index][3] = ijk2index(i, j, k + 1)

@ti.kernel
def compute_Dm_invs():
    for q in range(n_quads):
        i, j, k, l = quads[q]
        Dm = ti.Matrix.cols([x[i] - x[j], x[k] - x[j], x[l] - x[j]])
        Dm_invs[q] = Dm.inverse()
        A[q] = ti.abs(Dm.determinant()) * 0.5

@ti.kernel
def init_force():
    for i in range(n_points):
        f[i] = ti.Vector([0, 0, 0])

@ti.kernel
def compute_force():
    for q in range(n_quads):
        i, j, k, l = quads[q]
        Ds = ti.Matrix.cols([x[i] - x[j], x[k] - x[j], x[l] - x[j]])
        F = Ds @ Dm_invs[q]
        E = 0.5 * (F.transpose() @ F - ti.Matrix.identity(ti.f32, 3))
        P = 2 * mu[None] * E + la[None] * E.trace() * ti.Matrix.identity(ti.f32, 3)
        H = (A[q] * P @ Dm_invs[q].transpose())
        gb = H @ ti.Vector([1, 0, 0])
        gc = H @ ti.Vector([0, 1, 0])
        gd = H @ ti.Vector([0, 0, 1])
        ga = -gb - gc - gd
        f[i] += gb
        f[k] += gc
        f[l] += gd
        f[j] += ga


@ti.kernel
def apply_force():
    for i in range(n_points):
        acc = -f[i] / 1.0 + ti.Vector([0, -g, 0])
        v[i] += acc * dt
        x[i] += v[i] * dt
        v[i] *= ti.exp(-dt * 10.0)

        if x[i][1] < -0.5:
            x[i][1] = -0.5
            v[i][1] = 0.0
            v[i] *= 0.9 

window = ti.ui.Window("Jelly 3D", (800, 800))
canvas = window.get_canvas()
canvas.set_background_color((0.8, 0.8, 0.8))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

initPointPos()
init_quads()
compute_Dm_invs()
init_material()

while window.running:
    for e in window.get_events():
        if e.key == ti.ui.ESCAPE:
            window.running = False

    # Simulation
    for _ in range(substep):
        init_force()
        compute_force()
        apply_force()

    # Render
    camera.position(0.0, -0.25, 1)
    camera.lookat(0.0, -0.25, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.particles(x, radius=0.002, color=(0.5, 0.5, 0.5))
    canvas.scene(scene)
    window.show()
