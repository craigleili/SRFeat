import numpy as np
import open3d as o3d
from scipy.io import loadmat


def load_shape(filepath, return_vertex_normals=False):
    mesh = o3d.io.read_triangle_mesh(filepath)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if return_vertex_normals:
        mesh.compute_vertex_normals()
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        return vertices, faces, vertex_normals
    else:
        return vertices, faces


def load_geodist(filepath):
    data = loadmat(filepath)
    geodist = np.asarray(data['geodist'], dtype=np.float32)
    sqrt_area = data['sqrt_area'].toarray().flatten()[0]
    return geodist, sqrt_area


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        res = args
        for t in self.transforms:
            res = t(*res)
        return res


class Centering(object):

    def __call__(self, vertices, faces):
        center = np.mean(vertices, axis=0, keepdims=True)
        return vertices - center, faces


class RandomRotation(object):

    def __init__(self, max_x=360.0, max_y=360.0, max_z=360.0):
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

        self.randg = np.random.RandomState(0)

    def __call__(self, vertices, faces):
        anglex = self.randg.rand() * self.max_x * np.pi / 180.0
        angley = self.randg.rand() * self.max_y * np.pi / 180.0
        anglez = self.randg.rand() * self.max_z * np.pi / 180.0
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.asarray([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]], dtype=vertices.dtype)
        Ry = np.asarray([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]], dtype=vertices.dtype)
        Rz = np.asarray([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]], dtype=vertices.dtype)
        Rxyz = self.randg.permutation(np.stack((Rx, Ry, Rz), axis=0))
        R = Rxyz[2] @ Rxyz[1] @ Rxyz[0]

        return np.asarray(vertices @ R.T, dtype=vertices.dtype), faces


class RandomRotationX(RandomRotation):

    def __init__(self, max_x=360.0, max_y=0, max_z=0):
        super().__init__(max_x=max_x, max_y=max_y, max_z=max_z)


class RandomRotationY(RandomRotation):

    def __init__(self, max_x=0, max_y=360.0, max_z=0):
        super().__init__(max_x=max_x, max_y=max_y, max_z=max_z)


class RandomRotationZ(RandomRotation):

    def __init__(self, max_x=0, max_y=0, max_z=360.0):
        super().__init__(max_x=max_x, max_y=max_y, max_z=max_z)


class RandomScaling(object):

    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.randg = np.random.RandomState(0)

    def __call__(self, vertices, faces):
        scale = self.min_scale + self.randg.rand(1, 3) * (self.max_scale - self.min_scale)
        return np.asarray(vertices * scale, dtype=vertices.dtype), faces


class RandomNoise(object):

    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip

        self.randg = np.random.RandomState(0)

    def __call__(self, vertices, faces):
        shape = vertices.shape
        noise = self.std * self.randg.randn(*shape)
        noise = np.clip(noise, -self.clip, self.clip)
        return np.asarray(vertices + noise, dtype=vertices.dtype), faces


TRANSFORMS = {
    'centering': Centering,
    'rotation': RandomRotation,
    'rotationx': RandomRotationX,
    'rotationy': RandomRotationY,
    'rotationz': RandomRotationZ,
    'scaling': RandomScaling,
    'noise': RandomNoise,
}


def get_transforms(augments):

    def convert(s):
        try:
            return float(s)
        except ValueError:
            return s

    if isinstance(augments, (list, tuple)):
        tsfms = list()
        for aug in augments:
            aug_terms = aug.split('/')
            aug_name = aug_terms[0]
            if len(aug_terms) > 1:
                aug_args = [convert(t) for t in aug_terms[1:]]
                tsfms.append(TRANSFORMS[aug_name](*aug_args))
            else:
                tsfms.append(TRANSFORMS[aug_name]())
        tsfms = Compose(tsfms)
        return tsfms
    else:
        return None
