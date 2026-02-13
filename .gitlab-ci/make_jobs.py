from copy import deepcopy
from typing import Tuple

class V:
    def __init__(self, major, minor):
        self.major = int(major)
        self.minor = int(minor)

    @property
    def dot(self) -> str:
        return f'{self.major}.{self.minor}'
    @property
    def nodot(self) -> str:
        return f'{self.major}{self.minor}'

    @property
    def version(self) -> Tuple[int, int]:
        return (self.major, self.minor)

def Vs(l: list):
    return [V(*o.split('.')) for o in l]

oses = ['windows', 'linux']
pythons = Vs(['3.8', '3.9', '3.10', '3.11', '3.12', '3.13', '3.14'])
cuda_archs = {
    '11.8': '50;52;61;75;86',
    '12.6': '50;52;61;75;86;89',
}
cudas = Vs(cuda_archs)

def cuda_xver(cuda):
    if cuda.version >= (11, 2):
        return f'{cuda.major}x'
    else:
        return f'{cuda.nodot}'

def cuda_suffix(cuda):
    return '-cuda' + cuda_xver(cuda)

def add_jobs(jobs: dict, os: V, py: V, cuda: V) -> None:
    test_pkgs = []
    # NOTE: CuPy is not yet built for Python 3.14
    if py.version < (3, 14):
        test_pkgs = [f'cupy-cuda{cuda_xver(cuda)}']
    # since h5py causes aborts on Windows
    if os != 'windows':
        test_pkgs.append('h5py')

    vs = {
        'PY_VER': py.nodot,
        'PY_VER_DOT': py.dot,
        'CUDA_VER': cuda.nodot,
        'CUDA_VER_DOT': cuda.dot,
        'CUDA_VER_X': cuda_xver(cuda),
        'CUDAARCHS': cuda_archs[cuda.dot],
        'VORTEX_BUILD_SUFFIX': cuda_suffix(cuda),
        'TEST_PKGS': ' '.join(test_pkgs),
    }

    tags = [ f'py{py.nodot}', f'cuda{cuda.nodot}' ]
    build = f'build-{os}-py{py.nodot}-cuda{cuda_xver(cuda)}'
    test = f'test-{os}-py{py.nodot}-cuda{cuda_xver(cuda)}'

    jobs.update({
        build: {
            'tags': tags,
            'variables': vs,
            'extends': [f'.build-{os}'],
        },
        test: {
            'tags': tags,
            'variables': vs,
            'extends': [f'.test-{os}'],
            'needs': [build],
        }
    })

def _get_parents(jobs: dict, job: dict) -> list:
    extends = job.get('extends', [])
    extends = sum([_get_parents(jobs, jobs[e]) for e in extends], []) + extends
    return extends

def merge_extends(bases: dict, job: dict) -> None:
    for name in reversed(_get_parents(bases, job)):
        base = bases[name]

        for (k, v) in base.items():
            if k in job:
                if isinstance(v, list):
                    # prepend
                    job[k] = v + job[k]
                elif isinstance(v, dict):
                    # reverse update
                    d = v.copy()
                    d.update(job[k])
                    job[k] = d
                else:
                    raise ValueError(f'unsupported: {k} {v}')
            else:
                # assign directly
                job[k] = deepcopy(v)

    # no longer extends
    try:
        del job['extends']
    except KeyError:
        pass

if __name__ == '__main__':
    import sys
    from pathlib import Path

    jobs = {}

    from itertools import product
    for (os, py, cuda) in product(oses, pythons, cudas):
        add_jobs(jobs, os, py, cuda)

    base_text = Path(sys.argv[1]).read_text()

    import yaml
    bases = yaml.safe_load(base_text)

    for job in jobs.values():
        if isinstance(job, dict):
            merge_extends(bases, job)
    for base in bases.values():
        if isinstance(base, dict):
            merge_extends(bases, base)

    # remove bases
    for k in list(bases):
        if k.startswith('.'):
            del bases[k]

    print(yaml.safe_dump(bases, indent=4, width=99999))
    print(yaml.safe_dump(jobs, indent=4, width=99999))
