import subprocess
import os
from pathlib import Path

def _flatten(os):
    o2 = []
    for o in os:
        if isinstance(o, str):
            o2.append(o)
        else:
            o2.extend(_flatten(o))

    return o2

def _dump_vars(title, vars, delim=None, flip=False):
    if not vars:
        return
    if delim is None:
        delim = ' = '

    if flip:
        vars = {v: k for (k, v) in vars.items()}

    pad = max([len(k) for k in vars])

    print(f'-- {title}:', flush=True)
    print(flush=True)
    for k in sorted(vars):
        print(f'    {k: <{pad}}{delim}{vars[k]}', flush=True)
    print(flush=True)

def _dump_list(title, l):
    print(f'-- {title}:', flush=True)
    print(flush=True)
    for v in sorted(l):
        print(f'    {v}', flush=True)
    print(flush=True)

def _get_alazar_version():
    out = subprocess.check_output(['dpkg', '-s', 'libats']).decode('ascii')
    for line in out.splitlines():
        (key, _, value) = line.strip().partition(': ')
        if key.lower() == 'version':
            return value

    raise RuntimeError('could not detect Alazar version:\n{out}')

PY_VER_DOT = os.environ['PY_VER_DOT']
CUDA_VER_DOT = os.environ['CUDA_VER_DOT']

CODE_PATH = Path('/code')

CUDA_SRC_PATH = Path(f'/usr/local/cuda-{CUDA_VER_DOT}')
CUDA_DST_PATH = Path('/usr/local/cuda')

MOUNTS = [
    # CUDA
    (CUDA_SRC_PATH, CUDA_DST_PATH),

    # Alazar
    ('/usr/include/AlazarApi.h', ),
    ('/usr/include/AlazarCmd.h', ),
    ('/usr/include/AlazarDSP.h', ),
    ('/usr/include/AlazarError.h', ),
    ('/usr/include/AlazarRC.h', ),
    ('/usr/include/AlazarGalvo.h', ),
    ('/usr/lib/x86_64-linux-gnu/libATSApi.so', '/usr/lib/libATSApi.so'),

    # Teledyne
    ('/usr/include/ADQAPI.h', ),
    ('/usr/lib/x86_64-linux-gnu/libadq.so', '/usr/lib/libadq.so'),

    # NI DAQmx
    ('/usr/include/NIDAQmx.h', ),
    ('/usr/lib/x86_64-linux-gnu/libnidaqmx.so', '/usr/lib/libnidaqmx.so'),

    # bring in the source code
    ('.', CODE_PATH),

    # share cache with host
    ('~/.cache', '/.cache'),
]

ENVIRONMENT = [
    ('CUDAARCHS', ),
    ('CUDAToolkit_ROOT', CUDA_DST_PATH),
    ('CMAKE_GENERATOR', 'Ninja'),
    ('VCPKG_MANIFEST_FEATURES', ),
    ('VORTEX_BUILD_COMPILER', 'gcc'),
    ('VORTEX_BUILD_SUFFIX', ),
    # since the container cannot determine the Alazar version
    ('VORTEX_BUILD_CMAKE_ARGS', f'Alazar_VERSION={_get_alazar_version()}'),

    # provide extra help for vcpkg to find CUDA
    ('CUDA_PATH', CUDA_DST_PATH),
    ('CUDA_BIN_PATH', CUDA_DST_PATH / 'bin'),
]

EXTERNAL_LIBRARIES = [
    'libATSApi.so.0',
    'libnidaqmx.so.1',
    'libadq.so.12.0',
]

def build_mount_vars(mounts):
    mount_vars = {}
    for o in mounts:
        if len(o) == 2:
            (src, dst) = o
        else:
            src = dst = o[0]
        src = Path(src).expanduser().resolve()

        # ensure that the mount points sources exist
        if not src.exists():
            if src.suffixes:
                src.parent.mkdir(parents=True, exist_ok=True)
                src.touch()
            else:
                src.mkdir(parents=True, exist_ok=True)

        mount_vars[src.as_posix()] = str(dst)

    return mount_vars

def build_environment_vars(environment):
    env_vars = {}
    for o in environment:
        if len(o) == 2:
            (k, v) = o
        else:
            k = o[0]
            v = os.environ[k]

        env_vars[k] = str(v)

    return env_vars

def docker(*cmd, env_vars=None, mount_vars=None, log=True, name=None):
    name = name or f'py{PY_VER_DOT}-cuda{CUDA_VER_DOT}'
    env_vars = env_vars or []
    mount_vars = mount_vars or []

    args = _flatten([
        'docker',
        'run',
        [['--mount', f'type=bind,source={src},target={dst}'] for (src, dst) in mount_vars.items()],
        [['--env', f'{k}={v}'] for (k, v) in env_vars.items()],
        '--workdir', CODE_PATH.as_posix(),
        '-u', f'{os.getuid()}:{os.getgid()}',
        'vortex-manylinux2014_x86_64',
    ] + list(cmd))

    if log:
        print('-- Docker:', args, flush=True)
    subprocess.check_call(args)

if __name__ == '__main__':
    mount_vars = build_mount_vars(MOUNTS)
    env_vars = build_environment_vars(ENVIRONMENT)

    _dump_vars('Mount Points', mount_vars, delim=' -> ', flip=True)
    _dump_vars('Environment Variables', env_vars)
    _dump_list('External Libraries', EXTERNAL_LIBRARIES)

    intermediate_dir = Path('dist')
    sdist_path = next(intermediate_dir.glob('vortex*.tar.gz'))
    docker(f'/usr/local/bin/python{PY_VER_DOT}', '-m', 'pip', 'wheel', sdist_path.as_posix(), '-w', intermediate_dir.as_posix(), '-v', env_vars=env_vars, mount_vars=mount_vars)

    for wheel in intermediate_dir.glob('vortex*-linux_*.whl'):
        docker(f'auditwheel', 'repair', wheel.as_posix(), [[f'--exclude', lib] for lib in EXTERNAL_LIBRARIES], '-w', 'dist', env_vars=env_vars, mount_vars=mount_vars)
        wheel.unlink()
