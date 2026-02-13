# ref: http://www.benjack.io/2018/02/02/python-cpp-revisited.html

import os
import re
import sys
import platform
import subprocess
from pathlib import Path
import json
from textwrap import dedent
from tempfile import TemporaryDirectory

from packaging.version import Version
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

root = Path(__file__).resolve().parent

# NOTE: this function is imported by doc/build.py
def dump_vars(title, vars):
    if not vars:
        return

    pad = max([len(k) for k in vars])

    print(f'-- {title}:', flush=True)
    print(flush=True)
    for k in sorted(vars):
        print(f'    {k: <{pad}} = {vars[k]}', flush=True)
    print(flush=True)

# NOTE: this function is imported by setup.py.in
def cmake_bool(s: str):
    if s.lower() in ['on', 'yes', 'true', 'y']:
        return True
    try:
        return bool(int(s))
    except ValueError:
        return False

def _extract_version(s: str, prefix=None):
    if prefix is None:
        prefix = ''

    for l in s.splitlines():
        m = re.search(rf'{prefix}(\d+\.\d+\.\d+)', l)
        if m:
            return Version(m.group(1))

    raise RuntimeError('failed to detect version')

def _get_cmake_command():
    return os.environ.get('CMAKE_COMMAND', 'cmake')

# NOTE: this function is imported by doc/build.py
def detect_vortex_version():
    try:
        # detect version
        output = subprocess.check_output(['git', 'describe', '--tag', '--always', 'HEAD'], cwd=root).decode().strip()
        parts = output[1:].split('-')

        if len(parts) == 1:
            version = parts[0]
        else:
            version = f'{parts[0]}+{parts[1]}.{parts[2]}'
        print(f'-- Detected vortex v{version}', flush=True)

        # cache to disk
        Path(root / 'VERSION').write_text(version)

    except subprocess.CalledProcessError:
        # read cached version when building Python sdist
        version = Path(root / 'VERSION').read_text().strip()
        print(f'-- Detected vortex v{version} (cached)', flush=True)

    return version

def _detect_cuda_version():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # generate a simple CMake project for FindCUDAToolkit
        cmake_version = _extract_version((root / 'CMakeLists.txt').read_text())
        (tmp / 'CMakeLists.txt').write_text(dedent(f'''
            cmake_minimum_required(VERSION {cmake_version})
            project(DetectCUDAToolkit LANGUAGES C)
            find_package(CUDAToolkit)
        '''))

        # configure that project
        out = subprocess.check_output([_get_cmake_command(), '-S', tmp.as_posix(), '-B', (tmp / 'build').as_posix()]).decode()

        return _extract_version(out, 'CUDAToolkit.*?')

def _extract_cmake_configure_preset(path: Path, preset: str):
    root = json.load(path.open())
    presets = {o.get('name'): o for o in root.get('configurePresets', [])}

    vars = {}
    env = {}
    generator = None

    def _merge(src, dst):
        for (k, v) in src.items():
            # normalize Booleans for CMake
            if isinstance(v, bool):
                v = 'ON' if v else 'OFF'

            # ignore substitutions
            if '$' in v:
                continue

            # do not overwrite since walking hierarchy backwards
            if k not in dst:
                dst[k] = src[k]

    while preset:
        # extract preset
        try:
            obj = presets[preset]
        except KeyError as e:
            raise KeyError(f'unable to find configure preset {preset}') from e

        # extract configuration
        _merge(obj.get('cacheVariables', {}), vars)
        _merge(obj.get('environment', {}), env)

        # process parents
        preset = obj.get('inherits')
        generator = obj.get('generator', generator)

    return (vars, env, generator)

def _decode_extra_args():
    try:
        args_spec = os.environ['VORTEX_BUILD_CMAKE_ARGS']
    except KeyError:
        return {}

    args = {}
    for spec in args_spec.split(';'):
        key, _, value = spec.strip().partition('=')

        args[key] = value

    return args

def _decode_features():
    try:
        return os.environ['VCPKG_MANIFEST_FEATURES'].split(';')
    except KeyError:
        return ['asio', 'backward', 'cuda', 'reflexxes', 'fftw', 'hdf5', 'python']

def _ensure_feature(new_feature):
    features = _decode_features()
    if new_feature not in features:
        features.append(new_feature)
    return ';'.join(features)

def _auto_vcpkg_enabled():
    val = os.environ.get('VORTEX_DISABLE_AUTO_VCPKG', None)
    if not val:
        return True

    try:
        return bool(int(val))
    except ValueError:
        return True

def _path_or_none(s):
    if s is None:
        return s
    else:
        return Path(s)

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir=None, version=None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = Path(sourcedir or root).resolve()
        self.version = version

class CMakeBuild(build_ext):
    def run(self):
        self.cmake_command = _get_cmake_command()

        try:
            out = subprocess.check_output([self.cmake_command, '--version']).decode()
        except OSError:
            raise RuntimeError('CMake must be installed to build the following extensions: ' + ', '.join(e.name for e in self.extensions))

        cmake_version = _extract_version(out)
        required_version = _extract_version((root / 'CMakeLists.txt').read_text())
        if cmake_version < required_version:
            raise RuntimeError(f'CMake >= {required_version} is required')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension):
        # info needed for CMake variables
        python_var_name = 'Python'
        if sys.version_info.major != 2:
            python_var_name += str(sys.version_info.major)

        cmake_vars = {
            # force CMake to detect the currently executing Python
            f'{python_var_name}_EXECUTABLE:FILEPATH': sys.executable,

            # let setuptools build the wheel
            'BUILD_PYTHON_WHEEL:BOOL': 'OFF',

            # enable to modular builds for maximum compatibility
            'ENABLE_MODULAR_BUILD:BOOL': 'ON'
        }

        # add vortex version if known
        if ext.version:
            cmake_vars['VORTEX_VERSION_STRING'] = ext.version

        # environment variables
        cmake_env_vars = {}

        # determine configure preset and triplet
        preset = []
        triplet = []

        # choose platform
        if platform.system() == 'Windows':
            preset.extend(['clang', 'win'])
            triplet.append('windows')
            script_suffix = 'bat'

        else:

            # allow compiler override from envirionment variable
            compiler = os.environ.get('VORTEX_BUILD_COMPILER')
            if not compiler:
                try:
                    # prefer clang
                    subprocess.check_call(['clang', '-v'])
                    compiler = 'clang'
                except FileNotFoundError:
                    # fall back to GCC
                    compiler = 'gcc'
                    subprocess.check_call(['gcc', '-v'])

            preset.extend([compiler.lower(), 'linux'])
            triplet.append('linux')
            script_suffix = 'sh'

        # choose 32 or 64 bit
        if sys.maxsize > 2**32:
            preset.append('x64')
            triplet.insert(0, 'x64')
        else:
            preset.append('x86')
            triplet.insert(0, 'x86')

        # choose debug or release
        if self.debug:
            preset.append('debug')
        else:
            preset.append('release')

        preset = '-'.join(preset)
        triplet = '-'.join(triplet)
        print(f'-- Configure Preset: {preset}', flush=True)

        # populate variables from the preset
        (preset_vars, preset_env_vars, generator) = _extract_cmake_configure_preset(root / 'CMakePresets.json', preset)
        cmake_vars.update(preset_vars)
        cmake_env_vars.update(preset_env_vars)

        # populate variables from environment
        cmake_vars.update(_decode_extra_args())

        def show(k):
            for s in ['CMAKE', 'VORTEX', 'VCPKG', 'CUDAARCHS', 'CUDAToolkit']:
                if k.startswith(s):
                    return True

        # clean up build directory
        build_dir = Path(self.build_temp).resolve()

        # configure vcpkg, ensuring Python is in feature specification
        cmake_vars['VCPKG_MANIFEST_FEATURES'] = _ensure_feature('python')

        # check if toolchain or vcpkg root is specified
        toolchain = _path_or_none(os.environ.get('CMAKE_TOOLCHAIN_FILE', None))
        vcpkg_root = _path_or_none(os.environ.get('VCPKG_ROOT', None))
        path_to_vcpkg_cmake = Path('scripts/buildsystems/vcpkg.cmake')

        if toolchain:
            # warn if vcpkg root is also given
            if vcpkg_root and vcpkg_root / path_to_vcpkg_cmake != toolchain:
                print(f'-- Using CMAKE_TOOLCHAIN_FILE over VCPKG_ROOT', flush=True)
        elif vcpkg_root:
            toolchain = Path(vcpkg_root) / path_to_vcpkg_cmake

        # check if vcpkg should be installed
        if not toolchain and _auto_vcpkg_enabled():
            # try to set up vcpkg in build directory
            vcpkg_root = build_dir / 'vcpkg'
            toolchain = vcpkg_root / path_to_vcpkg_cmake

            # clone vcpkg
            if not toolchain.exists():
                print(f'-- Cloning vcpkg to {vcpkg_root}', flush=True)
                subprocess.check_call(['git', 'clone', 'https://github.com/microsoft/vcpkg.git', str(vcpkg_root)])
                subprocess.check_call([str(vcpkg_root / f'bootstrap-vcpkg.{script_suffix}'), '-disableMetrics'])

        if toolchain:
            # set CMake to use toolchain
            cmake_vars['CMAKE_TOOLCHAIN_FILE'] = str(toolchain)

        # update existing environment variables
        cmake_env = os.environ.copy()
        cmake_env.update(cmake_env_vars)

        # dump variables before vcpkg is invoked
        dump_vars('CMake Variables', cmake_vars)
        dump_vars('CMake Environment', {k: v for (k, v) in cmake_env.items() if show(k) or k in cmake_env_vars})

        # configure with CMake
        cmake_configure_cmd = [
            self.cmake_command
        ] + (['-G', generator] if generator else []) + [
            '-S', str(ext.sourcedir),
            '-B', str(build_dir),
        ] + [f'-D{k}={v}' for (k, v) in cmake_vars.items()]
        print('-- CMake Configure:', cmake_configure_cmd, flush=True)
        subprocess.check_call(cmake_configure_cmd, env=cmake_env)

        # build with CMake
        cmake_build_cmd = [self.cmake_command, '--build', str(build_dir)]
        print('-- CMake Build:', cmake_build_cmd, flush=True)
        subprocess.check_call(cmake_build_cmd, env=cmake_env)

        # read out module paths
        module_paths = [Path(p) for p in (build_dir / 'lib' / 'modules.txt').read_text().splitlines()]

        self.prepare_wheel(ext, build_dir, module_paths)

    def prepare_wheel(self, ext, build_dir, module_paths, deploy_dependencies=True, generate_stubs=True, deploy_path_restriction=None):
        # format files for wheel
        extpath = Path.cwd() / Path(self.get_ext_fullpath(ext.name)).resolve()
        wheel_dir = extpath.parent

        package_name = ext.name
        (wheel_dir / package_name).mkdir(parents=True, exist_ok=True)

        extension_path = build_dir / 'lib' / extpath.name
        vortex_dir = build_dir / 'bin'
        package_root = wheel_dir / ext.name

        # create link to allow import without placing binaries in site-packages
        (wheel_dir / f'{package_name}.pth').write_text(package_name)

        # copy Python extension and driver modules
        build_targets = [extension_path] + module_paths
        package_targets = [package_root / path.name for path in build_targets]
        for (src, dst) in zip(build_targets, package_targets):
            self.copy_file(src, dst)

        if deploy_dependencies:
            deploy_path_restriction = deploy_path_restriction or ''

            # deploy dependencies for extension and modules
            cmake_dependency_args = [
                self.cmake_command,
                f'-DBUILD_TARGET_PATHS={";".join([str(p) for p in build_targets])}',
                f'-DPACKAGE_TARGET_PATHS={";".join([str(p) for p in package_targets])}',
                f'-DVORTEX_DIR={vortex_dir}',
                f'-DPACKAGE_ROOT={package_root}',
                f'-DDEPLOY_PATH_RESTRICTION={deploy_path_restriction}',
                '-P', str(ext.sourcedir / 'cmake' / 'deploy_dependencies.cmake'),
            ]
            subprocess.check_call(cmake_dependency_args)

        if generate_stubs:
            # generate stubs

            print('-- Importing Module', flush=True)
            subprocess.check_call([sys.executable, '-c', 'import vortex; print("imported")'], cwd=package_root)

            print('-- Generating Stubs', flush=True)
            subprocess.check_call([sys.executable, '-c', 'from mypy.stubgen import main; main()', '-p', ext.name, '-o', str(wheel_dir), '-v', '--ignore-errors', '--include-docstrings'], cwd=package_root)

try:
    numpy_requirement = os.environ['VORTEX_NUMPY_REQUIREMENT'].strip()
except KeyError:
    # detect numpy version and match requirements from pyproject.toml
    import numpy
    version = Version(numpy.__version__)
    if version.major < 2:
        # use version that oldest-supported-numpy installed
        numpy_requirement = f'numpy>={numpy.__version__}'
    elif 2 <= version.major < 3:
        # any 2.x version is compatible
        numpy_requirement = 'numpy>=2.0,<3'
    else:
        # do not specify version
        numpy_requirement = 'numpy'

package_properties = dict(
    description='A library for building real-time OCT engines in C++ or Python.',
    long_description=Path(root / 'README.rst').read_text(),

    author='Mark Draelos',
    author_email='contact@vortex-oct.dev',

    license='BSD-3-Clause',
    url='https://www.vortex-oct.dev/',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    keywords=[
        'optical coherence tomography',
    ],

    install_requires=[
        numpy_requirement,
    ],

    platforms=[os.name],

    zip_safe=False,
)

if __name__ == '__main__':

    version = detect_vortex_version()

    try:
        suffix = os.environ['VORTEX_BUILD_SUFFIX'].strip()
    except KeyError:
        # determine CUDA version suffix
        if 'cuda' in _decode_features():
            cuda_version = _detect_cuda_version()
            suffix = f'-cuda{cuda_version.major}{cuda_version.minor}'

            print(f'-- Detected CUDA {cuda_version} for package suffix {suffix}', flush=True)
        else:
            suffix = ''

    setup(
        name='vortex-oct' + suffix,
        version=version,

        **package_properties,

        cmdclass=dict(build_ext=CMakeBuild),
        ext_modules=[CMakeExtension('vortex', root, version)],
    )
