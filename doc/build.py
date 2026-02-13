from pathlib import Path
root = Path(__file__).parent.parent.resolve()

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(description='documentation builder', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--build', default=(root / 'build').as_posix(), help='directory for build files')
parser.add_argument('--output', type=str, help='directory for output ZIP')
parser.add_argument('--draft', action='store_true', default=False, help='build in draft mode without C++/Python reference')
args = parser.parse_args()

import importlib.util
spec = importlib.util.spec_from_file_location('setup', root / 'setup.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
version = setup.detect_vortex_version()

# prepare build environments
import os
new_env_vars = dict(
    PROJECT_SOURCE_DIR=root,
    PROJECT_BINARY_DIR=args.build,
    VORTEX_VERSION_STRING=version,
)
setup.dump_vars('Documentation Environment', new_env_vars)
env_vars = os.environ.copy()
env_vars.update(new_env_vars)

import sys
import subprocess
build_path = Path(args.build).resolve() / 'doc' / 'html'
build_path.mkdir(parents=True, exist_ok=True)
if not args.draft:
    cmd = ['doxygen', (root / 'doc' / 'Doxyfile').as_posix()]
    print('-- Doxygen:', cmd)
    subprocess.check_call(cmd, cwd=root, env=env_vars)

cmd = [
    sys.executable,
    '-m', 'sphinx.cmd.build',
    '--builder', 'dirhtml']
if args.draft:
    cmd += ['--tag', 'draft']
cmd += [
    '--doctree-dir', (Path(args.build) / 'doc' / '_sphinx').as_posix(),
    (root / 'doc').as_posix(),
    build_path.as_posix()
]
print('-- Sphinx:', cmd)
subprocess.check_call(cmd, cwd=root, env=env_vars)

if args.output:
    target = Path(args.output).resolve() / f'vortex-{version}-doc.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    print('-- Archiving:', target.name)

    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(target, 'w', ZIP_DEFLATED) as zf:
        for (dirpath, dirnames, filenames) in build_path.walk():
            for filename in filenames:
                src = dirpath / filename
                dst = src.relative_to(build_path)
                print('  adding:', dst)
                zf.write(src, dst)

    print(build_path, '->', target)
