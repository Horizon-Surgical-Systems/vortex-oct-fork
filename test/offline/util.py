import pytest

def get_compute_capability(required):
    cupy = get_cupy_or_skip()

    try:
        result = int(cupy.cuda.Device().compute_capability)
    except Exception as e:
        pytest.skip(f'cannot determine compute capability: {e!r}')
    else:
        if result < required:
            pytest.skip(f'GPU compute capability is insufficient: {result} < {required}')

    return cupy

def get_cupy_or_skip():
    try:
        import cupy
    except ImportError as e:
        pytest.skip(f'CuPy fails to import: {e!r}')
    else:
        return cupy
