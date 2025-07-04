from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext
setup(
    name='pcdet',
    ext_modules=[
        # CUDAExtension(
        #     name='my_cuda_extension',
        #     sources=['my_kernel.cu'],
        # ),
        make_cuda_ext(
            name='flattened_window_cuda',
            module='pcdet.ops.win_coors',
            sources=['src/flattened_window.cpp',
                     'src/flattened_window_kernel.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
