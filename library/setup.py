import os
import sys
import platform
import subprocess
import multiprocessing

from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Setuptools calls cmake, which takes care of further
# C++ dependency resolution and the extension build.

# Inspired by:
# https://gist.github.com/hovren/5b62175731433c741d07ee6f482e2936
# https://www.benjack.io/2018/02/02/python-cpp-revisited.html
# Thanks a lot, very helpful :)


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):

    def run(self):

        try:
            _ = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        # preparing cmake arguments

        cmake_args = [
            ('-DCMAKE_LIBRARY_OUTPUT_DIRECTORY='
                + os.path.abspath(self.build_temp)),
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        # determining build type

        cfg = 'Debug' if self.debug else 'Release'
        self.build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # parallelize compilation, if using make files

        cpu_count = multiprocessing.cpu_count()
        self.build_args += ['--', '-j{}'.format(cpu_count)]

        # additional flags

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
                                env.get('CXXFLAGS', ''),
                                self.distribution.get_version())
        os.makedirs(self.build_temp, exist_ok=True)

        # call cmake to configure and build

        cmake_dir = Path(__file__).absolute().parent
        subprocess.check_call(['cmake', str(cmake_dir)] + cmake_args,
                              cwd=self.build_temp,
                              env=env)

        cmake_cmd = ['cmake', '--build', '.'] + self.build_args

        subprocess.check_call(cmake_cmd, cwd=self.build_temp)

        # move from temp. build dir to final position

        for ext in self.extensions:

            build_temp = Path(self.build_temp).resolve()
            dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            source_path = build_temp / self.get_ext_filename(ext.name)

            dest_directory = dest_path.parents[0]
            dest_directory.mkdir(parents=True, exist_ok=True)

            self.copy_file(source_path, dest_path)


setup(
    name='tpl',
    version='0.1.0',
    description='Trajectory planning/optimization algorithms library',
    url='https://mrm-git.e-technik.uni-ulm.de/ruof/trajectory_planning',
    author='Jona Ruof',
    author_email='jona.ruof@uni-ulm.de',
    packages=find_packages(include=['tpl', 'tpl.*']),
    package_data={'tpl': ['optim/templates/*']},
    ext_modules=[CMakeExtension("tplcpp")],
    zip_safe=False,
    python_requires=">=3.8",
    cmdclass=dict(build_ext=CMakeBuild),
    include_package_data=True,
    install_requires=['sympy>=1.11.0', 'numpy==1.23.0', 'imviz', 'numba', 'scipy'],
    scripts=['tpl/simulation/tplsim', 'tpl/gui/tplgui'],
)
