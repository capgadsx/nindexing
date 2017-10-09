from setuptools import setup, find_packages
from setuptools.command.install import install
from shutil import copyfile

import os
import glob
import sys
import subprocess 

def build_cfunctions():
    print('Building cfunctions module')
    working_dir = os.getcwd()
    module_dir = os.path.join(working_dir, 'nindexing/cfunctions')
    os.chdir(module_dir)
    subprocess.call(['python','setup.py', 'build'])
    for builded_lib in glob.glob("build/lib.linux*/cfunctions*.so"):
        copyfile(builded_lib, os.path.join(module_dir, os.path.basename(builded_lib)))
    os.chdir(working_dir)

def setup_package():
    os.environ['CC'] = 'gcc'
    os.environ['CCX'] = 'g++'
    if 'build' in sys.argv[1::]:
        build_cfunctions()
    setup(
        name = "nindexing",
        version = "0.1",
        zip_safe = False,
        packages = find_packages(),
        include_package_data = True,
        setup_requires = ['numpy>=1.13.1'],
        install_requires = ['numpy>=1.13.1', 'astropy>=2.0', 'scikit-image>=0.13.0', 'dask==0.15.1', 'distributed==1.18.0']
    )

setup_package()