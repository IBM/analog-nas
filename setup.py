import os
import sys
import subprocess
from setuptools import setup, find_packages

# Check for python version
if sys.version_info.major != 3 or sys.version_info.minor < 7 or sys.version_info.minor > 9:
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. AnalogNAS requires Python '
        '3.7, 3.8 or 3.9' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )


cwd = os.path.dirname(os.path.abspath(__file__))

version_path = os.path.join(cwd, 'analogainas', '__version__.py')
with open(version_path) as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())

print('-- Building version ' + version)
print('-- Note: by default installs pytorch-cpu version (1.9.0), update to torch-gpu by following instructions from: https://pytorch.org/get-started/locally/')

setup(
    name='analogainas',
    version=version,
    description='AnalogAINAS: A modular and extensible Analog-aware Neural Architecture Search (NAS) library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='IBM Research',
    author_email='aihwkit@us.ibm.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Typing :: Typed',
    ],
    keywords=['NAS', 'analog', 'torch'],
    packages=find_packages(),
    python_requires='>=3.7',
    platforms=['Linux'],
    install_requires=requirements
)