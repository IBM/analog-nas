from setuptools import setup, find_packages

setup(
    name='analog-nasbench',
    version='0.1.3',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'analog-nasbench': ['data.anb'],
    },
    data_files=[('analog-nasbench', ['analognasbench/data.anb'])],
    install_requires=[
        'pandas',
        'numpy',
    ],
)