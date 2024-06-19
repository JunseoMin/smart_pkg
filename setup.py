# smart_pkg/setup.py
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['smart_pkg', 'smart_pkg.deep_sort', 'smart_pkg.deep_sort.deep', 'smart_pkg.deep_sort.sort', 'smart_pkg.deep_sort.utils'],
    package_dir={'': 'src'}
)

setup(**d)
