from setuptools import setup, find_packages

from karas.version import __version__

print(find_packages())

setup(
    name="karas",
    version=__version__,
    keywords=["karas", "pytorch"],
    description="trainer for pytorch",
    long_description="A chainer-like trainer framework for pytorch users.",
    license="MIT Licence",

    url="http://karas.com",
    author="Xiaokang Tu",
    author_email="xiaokang.tu@qq.com",

    packages=find_packages(where='.'),
    include_package_data=True,
    zip_safe=False,
    platforms="any",
    install_requires=['six', 'torch', 'numpy', 'tensorboard', 'tensorboardx<=1.5'],
)
