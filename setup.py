from setuptools import setup

setup(
    name='photospice_project',
    version='0.1',
    packages=['photospice'],
    package_dir={
        "photospice": "./src"
    },
    zip_safe=False
)