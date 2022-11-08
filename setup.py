from setuptools import setup

with open('requirements.txt', 'r') as f:
    install_deps = f.readlines()

setup(
    name='vxpy_tools',
    version='0.0.1',
    packages=['vxpy-tools', 'vxpy-tools.summarize'],
    url='',
    license='',
    install_requires=install_deps,
    author='thladnik',
    author_email='',
    description=''
)
