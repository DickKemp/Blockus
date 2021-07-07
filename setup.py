from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='blokus',
    version='0.5.0',
    description='generates shapes in the spirit of blokus',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DickKemp/blokus',
    author='Dick Kemp',
    author_email='dickkemp14@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(),
    python_requires='>=3.6, <4'
)
