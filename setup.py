from setuptools import find_packages, setup


setup(
    name='nn-lib-bench',
    author='Kemal Kurniawan',
    author_email='kemal@kkurniawan.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch ~=1.0',
    ],
    python_requires='~=3.6',
)
