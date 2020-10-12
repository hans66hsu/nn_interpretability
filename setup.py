import setuptools

with open('requirements.txt') as fp:
    install_requires = fp.read()

setuptools.setup(
    name='nn_interpretability',
    version='1.0.0',
    packages=['nn_interpretability'],
    install_requires=install_requires,
    python_requires='>=3.6',
)