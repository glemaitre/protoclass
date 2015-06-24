from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'scikit-learn',
    #'SimpleITK',
    'UnbalancedDataset'
    ]

setup(name='protoclass',
      version='0.1',
      description='Python module for fast prototyping.',
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          ],
      author='Guillaume Lemaitre',
      author_email='guillaume.lemaitre@udg.edu',
      url='https://github.com/glemaitre/protoclass',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )
