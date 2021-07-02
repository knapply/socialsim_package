from setuptools import setup

package_structure = [
      'socialsim',
      'socialsim.measurements',
      'socialsim.measurements.model_parameters',
      'socialsim.visualizations'
]

package_requirements = [
    'pandas>=0.24.2',
    'matplotlib',
    'scipy>=1.2.1',
    'numpy',
    'scikit-learn>=0.20.2',
    'fastdtw>=0.2.0',
    'tqdm>=4.31.1',
    'burst_detection>=0.1.0',
    'tsfresh>=0.11.2',
    'joblib>=0.13.2',
    'python-louvain>=0.6.1',
    'louvain',
    'cairocffi>=1.0.2',
    'kiwisolver',
    'cycler',
    'seaborn',
    'python-igraph',
    'future',
]


package_data = {
      'socialsim.measurements.model_parameters': ['best_model.pkl']
      }

setup(
    name='socialsim',
    version='0.4.3',
    requirements=package_requirements, 
    packages=package_structure,
    package_data=package_data,
    license='',
    url='',
    long_description='None',
    maintainer='SocialSim Team',
    maintainer_email='SocialSimAdmin@leidos.com',
    install_requires=package_requirements
    )
