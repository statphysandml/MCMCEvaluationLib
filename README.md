MCMCEvaluationLib
=================

The MCMCEvaluationLib is a Python library that implements important algorithms for an evaluation of results of a Markov Chain Monte Carlo algorithms. This includes the computation of expectation values, error estimation and the computation of the autocorrelation time. The library is used by the MCMCSimulationLib (https://github.com/statphysandml/MCMCSimulationLib) which provides generic code for Markov Chain Monte Carlo algorithms in C++. Further, the library enables an easy loading of the simulation data and provides a convenient way to convert the data into a pytorch dataset.

The library currently consists of the following main modules:

- **loading** - Class for loading the simulation data. The simulation data is assumed to be stored columns wise in a .txt file. The ConfigurationLoader supports a simultaneous loading from multiple files and a piecewise loading (chunk by chunk).
- **modes** - Code that is used by the C++ MCMCSimulationLib for the computation of expectation values and further important results of a Markov Chain Monte Carlo Simulation.
- **pytorch** - Classes for a generation of a dataset consisting of samples/configurations of a Markov Chain Monte Carlo simulation. A possible batch-wise loading of the data boosts the computationl performance of the loading process. Samples of simulations with different hyperparameters can be mixed and loaded simultaneously.

Examples
--------

Examples to the different python modules can be found here: https://github.com/statphysandml/MCMCSimulationLib/tree/master/examples/python_scripts/examples. Simulation results of the Ising model are discussed as a more detailed example here: https://github.com/statphysandml/MCMCSimulationLib/blob/master/examples/jupyter_notebooks/ising_model_cheat_sheet.ipynb. The example covers almost all functionalities of the library and shows additionally possible ways to make use of the pystatplottools library (https://github.com/statphysandml/pystatplottools) to analyse the data in more detail.

Integration
-----------

So far, the library needs to be build locally. This can be done by

```bash
cd path_to_mcmcevaluationlib/

python setup.py sdist
pip install -e .
```

For virtual enviroments, the library needs to be activate beforehand.

After this step, the different modules of the library can be used, for example, by

```python
import mcmctools

from mcmctools.pytorch.data_generation.configdatagenerator import ConfigDataGenerator
```

Dependencies
------------

- matplotlib
- numpy
- pandas
- scipy
- pytorch
- (jupyter lab)

- pystatplottools (https://github.com/statphysandml/pystatplottools)

Projects using the MCMCEvaluationLib
----------------------------------

- MCMCSimulationLib (https://github.com/statphysandml/MCMCSimulationLib)
- LatticeModelImplementations (https://github.com/statphysandml/LatticeModelImplementations)

Support and development
----------------------

For bug reports/suggestions/complaints please file an issue on GitHub.

Or start a discussion on our mailing list: statphysandml@thphys.uni-heidelberg.de

