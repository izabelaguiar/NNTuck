# NNTuck
This github contains code and definitions for running NNTuck as described in A factor model for multilayer network interdependence. The `NNTuck_tutorial.pynb` steps through the methods discussed and performs the NNTuck on a multilayer social support network for a village surveyed in [Banerjee et al.'s (2013) Diffusion of Microfinance](https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:1902.1/21538).

The code depends on the following packages and the version number for which it is reproducible is noted. `numpy` (`version 1.22.2`), `tensorly` (`version 0.5.1`), `sklearn` (`version 0.23.2`), and `matplotlib` (`version 3.3.2`). When sweeping over parameters $K$ and $C$ in NNTuck it is most efficient to run the sweep in parallel, which depends on `joblib` (`version 1.0.1`) and on `os` to make sure the parallel runs don't use too much CPU.

