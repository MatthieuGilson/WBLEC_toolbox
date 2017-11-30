# WBLEC_toolbox

Whole-brain linear effective connectivity estimation. 

The script ParameterEstimation.py calculates the spatiotemporal functional connectivity for each session (or run) and subject from the BOLD time series. Then, it calls the model optimization (function in WBLECmodel.py) and stores the model estimates (effective connectivity matrix embedded in the Jacobian J and input variances Sigma) in an array.
The data are:
- BOLD time series in ts_emp.npy
- structural connectivity in SC_anat.npy
- ROI labels in ROI_labels.npy

**If you use this toolbox, please cite the relevant papers below. This script package is a collaborative work with Andrea Insabato, Vicente Pallares, Gorka Zamora-López and Nikos Kouvaris.

Data are from: Gilson M, Deco G, Friston K, Hagmann P, Mantini D, Betti V, Romani GL, Corbetta M. Effective connectivity inferred from fMRI transition dynamics during movie viewing points to a balanced reconfiguration of cortical interactions. 
Neuroimage 2017; doi.org/10.1016/j.neuroimage.2017.09.061

Model optimization is described in: Gilson M, Moreno-Bote R, Ponce-Alvarez A, Ritter P, Deco G. Estimation of Directed Effective Connectivity from fMRI Functional Connectivity Hints at Asymmetries of Cortical Connectome. PLoS Comput Biol 2016, 12: e1004762; dx.doi.org/10.1371/journal.pcbi.1004762

Classification procedure is described in: Pallares V, Insabato A, Sanjuan A, Kuehn S, Mantini D, Deco G, Gilson M. Subject- and behavior-specific signatures extracted from fMRI data using whole-brain effective connectivity. Biorxiv, doi.org/10.1101/201624

The classification script uses the scikit.learn library (http://scikit-learn.org)

