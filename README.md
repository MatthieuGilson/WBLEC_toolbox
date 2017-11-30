# WBLEC_toolbox

Whole-brain linear effective connectivity estimation. 

The script ParameterEstimation.py calculates the spatiotemporal functional connectivity for each session (or run) and subject from the BOLD time series. Then, it calls the model optimization (function in WBLECmodel.py) and stores the model estimates (effective connectivity matrix embedded in the Jacobian J and input variances Sigma) in an array.
The data are:
- BOLD time series in ts_emp.npy
- structural connectivity in SC_anat.npy
- ROI labels in ROI_labels.npy

Data are from: Gilson M, Deco G, Friston K, Hagmann P, Mantini D, Betti V, Romani GL, Corbetta M. Effective connectivity inferred from fMRI transition dynamics during movie viewing points to a balanced reconfiguration of cortical interactions. 
Neuroimage 2017; doi.org/10.1016/j.neuroimage.2017.09.061

Model details in: Gilson M, Moreno-Bote R, Ponce-Alvarez A, Ritter P, Deco G. Estimation of Directed Effective Connectivity from fMRI Functional Connectivity Hints at Asymmetries of Cortical Connectome. PLoS Comput Biol 2016, 12: e1004762; dx.doi.org/10.1371/journal.pcbi.1004762

The classification script uses the scikit.learn library (http://scikit-learn.org)

