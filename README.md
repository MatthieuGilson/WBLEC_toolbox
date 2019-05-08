**These scripts in Python 3.6 are a small package to analyze fMRI signals. It is a collaborative work with Andrea Insabato, Vicente Pallarés, Gorka Zamora-López and Nikos Kouvaris. If you use this toolbox, please cite the relevant papers below.**

## Whole-brain linear effective connectivity (WBLEC) estimation 

The script *ParameterEstimation.py* calculates the spatiotemporal functional connectivity for each session (or run) and subject from the BOLD time series. Then, it calls the model optimization (function in *WBLECmodel.py*) and stores the model estimates (effective connectivity matrix embedded in the Jacobian J and input variances Sigma) in an array.
The data are:
- BOLD time series in *ts_emp.npy*
- structural connectivity in *SC_anat.npy*
- ROI labels in *ROI_labels.npy*

## Classification

The script *Classification.py* compares the performances of two classifiers (multinomial linear regressor and 1-nearest-neighbor) in identifying subjects from EC taken as a biomarker.

The script *FeatureSelection.py* performs recursive feature elimination to identify the most informative EC connections for the rest-movie classification. It compares the resulting ranking for EC connections to the p-values obtained with statistical testing.

## References

Data are from: Gilson M, Deco G, Friston K, Hagmann P, Mantini D, Betti V, Romani GL, Corbetta M. Effective connectivity inferred from fMRI transition dynamics during movie viewing points to a balanced reconfiguration of cortical interactions. 
Neuroimage 2018, 180: 534-546; http://doi.org/10.1016/j.neuroimage.2017.09.061.
See also: Hlinka J, Palus M, Vejmelka M, Mantini D, Corbetta M. Functional connectivity in resting-state fMRI: is linear correlation sufficient? Neuroimage 2011, 54:2218-2225; http://doi.org/10.1016/j.neuroimage.2010.08.042

Model optimization is described in: Gilson M, Moreno-Bote R, Ponce-Alvarez A, Ritter P, Deco G. Estimation of Directed Effective Connectivity from fMRI Functional Connectivity Hints at Asymmetries of Cortical Connectome. PLoS Comput Biol 2016, 12: e1004762; http://dx.doi.org/10.1371/journal.pcbi.1004762

Classification procedure is described in: Pallarés V, Insabato A, Sanjuan A, Kühn S, Mantini D, Deco G, Gilson M. Subject- and behavior-specific signatures extracted from fMRI data using whole-brain effective connectivity. Neuroimage 2018, 178: 238-254; http://doi.org/10.1016/j.neuroimage.2018.04.070

The classification script uses the scikit.learn library (http://scikit-learn.org)

