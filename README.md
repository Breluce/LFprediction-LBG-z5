# z5_LFintegration

- LFint.py module: contains models for 2 versions of the luminosity function (Schechter and Douple-Power Law). All functions for predicting the count of LBG's detectable given survey area, magnitude limit, and redshift integration limits. Additional functions for converting SED flux to apparent magnitudes and performing k-corrections.
- LFparamscatolog.py module: script to produce pickle file for parameters of predicted luminosity functions as found in the literature. NOTE: contains specific values from Ono 2018, Harikane 2021 - if interested in other luminosity function predictions, script must be modified to produce new pkl files for LFplots.py
- LFplots.py module: script to read in luminosity function parameter predictions and produce a plot of all models 
