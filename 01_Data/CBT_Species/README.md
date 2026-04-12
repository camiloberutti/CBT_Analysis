# Optimal sampling interval for characterisation of the circadian rhythm of body temperature in homeothermic animals using periodogram and cosinor analysis

[https://doi.org/10.5061/dryad.1g1jwsv46](https://doi.org/10.5061/dryad.1g1jwsv46)

Five days of core body temperature from nine species of bird and mammal. The number of individuals per species varies (alpaca n = 22, cheetah n = 5, mouse n = 6, barnacle goose n = 9, Pekin duck n = 19, rabbit n = 36, rat n = 6, sheep n = 12, blue wildebeest n = 6).

## Description of the data and file structure

The date are provided as an excel file with one worksheet per species. Individual animals are in columns on each time. A time stamp is given in the first column of each worksheet. All of the data were calibrated against a certified mercury in glass thermometer before and after deployment.

## Metadata

The metadata in the JSON file was extracted from:
    Goh, G., Vesterdorf, K., Fuller, A., Blache, D., & Maloney, S. K. (2024).
    Optimal sampling interval for characterisation of the circadian rhythm of
    body temperature in homeothermic animals using periodogram and cosinor analysis.
    Ecology and Evolution, 14(4), e11243.
    https://doi.org/10.1002/ece3.11243

All body masses and sensor specifications are taken directly from the paper's
Methods §2.1.  Values are stored as reported (mean ± SD where available).

Keys match the Excel sheet names used in
"Five_days_of_Tc_data_in_nine_species.xlsx"
body_mass_kg_mean   : mean body mass in kg (as reported in paper)
body_mass_kg_sd     : SD of body mass in kg (null if not reported as SD)
body_mass_note      : free-text note on how mass was reported
n_individuals       : sample size used in the paper
species_latin       : binomial scientific name
taxon_class         : 'Mammalia' or 'Aves'
original_interval_min : sampling interval (min) of the data as provided
logger_model        : logger / sensor hardware used
logger_resolution_C : temperature resolution (°C) of the logger
logger_accuracy_C   : calibrated accuracy (°C); upper bound where "<" was stated
measurement_site    : where the sensor was implanted / placed
housing             : 'field' or 'captive' (lab/facility)
light_dark_cycle    : if captive, the L:D regime; null if field
reference           : key citation for the original temperature data

## Sharing/Access information

Five random days from each species were chosen for analysis. Total deployment in each species varied from weeks to months. Longer data sets, including the calibration data, are available from the authors.

## Code/Software

Raw data provided. No post-processing has been done, apart from the application of calibration to each logger.
