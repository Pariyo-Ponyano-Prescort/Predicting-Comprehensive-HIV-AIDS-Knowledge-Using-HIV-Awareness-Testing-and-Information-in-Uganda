
# Rainfall and Crop Suitability Analysis

A Python project for analysing monthly rainfall patterns across regions and assessing basic crop suitability using rainfall thresholds.

## Features

- Calculates annual and average monthly rainfall.
- Identifies the wettest and driest months.
- Detects rainfall peaks and classifies seasonal patterns.
- Screens rainfall levels for crop suitability, drought risk, and waterlogging risk.
- Compares regions using cosine similarity, Pearson correlation, and Euclidean distance.

## Requirements

- Python 3.10+
- NumPy
- SciPy

Install dependencies:

```
pip install numpy scipy
```

## Findings

The model helps identify rainfall differences between regions, detect potential rainy seasons, and compare monthly rainfall patterns. Crop rules provide a simple way to screen rainfall suitability.

## Limitations

- Results depend on the accuracy of the supplied rainfall data.
- Rainfall peak detection depends on the chosen prominence threshold.
- Crop suitability is based only on rainfall, not soil, temperature, or other growing conditions.
- One year of monthly data cannot fully describe long-term climate patterns.

