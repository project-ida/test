---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="220d2b82-5a92-4460-860c-2ef664b8c4af" -->
<a href="https://colab.research.google.com/github/project-ida/test/blob/main/Effects_of_Chamber_on_Neutron_Detection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/test/blob/main/Effects_of_Chamber_on_Neutron_Detection.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

```python id="cbb61a57"
# Libraries and helper functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image, display, Video, HTML

from scipy.signal import find_peaks
from scipy.stats import ttest_ind
from datetime import datetime


# Use our custom helper functions
# - process_data
# - plot_panels
# - plot_panels_with_scatter
# - print_info
from libs.helpers import *
```

```python colab={"base_uri": "https://localhost:8080/"} id="qLRrQRiKIdjK" outputId="df48b9da-a33c-423b-c1a4-1e46105f4572"
# RUN THIS IF YOU ARE USING GOOGLE COLAB
import sys
import os
!git clone https://github.com/project-ida/arpa-e-experiments.git
sys.path.insert(0,'/content/arpa-e-experiments')
os.chdir('/content/arpa-e-experiments')
```

```python colab={"base_uri": "https://localhost:8080/"} id="l9ZhSpdGtIfp" outputId="a7757d7f-b9e3-44c1-d62a-48660f531a22"
from google.colab import drive
drive.mount('/content/drive')
```

<!-- #region id="9ea666fa" -->
# Neutron Source in and out of Chamber Anlysis

This experiment involves analyzing the effects of a steel chamber on neutron detection by placing a neutron source in a graphite tunnel both inside and outside the chamber. Below are the two setups used for the experiment:

Online Data Dashboard at: https://lenr.mit.edu/load-panel.php?filename=he3-detectors-steeltest2
<!-- #endregion -->

<!-- #region id="30bb81bc" -->
## Setup 1: Neutron source **outside** the steel chamber in a graphite tunnel
The neutron source is placed in the graphite tunnel.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 437} id="47621fd1" outputId="b7397c22-fb0f-49d1-f4fd-be511826e024"
display(Image(filename="/content/Neutron source in graphite tunnel.png", width=500, height=400))
```

<!-- #region id="13003df3" -->
## Setup 2: Neutron source **inside** the steel chamber in a graphite tunnel
In this setup, the neutron source is placed in the steel chamber, within the graphite tunnel.This setup aims to analyze how the steel chamber affects neutron detection.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 437} id="ca1ca48a" outputId="283d4b78-169f-4804-d6d6-2c60e0ea5203"
display(Image(filename="/content/Neutron source in chamber in graphite tunnel.png", width=500, height=400))
```

<!-- #region id="WV269CAaIC3w" -->
# Reading the Raw Data

<!-- #endregion -->

<!-- #region id="J7UsdZ30IxG_" -->
## Counts per minute above 50 threshold


<!-- #endregion -->

```python id="Kr0YO2nBHv8w"
# Read the counts per minute data
CountsPerMinute = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=19f&random=35',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="tPHaO7lRIWzV" outputId="784c14ed-065e-44ab-aec5-bba5463e4e14"
# Print out basic description of the data, including any NaNs
print_info(CountsPerMinute)
```

<!-- #region id="ozpASpPyLg2Y" -->
We are only interested in the counts above the 50 thrshold so we will drop the data with higher thresholds. Since we'll only be interested in `ch50-1000`, we'll rename it to make plotting a bit easier later.
<!-- #endregion -->

```python id="BQFjxq6OLmzT"
CountsPerMinute.rename(columns={'Counts ch50-1000': 'Counts Per Minute (CPM)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="NCuDkXD8TtyO" outputId="175a4408-b0ad-476c-868d-e8e7b4615fa3"
# Plot the neutron counts over time
plt.figure(figsize=(10, 4))
plt.plot(CountsPerMinute.index, CountsPerMinute['Counts Per Minute (CPM)'], label='Counts Per Minute (CPM)', marker='', linestyle='-')
plt.title('Neutron Counts Over Time')
plt.xlabel('Time')
plt.ylabel('Counts Per Minute')
plt.legend()
plt.grid(True)
plt.show()
```

<!-- #region id="AQvIrVINYRkR" -->
## Comparing the Peaks
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 619} id="6lGwivolZP0Z" outputId="0d83833a-e82f-4a6b-f1f5-28704845194d"
# Extract the counts column
counts = CountsPerMinute['Counts Per Minute (CPM)'].values

# Smooth the data using a rolling mean (optional, to reduce noise)
window_size = 10  # Adjust window size as needed
smoothed_counts = np.convolve(counts, np.ones(window_size) / window_size, mode='same')

# Detect edges (start and end of peaks)
threshold = 50  # Adjust based on your data
edges = np.where(smoothed_counts > threshold)[0]

# Ensure edges are well-defined and identify ranges
if len(edges) > 0:
    # Split the detected edges into two distinct ranges
    mid_point = (edges[-1] - edges[0]) // 2 + edges[0]  # Midpoint between first and last edge
    tunnel_edges = edges[edges <= mid_point]  # First range for tunnel
    chamber_edges = edges[edges > mid_point]  # Second range for chamber

    if len(tunnel_edges) > 0 and len(chamber_edges) > 0:
        tunnel_range = (tunnel_edges[0], tunnel_edges[-1])
        chamber_range = (chamber_edges[0], chamber_edges[-1])

        tunnel_data = counts[tunnel_range[0]:tunnel_range[1]]
        chamber_data = counts[chamber_range[0]:chamber_range[1]]

        print(f"Tunnel Data Range: Start={tunnel_range[0]}, End={tunnel_range[1]}")
        print(f"Chamber Data Range: Start={chamber_range[0]}, End={chamber_range[1]}")

        # Plot the signal with detected regions
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_counts, label='Smoothed Counts', linestyle='-', color='blue')
        plt.axvspan(tunnel_range[0], tunnel_range[1], color='green', alpha=0.3, label='Tunnel Data')
        plt.axvspan(chamber_range[0], chamber_range[1], color='orange', alpha=0.3, label='Chamber Data')
        plt.title('Neutron Counts with Detected Regions')
        plt.xlabel('Index')
        plt.ylabel('Counts Per Minute')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("Failed to define clear tunnel or chamber ranges. Check edge detection logic.")
else:
    print("No significant edges found!")
```

<!-- #region id="oUzIzMZ7a7Qp" -->
## Peak Statistics
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 565} id="KJqgPeundVw2" outputId="50e3a2aa-21f0-4ed3-debc-c5a459e8e045"
plt.figure(figsize=(8, 6))
plt.boxplot([tunnel_data, chamber_data], labels=['Tunnel', 'Chamber in Tunnel'])
plt.title('Comparison of Neutron Counts')
plt.ylabel('Counts Per Minute')
plt.grid(True)
plt.show()
```

<!-- #region id="oRDLKlL8edq-" -->
We notice a few outliers that may disrupt our data so we will filter them out:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 565} id="nJNQATMsexcj" outputId="5762e1d1-5cc1-4416-a8e1-7ceefd69bf61"
'''

def remove_outliers(data, threshold=1):
    # Compute mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    filtered_data = data[np.abs(z_scores) <= threshold]
    return filtered_data

filtered_tunnel_data = remove_outliers(tunnel_data)
filtered_chamber_data = remove_outliers(chamber_data)
'''

def remove_ramp(data):
  return data[2:-2]

filtered_tunnel_data = remove_ramp(tunnel_data)
filtered_chamber_data = remove_ramp(chamber_data)

plt.figure(figsize=(8, 6))
plt.boxplot([filtered_tunnel_data, filtered_chamber_data], labels=['Tunnel', 'Chamber in Tunnel'])
plt.title('Comparison of Neutron Counts')
plt.ylabel('Counts Per Minute')
plt.grid(True)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Td6aIrlUa9x9" outputId="8aed2c63-b6f5-4fc1-a98a-b0f961006c47"
def compute_statistics(data, label):
  stats = {'Mean': np.mean(data),
            'Standard Deviation': np.std(data),
            'Median': np.median(data),
            'Variance': np.var(data),
            'Minimum': np.min(data),
            'Maximum': np.max(data),
            'Count': len(data)
        }
  print(f"\n{label} Statistics:")
  for key, value in stats.items():
      print(f"{key}: {value:.2f}")
  return stats
# Compute statistics for tunnel and chamber data

tunnel_stats = compute_statistics(filtered_tunnel_data, 'Tunnel Data')
chamber_stats = compute_statistics(filtered_chamber_data, 'Chamber Data')
```

<!-- #region id="p-H2HyeXbZ4u" -->
Let us now examine whether the difference between the two peaks is more significant that the standard deviation. We can start by doing a T-test
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="r1fhUMVVbfX5" outputId="87fea9d0-2848-4f03-cd81-8df661939455"
t_stat, p_value = ttest_ind(filtered_tunnel_data, filtered_chamber_data, equal_var=False)  # Use Welch's t-test if variances are unequal
print("\nT-Test Results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
```

<!-- #region id="AOIILiyPdr7c" -->
The standard significance level set in a t-test is 5%. We see here that our P-Value ~ 12.35% is above below the significance level. Hence, the means of the tunnel and chamber data are not significantly different.
<!-- #endregion -->

<!-- #region id="5xw8ImvZkZl_" -->
# Estimating our Detector Efficiency
<!-- #endregion -->

<!-- #region id="xRv2UVa9kxme" -->
We know that our source strength was 17.3 nCi on September 4th 2024. This measurement was taken on December 8th 2024, starting at 18:30:14.

Given the half-life of our source, Cf-252, which is 2.647 years, we can estimate what our source strength was the day of the experiment: 16.16094 nCi.

Furthermore, we know that 1 µCi = 3.7×10^4 disintegrations per second = 2.22×10^6 disintegrations per minute (dpm). Hence 1nCi = 37 dps = 2.22*10^3 dpm

Finally, "252Cf disintegrates by α emissions mainly to the 248Cm ground state level, and by spontaneous fission for 3,086%" (https://inis.iaea.org/search/search.aspx?orig_q=RN%3A45014763&utm)
<!-- #endregion -->

```python id="7ofMb4ofkw0k"
A0 = 17.3 # Initial activity in nCi
half_life = 2.647  # in years
decay_constant = np.log(2) / half_life
spontaneous_fission_fraction = 0.03086  # Fraction of disintegrations resulting in fission
neutrons_per_fission = 3.7573   # Average neutrons emitted per fission (± 0.0056)
conversion_factor = 37  # Conversion factor from nCi to dps (disintegrations per second)

date_initial = datetime(2024, 9, 4)  # activity meansurment taken September 4, 2024
date_experiment = datetime(2024, 12, 8)  # Experiment date: December 8, 2024
time_difference = (date_experiment - date_initial).days / 365.25

At = A0 * np.exp(-decay_constant * time_difference) # Activity on the day of the experiment
dps_t = At*conversion_factor
neutrons_ps_t = dps_t * spontaneous_fission_fraction
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ts7F-QGDl4XL" outputId="64675685-9f90-4a21-b331-37ac50f7c55b"
neutrons_pm_t = neutrons_ps_t * 60
# Experiment-specific data
neutrons_detected_per_minute =  np.mean(filtered_tunnel_data) # Total counts detected during the experiment

# Setup efficiency
detector_efficiency = neutrons_detected_per_minute /neutrons_pm_t

# Results
print(f"Expected neutron emissions per minute: {neutrons_pm_t:.2f} neutrons per second")
print(f"Measured neutrons per minute: {neutrons_detected_per_minute:.2f} neutrons")
print(f"Detector efficiency: {detector_efficiency:.6f} ({detector_efficiency * 100:.2f}%)")
```

<!-- #region id="kWBfSg8hGZlp" -->
In this computation, we assumed that our detectors communaly cover the sample's entire solid angle. In reality, this is not true as the gaps between each tube causes gaps in the solid angle. Hence, this efficiency, of 11.68% corresponds to the efficiency of our setup as a whole rather than that of each liquid scintillation detector.
<!-- #endregion -->

```python id="0leh-0IhO-Mx"

```

<!-- #region id="H_MLTM9Cr61H" -->
# Comparing the Neutron Tubes

Let us now have a look at each individual tube to compare their count rates.
<!-- #endregion -->

```python id="OLAzV4-CsSZK"
# Read the counts per minute data for each channel 0 - 7

CountsPerMinute_0 = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=1f&random=25',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)

CountsPerMinute_1 = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=3f&random=7',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)

CountsPerMinute_2 = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=5f&random=68',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)

CountsPerMinute_3 = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=7f&random=27',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)

CountsPerMinute_4 = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=9f&random=47',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)

CountsPerMinute_5 = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=11f&random=90',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)

CountsPerMinute_6 = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=13f&random=46',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)

CountsPerMinute_7 = pd.read_csv(
    'https://lenr.mit.edu/call-rscript.php?filename=he3-detectors-steeltest2&graphno=15f&random=23',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="mlHjjSqbucKH" outputId="1ef25480-2fb4-48b2-bdd0-f4eff3ff8755"
means = np.array([CountsPerMinute_0.mean(), CountsPerMinute_1.mean(), CountsPerMinute_2.mean(), CountsPerMinute_3.mean(),
         CountsPerMinute_4.mean(), CountsPerMinute_6.mean(), CountsPerMinute_7.mean()])

worst_efficiency = np.where(means == means.min())

means
```

```python id="RlDQJp3GvXrJ"

```
