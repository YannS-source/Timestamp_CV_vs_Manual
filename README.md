# Timestamp validation, analysis code

Analysis code for "Validation of Computer Vision in the Operating Room for
Ambient Documentation" (JAMA Surgery, research letter). Computer vision (CV)
and manual EHR timestamps of patient entry and exit are compared against
human-annotated video ground truth in 290 surgical cases.

## Data

The data are not distributed with the code. To rerun the analysis, place the
data file in this folder (its name is set in `data_utils.py`). It has one row
per case, with these columns:

| Column | Contents |
|---|---|
| `ground_truth_wheels_in`, `ground_truth_wheels_out` | Patient entry and exit from video annotation |
| `rs_wheels_in`, `rs_wheels_out` | Entry and exit from computer vision (RoomState) |
| `emr_wheels_in`, `emr_wheels_out` | Entry and exit entered manually in the EHR |
| `Collapsed_Specialties` | Surgical specialty group |

Timestamps are full date-times in ISO 8601 format.

## Requirements

Python 3.13 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Run

```
python analysis.py    # every number reported in the paper -> results.txt
python figure1.py     # Figure 1 (panels A-D) -> figures/
```

`results.txt` and `figures/` contain the outputs used in the paper.

## Files

| File | Contents |
|---|---|
| `data_utils.py` | Loads the data and computes signed errors (timestamp minus ground truth, seconds) |
| `analysis.py` | All reported statistics, in the order they appear in the paper |
| `figure1.py` | Figure 1 as vector PDF, EPS and SVG, plus PNG preview and 600 dpi TIFF |
| `results.txt` | Output of `analysis.py` |
| `figures/` | Output of `figure1.py` (the TIFF is regenerated, not stored) |

## Analysis

Signed error is timestamp minus ground truth, in seconds, per method and
event. The primary analysis tests noninferiority of CV to manual entry on
the difference in mean absolute error (MAE), with a prespecified margin of
30 seconds: CV is noninferior if the 1-sided 95% upper bound of a
10,000-resample paired bootstrap of the MAE difference is below the margin.
A 1-sided paired Wilcoxon signed-rank test of the same hypothesis is used as
a supportive analysis.

Agreement of each method with ground truth is described with Bland-Altman
statistics (bias, 95% limits of agreement, fraction of cases within 1
minute), and each method's bias is tested with a 1-sample Wilcoxon test.

Stability over the day is assessed with Spearman correlations of absolute
error against the hour of patient entry, and with MAE by shift block (AM, PM,
and overnight). Manual error in the overnight block is compared with the
other shifts with a Mann-Whitney test.
Exit timing relative to each method's own entry timing is tested within case
with a Wilcoxon signed-rank test.
