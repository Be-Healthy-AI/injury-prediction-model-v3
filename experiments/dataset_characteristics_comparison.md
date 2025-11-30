# Dataset Characteristics Comparison

**Date:** 2025-11-27

## Number of Unique Players

| Dataset | Number of Players |
|---------|-------------------|
| Training | 225 |
| In-Sample Validation | 225 |
| Out-of-Sample Validation | 201 |

## Player Overlap

| Overlap | Number of Players |
|---------|-------------------|
| Training ∩ In-Sample | 225 |
| Training ∩ Out-of-Sample | 201 |
| In-Sample ∩ Out-of-Sample | 201 |
| All Three | 201 |

## Career Length Distribution

| Dataset | Count | Mean | Median | Std Dev | Min | Max | Q25 | Q75 |
|---------|-------|------|--------|---------|-----|-----|-----|-----|
| Training | 225 | 5.37 | 5.24 | 2.41 | 0.39 | 12.19 | 3.60 | 7.06 |
| In-Sample Validation | 225 | 5.37 | 5.23 | 2.50 | 0.18 | 12.07 | 3.41 | 6.85 |
| Out-of-Sample Validation | 201 | 9.03 | 8.38 | 4.24 | 0.99 | 24.52 | 6.10 | 11.74 |

## Career Length Comparison

- **Training mean:** 5.37 years
- **In-Sample Validation mean:** 5.37 years (+0.00 vs training)
- **Out-of-Sample Validation mean:** 9.03 years (+3.66 vs training)

⚠️ **SIGNIFICANT DIFFERENCE:** Out-of-sample players have +3.66 years different career length on average.
This could contribute to distribution shift and model performance issues.

## Date Ranges

| Dataset | First Date | Last Date | Span (years) |
|---------|------------|-----------|--------------|
| Training | 2001-10-28 00:00:00 | 2025-06-30 00:00:00 | 23.7 |
| In-Sample Validation | 2001-10-15 00:00:00 | 2025-06-29 00:00:00 | 23.7 |
| Out-of-Sample Validation | 2025-07-01 00:00:00 | 2025-11-08 00:00:00 | 0.4 |
