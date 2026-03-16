# APRScast

**End-to-end energy time series analysis: anomaly detection → robust forecasting → uncertainty quantification**

APRScast is a Python package for energy time series analysis, powered by the **Adaptive Penalized Regression Spline (APRS)** — a change point detection method that replaces the numerically unstable falling factorial basis of Trend Filtering with the stable B-spline basis, integrated with spatially adaptive penalties.

## Why APRScast?

Energy time series (electricity demand, renewable generation, grid frequency) are noisy, unevenly sampled, and full of structural breaks. Existing forecasting tools often ignore these issues, leading to poor predictions when demand patterns shift.

APRScast solves this with a **diagnose-then-predict** pipeline:

```
Raw Energy Data
    │
    ▼
┌─────────────────────────┐
│  Stage 1: Detection     │  ← APRS (+ PELT, TF, NOT for comparison)
│  Change points &        │
│  anomaly identification │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Stage 2: Forecasting   │  ← Regime-aware prediction
│  Based on current       │
│  regime only            │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Stage 3: Uncertainty   │  ← Prediction intervals
│  Quantification         │
└─────────────────────────┘
```

## Key Features

- **APRS Engine**: B-spline based change point detection with adaptive weights — dramatically fewer false positives than Trend Filtering or Fused Lasso
- **Multiple Methods**: Compare APRS against PELT, Trend Filtering, NOT, and Binary Segmentation
- **Arbitrary Degree**: Support for piecewise constant (d=0), linear (d=1), quadratic (d=2), and cubic (d=3) splines
- **Uneven Spacing**: Handles irregularly sampled time series without interpolation
- **Energy-Focused**: Built-in data loaders for public energy datasets (weather API, electricity consumption)
- **Uncertainty Quantification**: Prediction intervals via conformal prediction and quantile regression

## Installation

```bash
pip install aprscast
```

## Quick Start

```python
import aprscast as ac

# Load energy time series
data = ac.load_example("electricity_demand")

# Run end-to-end pipeline
result = ac.pipeline(data, target="demand", freq="1H")

# Inspect results
print(result.changepoints)   # Detected structural breaks
print(result.forecast)       # Next 24h forecast
print(result.intervals)      # 90% prediction intervals

# Visualize
result.plot()
```

## Comparison with Other Methods

```python
# Compare APRS against other change point detection methods
comparison = ac.compare_methods(
    data,
    methods=["aprs", "trend_filtering", "pelt", "not"],
    degree=1
)
comparison.summary()
comparison.plot()
```

## Academic References

APRScast implements the methodology from:

1. **Lee, D.-Y.**, Bak, K.-Y., & Jhong, J.-H. (2024). Adaptive Weighted Total Variation Penalty for Precise Change Point Detection. *Australian & New Zealand Journal of Statistics*. (d=0, piecewise constant)

2. **Lee, D.-Y.**, Bak, K.-Y., & Jhong, J.-H. (2025). Change Point Detection in Unevenly Spaced Time Series via Generalized Adaptive Penalized Regression Splines. *(Under review)*. (General d≥0, B-splines)

## Project Structure

```
APRScast/
├── src/aprscast/
│   ├── __init__.py          # Package API
│   ├── detection.py         # APRS + comparison methods
│   ├── bspline.py           # B-spline basis construction
│   ├── forecasting.py       # Regime-aware forecasting
│   ├── uncertainty.py       # Prediction intervals
│   ├── pipeline.py          # End-to-end pipeline
│   └── visualization.py     # Plotting utilities
├── tests/                   # Unit tests
├── examples/                # Jupyter notebooks & tutorials
├── data/                    # Example datasets
├── docs/                    # Documentation
├── pyproject.toml           # Package configuration
├── LICENSE
└── README.md
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
