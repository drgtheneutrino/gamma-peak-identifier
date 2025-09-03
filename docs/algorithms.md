# Algorithms Documentation

Mathematical methods and theoretical background for gamma spectroscopy analysis.

## Table of Contents

1. [Peak Detection Algorithms](#peak-detection-algorithms)
2. [Peak Fitting Models](#peak-fitting-models)
3. [Background Estimation](#background-estimation)
4. [Energy Calibration](#energy-calibration)
5. [Uncertainty Propagation](#uncertainty-propagation)
6. [Quality Metrics](#quality-metrics)
7. [References](#references)

## Peak Detection Algorithms

### 1. Local Maxima Method (SciPy)

The default peak detection uses SciPy's `find_peaks` algorithm with statistical filtering.

**Algorithm:**

1. **Find local maxima**: A point is a local maximum if `counts[i] > counts[i-1]` and `counts[i] > counts[i+1]`

2. **Apply prominence filter**: Prominence is the minimum height descent to reach higher ground
   ```
   prominence = peak_height - min(left_base, right_base)
   ```

3. **Statistical significance test**: Peak must be significant above local background
   ```
   significance = (peak_height - background) / sqrt(background)
   ```
   Require significance > 3σ (99.7% confidence)

**Implementation:**
```python
def detect_peaks_scipy(counts, min_prominence=50):
    peaks, properties = scipy.signal.find_peaks(
        counts,
        prominence=min_prominence,
        width=1
    )
    
    # Statistical filtering
    for peak in peaks:
        background = median(counts[peak-window:peak+window])
        noise = median_absolute_deviation(counts[peak-window:peak+window])
        if counts[peak] < background + 3*noise:
            remove peak
    
    return filtered_peaks
```

### 2. Derivative Method

Uses zero-crossings of the first derivative to identify peaks.

**Algorithm:**

1. **Calculate first derivative**:
   ```
   d[i] = (counts[i+1] - counts[i-1]) / 2
   ```

2. **Apply smoothing** to derivative to reduce noise

3. **Find zero crossings** where derivative changes from positive to negative:
   ```
   if d[i-1] > 0 and d[i+1] < 0:
       peak at i
   ```

4. **Apply second derivative test** to confirm maximum:
   ```
   d2[i] = counts[i+1] - 2*counts[i] + counts[i-1]
   ```
   Require `d2[i] < 0` for maximum

### 3. Template Matching

Correlates spectrum with expected peak shape.

**Algorithm:**

1. **Create template**: Gaussian with expected FWHM
   ```
   template[x] = exp(-0.5 * (x/σ)²)
   ```

2. **Cross-correlate** with spectrum:
   ```
   correlation[i] = Σ counts[i+j] * template[j]
   ```

3. **Find peaks** in correlation function

**Advantages:**
- Robust to noise
- Can detect weak peaks
- Works well for known peak shapes

## Peak Fitting Models

### 1. Gaussian Model

The primary model for gamma peaks, based on detector statistics.

**Mathematical Form:**
```
G(x) = A * exp(-0.5 * ((x - μ) / σ)²)
```

Where:
- A = amplitude (peak height)
- μ = centroid (peak position)
- σ = standard deviation (width parameter)
- FWHM = 2.355 * σ

**With Linear Background:**
```
f(x) = A * exp(-0.5 * ((x - μ) / σ)²) + mx + b
```

**Physical Basis:**
- Random detector processes follow Gaussian statistics
- Energy resolution dominated by statistical fluctuations
- Valid for well-resolved peaks

### 2. Gaussian with Low-Energy Tail

Accounts for incomplete charge collection and detector effects.

**Mathematical Form:**
```
f(x) = G(x) + T(x)

where T(x) = {
    B * exp((x - μ) / τ)  for x < μ
    0                      for x ≥ μ
}
```

Parameters:
- B = tail amplitude
- τ = tail decay constant

**Physical Basis:**
- Incomplete charge collection
- Compton scattering in detector
- Ballistic deficit in electronics

### 3. Voigt Profile

Convolution of Gaussian and Lorentzian, for high count rates.

**Mathematical Form:**
```
V(x; σ, γ) = ∫ G(x'; σ) * L(x - x'; γ) dx'
```

Where:
- G = Gaussian component (detector resolution)
- L = Lorentzian component (natural line width)
- σ = Gaussian width
- γ = Lorentzian width

**Applications:**
- High count rate spectroscopy
- Doppler broadening
- Natural line width significant

### 4. Double Gaussian (Multiplets)

For overlapping peaks that cannot be resolved.

**Mathematical Form:**
```
f(x) = A₁ * exp(-0.5 * ((x - μ₁) / σ₁)²) + 
       A₂ * exp(-0.5 * ((x - μ₂) / σ₂)²) + 
       mx + b
```

**Constraints:**
- |μ₁ - μ₂| < 3 * max(σ₁, σ₂) for overlap
- Shared background parameters

## Background Estimation

### 1. SNIP Algorithm

Statistics-sensitive Nonlinear Iterative Peak-clipping.

**Algorithm:**

1. **Transform to log space** to handle Poisson statistics:
   ```
   v[i] = log(counts[i])
   ```

2. **Iterative clipping** for p iterations:
   ```
   for p = window_max down to 1:
       for each channel i:
           v[i] = min(v[i], (v[i-p] + v[i+p])/2)
   ```

3. **Transform back**:
   ```
   background[i] = exp(v[i])
   ```

**Parameters:**
- iterations: typically 10-30
- window: peak width estimate

### 2. Rolling Ball Algorithm

Simulates rolling a ball under the spectrum.

**Algorithm:**

1. **For each point**, find minimum in radius r:
   ```
   background[i] = min(counts[i-r:i+r])
   ```

2. **Smooth result** to remove discontinuities

**Advantages:**
- Simple and intuitive
- No assumptions about peak shape
- Works for complex backgrounds

### 3. Percentile Filter

Uses local statistics for background estimation.

**Algorithm:**

1. **For each channel**, take percentile of surrounding window:
   ```
   background[i] = percentile(counts[i-w:i+w], p)
   ```

2. **Typical values**: p = 10-20%

**Applications:**
- Noisy spectra
- Variable background
- Quick estimation

## Energy Calibration

### 1. Linear Calibration

Most common for small energy ranges.

**Model:**
```
E = a * channel + b
```

**Fitting:** Least squares minimization
```
minimize Σ (E_known[i] - (a * ch[i] + b))²
```

**Uncertainty Propagation:**
```
σ_E² = σ_a² * ch² + σ_b² + 2 * ch * cov(a,b)
```

### 2. Polynomial Calibration

For wide energy ranges or non-linear detectors.

**Second Order:**
```
E = a₂ * ch² + a₁ * ch + a₀
```

**Higher Orders:**
```
E = Σ aᵢ * chⁱ  (i = 0 to n)
```

**Fitting:** Weighted least squares
```
minimize Σ w[i] * (E_known[i] - P(ch[i]))²
```

Where weights `w[i] = 1/σ_E[i]²`

### 3. Automatic Calibration

Pattern matching algorithm for known isotopes.

**Algorithm:**

1. **Detect all peaks** in spectrum

2. **Calculate ratios** between peak positions:
   ```
   ratio[i,j] = channel[i] / channel[j]
   ```

3. **Match to expected ratios** from known isotope:
   ```
   expected_ratio[i,j] = energy[i] / energy[j]
   ```

4. **Score matches**:
   ```
   score = Σ |ratio - expected_ratio| / expected_ratio
   ```

5. **Select best match** and create calibration

**Tolerance:** Typically 1-2% for ratio matching

## Uncertainty Propagation

### 1. Peak Area Uncertainty

**Poisson Statistics:**
```
σ_gross = sqrt(gross_area)
σ_background = sqrt(background_area)
σ_net = sqrt(σ_gross² + σ_background²)
```

**From Fit Parameters:**
```
Area = A * σ * sqrt(2π)
σ_Area = Area * sqrt((σ_A/A)² + (σ_σ/σ)²)
```

### 2. Centroid Uncertainty

From covariance matrix of fit:
```
σ_centroid = sqrt(Cov[μ,μ])
```

**Energy uncertainty:**
```
σ_E = sqrt((∂E/∂ch)² * σ_ch² + (∂E/∂a)² * σ_a² + ...)
```

### 3. Resolution Uncertainty

**FWHM uncertainty:**
```
σ_FWHM = 2.355 * σ_σ
```

**Resolution uncertainty:**
```
R = FWHM/E * 100%
σ_R = R * sqrt((σ_FWHM/FWHM)² + (σ_E/E)²)
```

## Quality Metrics

### 1. Chi-Square Test

Goodness of fit metric.

**Calculation:**
```
χ² = Σ ((observed - model) / σ)²
```

**Reduced chi-square:**
```
χ_reduced² = χ² / (n - p)
```

Where:
- n = number of data points
- p = number of parameters

**Interpretation:**
- χ² ≈ 1: Good fit
- χ² << 1: Overfit or overestimated errors
- χ² >> 1: Poor fit or underestimated errors

### 2. Signal-to-Noise Ratio

Peak quality metric.

**Definition:**
```
SNR = peak_height / noise_level
```

**Noise estimation:**
```
noise = std(counts - fitted_model)
```

Or from Poisson statistics:
```
noise = sqrt(background_level)
```

**Minimum detection limit:**
```
MDL = 3 * sqrt(2 * background)  (Currie limit)
```

### 3. Resolution

Detector performance metric.

**Energy resolution:**
```
R(%) = (FWHM_energy / peak_energy) * 100
```

**Typical values:**
- NaI(Tl): 6-8% at 662 keV
- HPGe: 0.2-0.3% at 1332 keV
- CZT: 1-2% at 662 keV

### 4. Peak Shape Parameters

**Tailing factor:**
```
T = (b + a) / (2 * a)
```
Where a and b are widths at 10% height

**Asymmetry:**
```
A = (right_width - left_width) / FWHM
```

## Optimization Techniques

### 1. Levenberg-Marquardt Algorithm

Used for non-linear least squares fitting.

**Update rule:**
```
θ[k+1] = θ[k] - (J^T J + λI)^(-1) J^T r
```

Where:
- θ = parameters
- J = Jacobian matrix
- r = residuals
- λ = damping parameter

### 2. Bounded Optimization

Ensures physical parameter constraints.

**Typical bounds:**
- Amplitude > 0
- 0 < σ < spectrum_width/4
- centroid within fit region
- background_intercept > 0

### 3. Initial Parameter Estimation

Critical for convergence.

**Amplitude:**
```
A_init = peak_height - background_estimate
```

**Centroid:**
```
μ_init = channel_of_maximum
```

**Width:**
```
σ_init = FWHM_estimate / 2.355
```

**Background:**
```
bg_init = mean(edge_channels)
```

## Performance Considerations

### Computational Complexity

| Algorithm | Complexity | Typical Time (4k channels) |
|-----------|------------|---------------------------|
| Peak Detection | O(n) | <10 ms |
| Single Peak Fit | O(m³) | ~50 ms |
| SNIP Background | O(n*p) | ~20 ms |
| Calibration | O(k²) | <1 ms |

Where:
- n = number of channels
- m = fit window size
- p = SNIP iterations
- k = calibration points

### Memory Requirements

```
Peak detection: 3n floats (counts, smoothed, background)
Peak fitting: m² floats per peak (covariance matrix)
Visualization: 4n floats (multiple arrays)
```

## References

1. **Peak Detection**
   - Morháč, M. et al. "Background elimination methods for multidimensional coincidence γ-ray spectra." NIM A 401 (1997): 113-132.

2. **Peak Fitting**
   - Bevington, P.R. & Robinson, D.K. "Data Reduction and Error Analysis for the Physical Sciences" (2003)

3. **SNIP Algorithm**
   - Ryan, C.G. et al. "SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications." NIM B 34 (1988): 396-402.

4. **Energy Calibration**
   - Gilmore, G. "Practical Gamma-ray Spectrometry" 2nd Ed. (2008)

5. **Uncertainty Analysis**
   - JCGM 100:2008 "Evaluation of measurement data — Guide to the expression of uncertainty in measurement"

6. **Detector Resolution**
   - Knoll, G.F. "Radiation Detection and Measurement" 4th Ed. (2010)

---

*For implementation details, see the [API Reference](api_reference.md)*
