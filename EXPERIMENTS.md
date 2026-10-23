# Experiment Log

Keep one section per meeting. Add the tables and plots you make from
`runs/*/results.json` and `test-results.json` (save the plots in `evidence/`),
link the pools' contact sheets, and explain what you see. A result that shows no
effect is still a result: say so and say why.

## Meeting 1 — Real-data baselines

- **Question:** How much do oversampling and ordinary augmentation help Egyptian
  Mau recall compared with the imbalanced data?
  - **Oversampling (`m1-oversample`):**
    - **Validation:** Boosts Egyptian Mau recall from **49.0%** to **65.0%** (an absolute gain of **+16.0 percentage points**, or a **32.7% relative improvement**), with Macro F1 increasing from 0.863 to 0.884.
    - **Test:** Boosts recall from **52.0%** to **68.0%** (an absolute gain of **+16.0 percentage points**, or a **30.8% relative improvement**), with Macro F1 increasing from 0.858 to 0.873.
  - **Ordinary Augmentation (`m1-augment`):**
    - **Validation:** Further elevates recall to **68.0%** (an absolute gain of **+19.0 percentage points** over imbalanced data, and **+3.0 percentage points** over oversampling), with Macro F1 reaching 0.890 and lower seed variance (std drops from 0.058 to 0.040).
    - **Test:** Delivers the strongest result with **73.3%** recall (an absolute gain of **+21.3 percentage points** over imbalanced data, and **+5.3 percentage points** over oversampling), alongside the highest Macro F1 of **0.887** and very high consistency across seeds (std = 0.021).
  - **Summary:** Both techniques provide substantial improvements without harming recognition of majority breeds. While oversampling fixes class-frequency bias (+16.0% recall), augmentation provides an essential extra boost (+21.3% total recall) by preventing the model from overfitting to static background features.

- **How `oversample()` works:**
  `oversample(minority_photos, budget, rng)` takes the 15 Egyptian Mau training photos and repeatedly appends random permutations (`rng.shuffle(order)`) until reaching `budget` (55), taking `chosen[:budget]`. Because 55 = 15 × 3 + 10, exactly 10 photos appear 4 times and 5 photos appear 3 times. This guarantees the minority training set is distributed as evenly as possible ($\max - \min \le 1$) while remaining strictly reproducible from the seed passed to `rng`.

- **How `augment()` works, and why these transforms:**
  `augment(image)` generates a new random variant of an Egyptian Mau photo each time an added sample is drawn in an epoch. It uses:
  - `RandomHorizontalFlip(p=0.5)`: Reflects the cat along the vertical axis (cats exhibit bilateral symmetry).
  - `RandomRotation(degrees=15)`: Gentle tilting ($\pm 15^\circ$) to simulate slight head and body poses without unrealistic upside-down angles.
  - `RandomAffine(translate=(0.04, 0.04), scale=(0.95, 1.05))`: Subtle translation and zoom to simulate variations in camera distance and framing.
  - `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)`: Simulates varied lighting conditions across indoor and outdoor shots.
  
  *Why these transforms:* Egyptian Maus are distinguished by their dark gooseberry-spotted coat pattern, green/amber eye coloration, and facial mascara lines. These moderate transformations inject realistic photographic diversity without distorting anatomical proportion or obscuring the characteristic spot patterns.

- **Commands:**
  ```bash
  # Training baselines
  uv run synth-train --name m1-real --method real
  uv run synth-train --name m1-oversample --method oversample
  uv run synth-train --name m1-augment --method augment

  # Training augmentation variation (stronger transforms)
  AUGMENT_LEVEL=strong uv run synth-train --name m1-augment-strong --method augment

  # Evaluating on test set
  uv run synth-test --run m1-real
  uv run synth-test --run m1-oversample
  uv run synth-test --run m1-augment
  ```

- **Results on validation:**
  Plot saved in [`evidence/m1_validation_comparison.png`](evidence/m1_validation_comparison.png).

  | Run | Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± Std |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `m1-real` | Mau Recall | 0.450 | 0.500 | 0.450 | 0.450 | 0.600 | 0.490 ± 0.058 |
  | `m1-real` | Macro F1 | 0.861 | 0.856 | 0.861 | 0.855 | 0.884 | 0.863 ± 0.011 |
  | `m1-oversample` | Mau Recall | 0.600 | 0.600 | 0.650 | 0.650 | 0.750 | 0.650 ± 0.055 |
  | `m1-oversample` | Macro F1 | 0.890 | 0.872 | 0.879 | 0.891 | 0.886 | 0.884 ± 0.007 |
  | `m1-augment` | Mau Recall | 0.700 | 0.650 | 0.650 | 0.650 | 0.750 | 0.680 ± 0.040 |
  | `m1-augment` | Macro F1 | 0.905 | 0.867 | 0.898 | 0.892 | 0.886 | 0.890 ± 0.013 |
  | `m1-augment-strong` | Mau Recall | 0.700 | 0.700 | 0.650 | 0.600 | 0.800 | 0.690 ± 0.066 |
  | `m1-augment-strong` | Macro F1 | 0.904 | 0.880 | 0.879 | 0.878 | 0.912 | 0.891 ± 0.014 |

- **What I see:**
  1. **Impact of Class Balancing:** The imbalanced baseline (`m1-real`) struggles with minority recall (0.490 on validation). Simply oversampling minority photos (`m1-oversample`) boosts Egyptian Mau recall by +16.0 percentage points (to 0.650) while also raising overall Macro F1 from 0.863 to 0.884. This proves that equalizing class exposure allows the classifier to build more representative decision boundaries without harming the other 7 majority breeds.
  2. **Impact of Augmentation:** `m1-augment` provides an additional gain, pushing recall to 0.680 and Macro F1 to 0.890, while reducing variance across seeds (standard deviation dropped from 0.055 to 0.040). Presenting varied geometric and photometric versions across epochs prevents the CNN from memorizing static background artifacts of the 15 photos.
  3. **Moderate vs. Strong Augmentation:** Testing `m1-augment-strong` (degrees=30, perspective=0.2, color jitter=0.25) achieved a comparable mean recall (0.690 vs 0.680), but resulted in notably higher variance ($\text{std} = 0.066$ vs $0.040$), dipping as low as 0.600 on Seed 3. Excessive warping and rotation occasionally degrades distinctive facial and spotting features.
  4. **Choice:** I keep moderate `m1-augment` because it delivers consistent, high recall and low cross-seed variance without risking feature distortion.

- **Test results of the three baselines:**
  Plot saved in [`evidence/m1_test_comparison.png`](evidence/m1_test_comparison.png).

  | Run | Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± Std |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `m1-real` | Mau Recall | 0.433 | 0.533 | 0.533 | 0.467 | 0.633 | 0.520 ± 0.069 |
  | `m1-real` | Macro F1 | 0.838 | 0.862 | 0.871 | 0.848 | 0.872 | 0.858 ± 0.013 |
  | `m1-oversample` | Mau Recall | 0.700 | 0.633 | 0.667 | 0.700 | 0.700 | 0.680 ± 0.027 |
  | `m1-oversample` | Macro F1 | 0.878 | 0.869 | 0.882 | 0.883 | 0.853 | 0.873 ± 0.011 |
  | `m1-augment` | Mau Recall | 0.733 | 0.733 | 0.733 | 0.700 | 0.767 | 0.733 ± 0.021 |
  | `m1-augment` | Macro F1 | 0.879 | 0.882 | 0.900 | 0.891 | 0.883 | 0.887 ± 0.007 |

  On the unseen 240 test photos, `m1-augment` establishes the strongest baseline: **0.733 ± 0.021 recall** (+21.3 percentage points over real, +5.3 over oversample) and **0.887 ± 0.007 Macro F1**, demonstrating strong generalization with remarkably stable performance across all 5 seeds. Every later meeting will compare against these reference test metrics.

## Meeting 2 — Text-to-image generation

- **Question:**
- **Setting changed in each pool (one at a time):**
- **Commands:**
- **Contact sheets:** (good and bad images, with the failure types you name)
- **Results on validation:** (table and plot)
- **What I see:**
- **Pool I take forward, and why:**
- **Its test result, compared with the baselines:**

## Meeting 3 — Generation from training photos

- **Question:** How close to the real photos should the synthetic images be?
- **img2img strength values and IP-Adapter scale values tried:**
- **Commands:**
- **Contact sheets:**
- **Results on validation:**
- **What I see:**
- **Best run of each mode, and its test result:**

## Meeting 4 — Editing training photos

- **Question:** Does the classifier need new cats or new backgrounds?
- **Background prompts and outpainting scales tried:**
- **Commands:**
- **Contact sheets:**
- **Results on validation:**
- **What I see:**
- **Best run of each mode, and its test result:**

## Final comparison

- **Runs chosen in Meetings 1–4:**
- **Results:** (your table and plot of their `test-results.json`)
- **Did the validation ranking hold on the test photos?**
- **Recommendation:** use, revise, or reject synthetic data for this task.
