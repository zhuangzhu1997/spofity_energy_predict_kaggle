---
layout: default
title: Predicting Spotify Track Energy Across 114 Genres
---

## Introduction

Can we predict Spotify's Energy score from a track's audio features
and catalog metadata? I study 89,740 tracks across 114 genres.

Spotify's Energy score ranges from 0 to 1 and measures the perceived intensity and activity of a track. I first explore how Energy and other audio features vary by genre, then test how well the available features can reconstruct Energy for held-out tracks.

The original track table, `music_tracks.csv`, contains 114,000 rows with 1,000 records for each genre. A track can appear under multiple genre labels. The accompanying `artists.csv` contains artist names, follower counts, popularity scores, and genre tags. The analysis uses the DSC 80 Spotify dataset, which is adapted from an upstream Kaggle dataset. [Course dataset description](https://dsc80.com/proj04/project4-music/) · [Upstream dataset](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks/data)

The key track columns are `track_id`, Spotify's identifier for each track, and `track_genre`, the label attached to each record. `energy` is the target. `loudness` measures overall level in decibels, `danceability` describes suitability for dancing, `acousticness` measures how likely a track is to be acoustic, `tempo` is estimated beats per minute, and `time_signature` is the estimated number of beats per measure. The `artists` column records artist credits, and `popularity` is a Spotify score from 0 to 100.

## Data Cleaning and Exploratory Data Analysis

### Data cleaning

- I removed the CSV index column `Unnamed: 0` and dropped one record with no track, album, or artist name. That record also had zero duration. I trimmed surrounding whitespace from the name columns.
- I treated zeros in `tempo` and values of 0 or 1 in `time_signature` as missing. When a track appeared under multiple labels and a copy had a valid tempo, I used that value. All valid tempo values agreed within each track ID.
- I dropped `release_date` because some records with the same track ID had conflicting dates. For example, the dataset spreads tracks from *The Complete Hank Williams* across 1956–2023, while Spotify gives the album a single release date in 1998.
- I kept one row per `track_id` in `tracks_cleaned` so the same track would appear only once in the analyses and model splits. I used the median of the distinct popularity scores as the track's reference popularity. A separate `genre_track_cleaned` table preserves all 113,549 unique track–genre pairs, including tracks with multiple labels.
- In the artist table, I kept `id` as the key because different artists can share a name. I parsed genre tags into lists and retained the 11 artists whose follower counts were missing. The string `"None"` was retained where it was an actual artist name.

The first five rows of `tracks_cleaned`, shown with selected columns, are below.

| Track | Artist credits | Energy | Loudness (dB) | Danceability | Tempo (BPM) |
| --- | --- | ---: | ---: | ---: | ---: |
| Comedy | Gen Hoshino | 0.461 | −6.746 | 0.676 | 87.917 |
| Ghost - Acoustic | Ben Woodward | 0.166 | −17.235 | 0.420 | 77.489 |
| To Begin Again | Ingrid Michaelson and ZAYN | 0.359 | −9.734 | 0.438 | 76.332 |
| Can't Help Falling In Love | Kina Grannis | 0.0596 | −18.515 | 0.266 | 181.740 |
| Hold On | Chord Overstreet | 0.443 | −9.681 | 0.618 | Missing |

### Univariate analysis

Energy is left-skewed, with more tracks concentrated toward the high end of the scale. The median Energy score among the cleaned tracks is 0.676.

<iframe src="assets/energy-distribution.html" title="Distribution of Spotify Energy scores" style="display: block; width: 100%; border: 0;" height="410" loading="lazy"></iframe>

### Bivariate analysis

#### Energy and loudness

Energy and loudness are strongly positively correlated across the cleaned tracks (Pearson *r* = 0.759). Louder tracks generally have higher Energy scores, consistent with both measures being derived from the audio.

<iframe src="assets/energy-vs-loudness.html" title="Spotify Energy and loudness scatterplot" style="display: block; width: 100%; border: 0;" height="440" loading="lazy"></iframe>

#### Energy by genre

I chose classical, sleep, jazz, EDM, and heavy metal to compare quiet listening contexts, an intermediate group, and styles often associated with louder or more active recordings. Classical tracks generally have lower Energy scores, while EDM and heavy metal tracks have higher scores. Sleep tracks have a low median and show two concentrations at different Energy levels.

<iframe src="assets/energy-by-genre.html" title="Spotify Energy distributions across five genres" style="display: block; width: 100%; border: 0;" height="430" loading="lazy"></iframe>

The relationship between genre and audio features also varies across the full set of labels. Among the seven audio features examined, speechiness has the strongest absolute association with the comedy label (*r* = 0.625), consistent with substantial spoken content. Opera tracks have lower average danceability than the remaining tracks (0.313 versus 0.565).

### Interesting aggregates

I grouped tracks by their reference popularity score and computed the mean of five audio features in each band.

| Popularity band | Tracks | Danceability | Energy | Valence | Acousticness | Liveness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9,347 | 0.556 | 0.578 | 0.483 | 0.390 | 0.190 |
| (0, 20] | 15,738 | 0.562 | 0.652 | 0.449 | 0.287 | 0.203 |
| (20, 40] | 30,033 | 0.545 | 0.654 | 0.492 | 0.336 | 0.244 |
| (40, 60] | 25,786 | 0.572 | 0.618 | 0.444 | 0.341 | 0.216 |
| (60, 80] | 8,335 | 0.595 | 0.640 | 0.487 | 0.276 | 0.181 |
| (80, 100] | 501 | 0.659 | 0.668 | 0.513 | 0.209 | 0.175 |

Tracks in the highest popularity band have higher average danceability and lower average acousticness and liveness than tracks in the other bands. Energy and valence show no clear monotonic relationship with popularity.

## Assessment of Missingness

### NMAR analysis

Tempo may be harder to estimate for tracks with weak rhythmic structure. If the missingness still depended on the true tempo after accounting for observed features, it would be missing not at random (NMAR). The available data do not establish that mechanism. Rhythm-detection confidence or feature-extraction failure codes could help explain which tracks are missing tempo or time signature values.

### Missingness dependency

I tested whether tempo missingness was associated with Energy and whether time signature missingness was associated with danceability. In each test, the null hypothesis was that missingness and the observed feature were independent. The alternative was that there was an association. I permuted the missingness labels 5,000 times and compared the absolute observed difference in group means with the permutation distribution.

| Missing field | Missing tracks | Compared feature | Mean if missing | Mean if observed | Difference | *p*-value |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `tempo` | 15,212 (16.95%) | Energy | 0.490 | 0.664 | −0.1735 | 0.0002 |
| `time_signature` | 1,008 (1.12%) | Danceability | 0.351 | 0.565 | −0.2139 | 0.0002 |

Neither test produced a permuted difference as extreme as its observed difference. The results show that the missingness of these fields is associated with observed audio features, which is consistent with a missing-at-random (MAR) explanation.

<iframe src="assets/audio-feature-missingness.html" title="Energy and danceability distributions by missingness status" style="display: block; width: 100%; border: 0;" height="600" loading="lazy"></iframe>

## Hypothesis Testing

The [genre comparison](#energy-by-genre) shows that sleep and classical tracks have similarly low median Energy scores but visibly different distribution shapes. I used a two-sample Kolmogorov–Smirnov (KS) test to compare the full distributions.

- **Null hypothesis.** The Energy distributions of sleep and classical tracks are the same.
- **Alternative hypothesis.** Their Energy distributions differ.
- **Test statistic.** The maximum absolute difference between the two empirical cumulative distribution functions. This captures differences in shape as well as differences in location.
- **Significance level.** α = 0.05.

The observed KS statistic was **0.2331** and the two-sided *p*-value was approximately **1.67 × 10⁻²³**. At the 0.05 significance level, I reject the null hypothesis. The sample provides evidence that the two genres have different Energy distributions.

## Framing a Prediction Problem

I frame the prediction problem as **regression**. The target is `energy`, a continuous value between 0 and 1 that is present for every track in the cleaned table. The model covers the full catalog in the dataset, spanning 114 genre labels.

Spotify computes Energy from the audio. This task reconstructs that score from related audio summaries, including loudness. In the catalog completion setting, other audio summaries, genre tags, and artist credits can be available when Energy needs to be estimated. The artist encoding uses Energy observations from the data used to fit the model.

I use **mean absolute error (MAE)** as the primary metric because it expresses the average prediction error directly in Energy units. I also report R² to show how much variation in Energy the models explain.

## Baseline Model

After cleaning, each row represents one unique Spotify track ID. I split the catalog tracks into training (60%), validation (20%), and test (20%) sets. The same artist could appear in multiple splits. The holdout measures performance on tracks the models did not use for fitting.

- **Training median.** A constant predictor that minimizes absolute error on the training targets.
- **Ridge regression.** A linear model with a regularization strength of α = 10. The pipeline constructs features, imputes missing values with medians and adds missingness indicators, standardizes the features, and fits Ridge.

The Ridge pipeline uses eight continuous audio measurements, log duration, time signature as a discrete count, mode as a binary major/minor indicator, and a sine/cosine encoding of musical key. That gives 13 feature columns before the imputer adds its indicators. Regression predictions are clipped to the interval from 0 to 1 for evaluation.

The EDA showed a strong relationship between Energy and loudness, giving a linear model a useful starting point. The initial Ridge fit achieved a test MAE of approximately 0.093, compared with 0.212 for the training median. About 62% of its predictions were within 0.10 of the true Energy score.

## Final Model

The final model is a ReLU multilayer perceptron (MLP) regressor. It extends the Ridge inputs with two engineered audio features and two sources of catalog metadata.

- **Relative amplitude.** `10^(loudness / 20)` maps loudness from decibels to a linear amplitude scale.
- **Loudness × acousticness.** `(loudness + 30) × acousticness` lets the model learn how the relationship between loudness and Energy varies with acousticness.
- **Genre tags.** Each genre gets a binary feature. A track with multiple tags can activate multiple features.
- **First credited artist.** The model uses a smoothed historical Energy mean estimated from the fitting data. I use five-fold cross-fitting for rows used to fit the pipeline. At prediction time, unseen artists receive the mean Energy of the fitting data.

### Model selection and refit

I first compared four feature sets at the same MLP settings. Audio features with both genre and artist metadata achieved the lowest validation MAE. I then compared one hidden layer of 64 units with two layers of 128 and 64 units, using α values of 0.0001 and 0.01 to test different levels of L2 regularization. The selected MLP used one 64-unit hidden layer and α = 0.01, with a validation MAE of 0.07003.

After locking the feature set and hyperparameters, I refit the MLP and both baselines on the same 71,792 training and validation tracks. I evaluated all three models on the original 17,948 held-out tracks.

| Model | Test MAE | Test R² | Predictions within 0.10 |
| --- | ---: | ---: | ---: |
| Training median | 0.2115 | −0.0242 | 26.14% |
| Ridge regression | 0.0933 | 0.7743 | 62.01% |
| **MLP regression** | **0.0690** | **0.8679** | **76.42%** |

The final MLP lowered MAE by about 26% compared with Ridge. As a supplementary check, I used the continuous Energy predictions to rank tracks above and below 0.5. The MLP achieved a ROC AUC of 0.9707 on that comparison.

## Fairness Analysis

### Does the model perform worse on tracks by unseen artists?

I grouped test tracks by whether their first credited artist appeared as the first credited artist on any track in the combined training and validation splits. The refitted model uses an artist-specific historical Energy estimate for seen artists and the mean Energy of its fitting data for unseen artists.

- **Null hypothesis.** Artist-level mean squared errors are exchangeable between the seen and unseen groups.
- **Alternative hypothesis.** Artist-weighted RMSE is higher for unseen artists.
- **Test statistic.** Artist-weighted RMSE for unseen artists minus that for seen artists.
- **Significance level.** α = 0.05.

I first averaged the squared errors within each artist, then averaged across artists in each group and took the square root. This gives each artist equal weight. I ran 10,000 permutations of group membership at the artist level, keeping each artist's tracks together.

| First credited artist | Artists | Test tracks | Artist-weighted RMSE |
| --- | ---: | ---: | ---: |
| Seen during fitting | 6,353 | 16,284 | 0.0919 |
| Unseen during fitting | 1,533 | 1,664 | 0.0991 |

The estimated RMSE difference was **0.0072** Energy units, with a one-sided permutation *p*-value of **0.0014**. At the 0.05 significance level, I reject the null hypothesis. The refitted model had higher prediction error for unseen artists on this test set. Differences in genre mix or other track characteristics may also contribute to the observed gap.
