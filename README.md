# pitchladder
model for predicting the next pitch
Goal: A lightweight, explainable model that estimates the probability of the next pitch type using only pre‑pitch information. Built to be fast, transparent, and good enough for a one‑day PoC—with a clear path to MVP and state‑of‑the‑art quality.

## 1) Problem statement
Goal: Estimate the most likely next pitch using only pre‑pitch information. 

A working model will be able to anticipate likely 'pitch families' (ie top‑2 most likely pitches). It will also be able to quantify tendencies by situation and sequence.

Future work: 
- Try other models: Random Forest, SVM, Nueral Networks, etc. 
- Predict velocity & location
- More robust "per‑pitcher" models
- Evaluate model against many scenarios (runners on, pitch count, etc.)
- front end UI so others can easily engage with the model

## 2) Approach (PoC)
- Custom Bayesian Model: the "Pitch Ladder"
- Predicting the next pitch based on available information made me think of weather prediction, which made me think of Bayesian models
- Based on the literature: combine some concepts into a hierarchical empirical‑Bayes frequency model with a small per‑pitcher sequence prior
-- Hierarchical frequency + shrinkage is good for handling sparse data (Efron, B. and Morris, C. (1975). Data analysis using Stein’s estimator and its generalizations. J. Amer. Stat. Assoc. 70 311–319. MR0391403 - they estimated season batting averages off small sample sizes)
-- Pitch choice shows sequence dependence (Gray R. "Markov at the bat": a model of cognitive processing in baseball batters. Psychol Sci. 2002 Nov;13(6):542-7. doi: 10.1111/1467-9280.00495. PMID: 12430839.)
-- "ladder"/"backoff" mechanism: the idea is to combine specific counts with broader contexts to get calibrated probabilities, like used in the domain of language‑model smoothing (Katz, S. M. (1987). Estimation of probabilities from sparse data for the language model component of a speech recognizer. IEEE Transactions on Acoustics, Speech, and Signal Processing, 35(3), 400–401.)
- I'll build a model based on these papers.
- The model will cliimb a "ladder" of questions, like:
-- League: “In the whole league, how often is each pitch thrown?”
-- Pitcher: “What does this pitcher like to throw?”
-- Situation: “Given the count and handedness, what does he usually throw?”
-- Recent sequence: “After the last pitch type he just threw, what does he usually throw next?”
-- At each rung, count how often each pitch shows up. If a rung doesn’t have many examples (it’s “sparse”), lean back toward the broader rungs below it (like trusting the league average more when the situation is rare).
- The model will return top-n most likely pitches; next model should return top pitches broken out by liklihood.
- I'll call the model the *Pitch Ladder* because of how it processes and reports pitch likelihoods
- Data split: 70% / 30% 
- Metrics: log loss (primary), top‑1, top‑2.
- Why: fast, low-memory, simple code, suits the data.

## 3) Literature snapshot
- The most consistently useful pre‑pitch signals seem to include: count, handedness matchups, previous pitch type/result, and on‑base state.
- Random Forests (per pitcher, 100 trees) were the strongest baseline (~66.6% average accuracy, 287 MLB pitchers) in a widely‑cited study. 
- Markov/sequence structure has been explored for pitch choice; layering a situational model on top of simple transitions is a practical improvement for live prediction. 

## 4) model card:
- Intended use: Real‑time pre‑pitch guidance for hitters and coaches; scouting/advance reports.
- Inputs (pre‑pitch): pitcher_id, stand (batter side), p_throws (pitcher hand), balls, strikes, prev_pitch_type, on_1b/on_2b/on_3b, optional tto_bucket.
- Output: 7‑way probability distribution over {FF, CT, SI, SL, CU, CH, KN}, plus top‑1/top‑2.
- Training data: 2011 MLB‑style pitch records; canonicalized to 7 pitch classes.
- Data split: 70/30
- Evaluation: top‑1 = 0.505, top‑2 = 0.767, logloss = 1.166.
- Random performance = 14.3% top‑1; uniform logloss = ln(7) ≈ 1.946 → this model is substantially better and reasonably calibrated.
- Uses short sequence memory (prev pitch only) and simple context; not batter‑specific; no recency weighting (yet).
- Performance varies by pitcher/data volume and by count segment.

## 5) Action plan
PoC: 
- simple model, ie "Pitch Ladder"
- minimal features, ie pitcher, count, handedness, prev pitch, base state.
- Evaluate with top‑1/top‑2/logloss.

Next Steps: 
- Add per‑pitcher Random Forest baseline using exactly the same pre‑pitch features; confirm we can match/approach the ~66% in the literature on comparable cohorts. 
- hyperparameter tuning (ie grid search)
- better sequence modeling (Markov, RNN, etc)
- batter‑specific adjustments
- knowledge representation: ie modeling 'game flow' differently
- calibration check to make sure the results are realistic and reliable
- stronger naive baselines (aka fairer comparisons)
- add more context, ie by count bucket (0‑0, 1‑2, 3‑1, etc.), by handedness matchup (RHP vs LHB / LHP vs RHB), and by times‑through‑order (TTO).

MVP:
- Represent the results in a way that batters, coaches, scouts, etc. can easily use
- hosted env. 
- user authentication
- database
- etc. 

## 6) Results 
### Pitch Ladder Model
#### Metrics: 
- Top‑1 = 0.505
-- this means the model correctly guesses the next pitch type about half the time. 
-- Assuming 7 possible pitches of equal likelihood, random guessing would be 1/7 ≈ 14.3%. So this model is +36 points over pure chance. 
- Top‑2 = 0.767
-- When given two guesses, the model correctly predicts the next pitch ~77% of the time. In other words, if the batter “hunts” one of the top‑two predicted pitches, he’ll be right ~77% of the time. 
-- Note 1: As promising as these Top-1 and Top-2 results seem, in practice baseball is usually more fastball-heavy, so a next step must include normalizing for this. Put another way- random guessing should not be 14.3%.... If you guess "knuckleball" for RA Dickey you'll be right around 80% of the time, and the  model should reflect this. 
-- Note 2: a quick look at the data shows fastballs thrown ~35% of the time. Fastball and Sliders are thrown ~58% of the time. The "Pitch Ladder" model correctly guesses the next pitch ~50% of the time, and ~77% of the time when given two chances. This means the "Pitch Ladder" model is still *~46% better* than blindly guessing "Fastball" and *~33% better* than guessing "Fastball or Slider", which are meaningful lifts. The literature suggests Random Forest models can achieve 66% accuracy, so there is likely still lots of room to grow! 
-- Note 3: These are only league‑level priors. They ignore pitcher, count, previous pitch, handedness, etc. These are simple yardsticks that indicate the model can be improved by accounting for more context. 
- Logloss = 1.166 (lower is better)
-- A uniform 7‑way guess has logloss ln(7) ≈ 1.946. We have 1.166, i.e., ~40% reduction in negative log‑likelihood vs uniform. 
-- This means probabilities are reasonably calibrated and informative.
