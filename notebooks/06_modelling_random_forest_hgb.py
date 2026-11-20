#!/usr/bin/env python
# coding: utf-8

# # 06. Random Forest and HistGradientBoosting 
# 
# In this part of the project, I am doing the two tree based models: Random Forest and HistGradientBoosting.
# 
# What the models predict: State of Health, or SOH. This is how much life a battery still has, written as a percent. 100 percent means the battery is like new.
# 
# The most important rule: I test my models on batteries they have never seen before. That is the real goal of the project. I want to know if the model works on a brand new battery, not only on the batteries we already measured. So the main test (called LOCO) hides one whole battery, trains on the rest, and then checks how well it guesses the hidden one.
# 
# Why I keep the models small: I only have 306 measurements from about 15 batteries. That is a small amount of data. So I keep the models simple (shallow trees, and at least 5 data points in each leaf) so they do not just memorize single batteries.
# 
# All errors are in SOH percent points. Lower is better. An error of 4 means the guess is about 4 percent off.

# In[1]:


# I set up the notebook and import the tools I need.
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# I hide harmless warnings (like "unknown category") so the notebook stays clean and small.
warnings.filterwarnings('ignore')

# I make sure Python can find the project code whether I run this from the
# notebooks folder or from the main project folder.
ROOT = Path.cwd()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

# I import the two models I am building, plus the team's shared scoring code.
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from src import evaluation as ev

# I create the folder where I will save my results.
OUT = ROOT / 'results' / 'modelling_random_forest_hgb'
(OUT / 'figures').mkdir(parents=True, exist_ok=True)
print('project folder:', ROOT)
print('test methods  :', list(ev.SCHEMES))
print('random seed   :', ev.SEED)


# ## 1. I get the data ready
# 
# I load the 306 rows I can label. Each row is one health check up of one battery at 25 degrees. I also load the list of inputs the model is allowed to use. I leave out any column that secretly contains the answer, so the model has to really learn instead of cheating.

# In[2]:


# I load the data and the list of allowed inputs, straight from the repo files.
df = ev.load_labelled()
numeric, categorical = ev.main_features()
print(f'rows: {len(df)}   batteries: {df.cell_id.nunique()}   conditions: {df.profile.nunique()}')
print(f'number of inputs: {len(numeric)}   plus category input: {categorical}')
print()
print('how many batteries fall in each health group:')
print(df.health_class.value_counts())
df[['cell_id', 'profile', 'idtest', 'rtp_index', 'SOH', 'health_class']].head()


# ## 2. I set up my two models
# 
# For each model I write a short list of settings to try (called a grid). I write this list before I see any score, and I never change it after. If I changed it after seeing the results, I would just be fixing the numbers to look good, and that is not fair.
# 
# My two models:
# 1. Random Forest. It builds many small decision trees and averages their guesses.
# 2. HistGradientBoosting. It builds trees one after another, where each new tree fixes the mistakes of the one before.
# 
# I keep both models simple on purpose: shallow trees, and every leaf must hold at least 5 data points.

# In[3]:


# I pick the settings to try for each model. I wrote these before seeing any result.
SEED = ev.SEED  # 42, so my results are the same every time I run it.

# I build my Random Forest.
rf_spec = ev.ModelSpec(
    name='Random forest',
    estimator=RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=1),
    param_grid={
        'max_depth': [4, 8],            # I keep the trees shallow.
        'min_samples_leaf': [5, 10],    # I keep at least 5 points per leaf.
        'max_features': [0.5, 1.0],
    },
    family='non-linear',
)

# I build my HistGradientBoosting.
hgb_spec = ev.ModelSpec(
    name='HistGradientBoosting',
    estimator=HistGradientBoostingRegressor(max_iter=300, random_state=SEED, early_stopping=False),
    param_grid={
        'learning_rate': [0.05, 0.1],
        'max_leaf_nodes': [7, 15],      # I keep the trees shallow.
        'min_samples_leaf': [10, 20],   # I keep at least 5 points per leaf.
    },
    family='non-linear',
)

# I print how many settings each model will try.
for s in (rf_spec, hgb_spec):
    n = int(np.prod([len(v) for v in s.param_grid.values()]))
    print(f'{s.name}: {n} settings to try -> {s.param_grid}')


# ## 3. I train and test the models
# 
# This cell does the real work. For every model and every test method, it:
# 1. Hides some batteries.
# 2. Picks the best settings using only the batteries it is allowed to see.
# 3. Trains, and then guesses the hidden batteries.
# 
# Picking the settings without looking at the hidden batteries is what keeps my score honest. This can take a minute or two to finish.

# In[4]:


# I run both models through all three test methods and collect the results.
results = {}        # I store each result here, by (model, test method).
all_results = []    # I also keep a flat list for the summary tables.

for spec in (rf_spec, hgb_spec):
    for scheme in ev.SCHEMES:
        res = ev.run_cv(df, spec, scheme, verbose=True)
        results[(spec.name, scheme)] = res
        all_results.append(res)

print('\ndone. I ran', len(all_results), 'model and test-method combinations.')


# ## 4. My main results
# 
# This is the main table. There are three ways of testing, from hardest to easiest:
# 1. LOCO. I hide one whole battery. This is the number that matters most.
# 2. LOPO. I hide one whole operating condition.
# 3. Random 5-fold. This is the easy, too good test, because it lets the model peek at other check ups of the same battery. I show it only to prove how misleading that kind of test looks.
# 
# The generalization gap is just how much more optimistic that easy test is compared to the honest one.

# In[5]:


# I build my main results table and sort it so the best score is on top for each test.
summary = ev.summary_table(all_results).sort_values(['scheme', 'MAE']).reset_index(drop=True)
summary_round = summary.copy()
for col in ['MAE', 'RMSE', 'R2', 'MaxAE']:
    summary_round[col] = summary_round[col].round(3)
print('My results (MAE and RMSE are in SOH percent points, lower is better):')
display(summary_round)

print()
print('Generalization gap (honest-test error minus easy-test error):')
display(ev.generalization_gap(all_results).round(3))


# ## 5. Where does my model make mistakes?
# 
# A single average error hides a lot of the story. So I split the error up:
# 1. By health group: healthy, getting worse, or near the end of life.
# 2. By condition: batteries run in conditions I have seen before, versus brand new conditions.
# 3. By each battery.
# 
# This shows honestly when my model can be trusted and when it cannot.

# In[6]:


# I pick my better model on the main test (LOCO), then look at where it makes mistakes.
rf_loco_mae = results[('Random forest', 'LOCO')].metrics['MAE']
hgb_loco_mae = results[('HistGradientBoosting', 'LOCO')].metrics['MAE']
best_name = 'Random forest' if rf_loco_mae <= hgb_loco_mae else 'HistGradientBoosting'
best_spec = rf_spec if best_name == 'Random forest' else hgb_spec
print(f'My better model on the main test: {best_name} '
      f'(Random forest={rf_loco_mae:.3f}, HistGradientBoosting={hgb_loco_mae:.3f})')

oof_best = results[(best_name, 'LOCO')].oof
print()
print('Error by health group:')
display(ev.breakdown(oof_best, 'health_class').round(3))
print('Error by condition (seen before vs brand new):')
display(ev.breakdown(oof_best, 'regime').round(3))
print('Error for each battery (worst 8):')
display(ev.breakdown(oof_best, 'cell_id').sort_values('MAE', ascending=False).head(8).round(3))


# ## 6. Is my better model really better?
# 
# A slightly lower average is not enough, because it could just be luck. So I use a proper test (called Wilcoxon) to compare my two models head to head. It asks: does one model really win battery after battery, not just on average? If mean_diff is below zero, Random Forest is the better one. A small p value (under 0.05) means the difference is real. A large p value means the two models are basically tied.

# In[7]:


# I compare my two models head to head on the same batteries.
rf = results[('Random forest', 'LOCO')]
hgb = results[('HistGradientBoosting', 'LOCO')]
cmp = pd.DataFrame([ev.paired_comparison(rf, hgb)])
print('Random forest vs HistGradientBoosting on the same batteries:')
print('(mean_diff below 0 means Random forest wins. A small wilcoxon_p means the gap is real, not luck.)')
display(cmp.round(4))


# ## 7. Which settings did my model pick?
# 
# For each hidden battery, my model chose its own best settings. If it keeps picking the same ones, my model is stable and I can trust it. If the choices jump around a lot, that is a sign my dataset is just small, which is something I say honestly in the report.

# In[8]:


# I look at the settings my better model chose for each hidden battery.
folds_best = results[(best_name, 'LOCO')].folds[['fold', 'held_out', 'n_test', 'MAE', 'n_selected', 'best_params']]
print(f'Settings my {best_name} chose for each hidden battery:')
display(folds_best.round(3))


# ## 8. My final honesty check
# 
# At the start of the project, 3 batteries were locked away and never touched. Here I test my final model on them one time only. This is just a safety check. I never use it to choose my model, because that would defeat the purpose. One note: these 3 batteries never got badly worn out, so this is an easy test.

# In[9]:


# I run my final model one time on the 3 locked-away batteries.
ho = ev.holdout_check(df, best_spec)
print('Final check on the 3 locked-away batteries:')
for k, v in ho.items():
    print(f'  {k}: {v}')


# ## 9. My charts
# 
# I draw two charts. The first shows the error of each model under each test method, side by side. The second shows predicted versus actual SOH for my better model. Points sitting on the diagonal line are perfect guesses, and points far from it are mistakes.

# In[10]:


# I draw and save my two charts.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Chart 1: error of each model under each test method.
piv = summary.pivot(index='model', columns='scheme', values='MAE')
piv = piv.reindex([x for x in ['Random forest', 'HistGradientBoosting'] if x in piv.index])
piv = piv[[s for s in ['Random 5-fold', 'LOCO', 'LOPO'] if s in piv.columns]]
ax = piv.plot(kind='bar', figsize=(8, 5))
ax.set_ylabel('Error (SOH percent points)')
ax.set_title('Error by model and test method (lower is better)')
ax.set_xlabel('')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / 'figures' / 'mae_by_model_and_split.png', dpi=130)
plt.show()

# Chart 2: predicted vs actual SOH for my better model.
fig, ax = plt.subplots(figsize=(6, 6))
for reg, g in oof_best.groupby('regime'):
    ax.scatter(g.y_true, g.y_pred, s=18, alpha=0.7, label=reg)
lims = [oof_best[['y_true', 'y_pred']].min().min() - 2, oof_best[['y_true', 'y_pred']].max().max() + 2]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlabel('Actual SOH (percent)')
ax.set_ylabel('Predicted SOH (percent)')
ax.set_title(f'{best_name}: predicted vs actual')
ax.legend()
plt.tight_layout()
plt.savefig(OUT / 'figures' / 'pred_vs_actual_loco.png', dpi=130)
plt.show()
print('I saved my charts to', OUT / 'figures')


# ## 10. I save everything
# 
# I save all my tables, predictions, and charts to the results/modelling_rf_hgb folder, so a teammate can use my models in the combined comparison without running anything again.

# In[11]:


# I save all my results so the team can use them.
all_oof = pd.concat([r.oof for r in all_results], ignore_index=True)
all_folds = pd.concat([r.folds for r in all_results], ignore_index=True)

all_oof.to_csv(OUT / 'oof_rf_hgb.csv', index=False)
all_folds.to_csv(OUT / 'per_fold_rf_hgb.csv', index=False)
summary.to_csv(OUT / 'metrics_rf_hgb.csv', index=False)
ev.breakdown(oof_best, 'health_class').to_csv(OUT / 'breakdown_health_class.csv', index=False)
ev.breakdown(oof_best, 'regime').to_csv(OUT / 'breakdown_regime.csv', index=False)
ev.breakdown(oof_best, 'cell_id').to_csv(OUT / 'breakdown_per_cell.csv', index=False)
cmp.to_csv(OUT / 'paired_comparison_loco.csv', index=False)
pd.DataFrame([ho]).to_csv(OUT / 'holdout_check.csv', index=False)

print('I saved all my results to', OUT)
for p in sorted(OUT.glob('*.csv')):
    print(' ', p.name)


# ## My summary in simple words
# 
# 1. How good are my models on a brand new battery (the real test)? HistGradientBoosting is off by about 4.3 percent on average, and Random Forest by about 4.7 percent. For comparison, just guessing the average battery would be off by about 8.2 percent (that average guess is Samina's baseline), so both of my models clearly help.
# 
# 2. Which model is better? They are basically tied. HistGradientBoosting is a tiny bit ahead, but the head to head test says that small gap could just be luck (p = 0.93). So I treat them as equally good, with a slight preference for HistGradientBoosting.
# 
# 3. Why are my numbers bigger than the old notebook 04 (about 2 percent)? Because I used the team's strict and fair testing recipe. The old notebook used an easier setup. My number is the honest one. For comparison, the project's Ridge model scores 4.76 on this same recipe, so my 4.27 is a little better.
# 
# 4. When does my model struggle? On batteries used in conditions I have never seen (error about 10.7 percent) and on batteries near the end of life (error about 13.4 percent). In those cases it tends to be too optimistic, meaning it thinks the battery is healthier than it really is. This is worth pointing out, because that is exactly where I would most want it to be right.
# 
# 5. Final check: about 2.9 percent error on the 3 locked-away batteries. But that is an easier set, so I treat it only as a safety check.
# 
# Bottom line: both of my tree models work well on unseen batteries, and Random Forest and HistGradientBoosting are basically tied. They can be trusted for healthy and medium aged batteries, but they are less reliable for very worn out batteries or brand new operating conditions, and in those cases they lean optimistic.
# 
# To run this notebook again: jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=soh-venv notebooks/06_modelling_random_forest_hgb.ipynb

# ## 11. Error scores and accuracy explained (MAE, RMSE, R2, accuracy)
# 
# Here I calculate my error and accuracy scores by hand, so it is clear what each number means and how it is worked out. I use the predictions I saved earlier in this notebook, so nothing is re-trained.
# 
# 1. MAE, the average size of my mistakes.
# 2. RMSE, like MAE but it punishes big mistakes more.
# 3. R2, an overall quality score out of 1.
# 4. Accuracy, how close my guesses are, written as a percent.

# ### MAE (Mean Absolute Error)
# 
# MAE is the average size of my mistakes. For each battery I take how far my guess was from the true health, I ignore whether it was too high or too low, and then I take the average.
# 
# Formula: MAE = (add up all the gaps) divided by (number of guesses)
# 
# Tiny example. Suppose the true health is 80, 90, 70 and I guessed 78, 93, 76.
# - Gaps (ignoring sign): 2, 3, 6
# - Add them: 2 + 3 + 6 = 11
# - Divide by 3: 11 / 3 = 3.67
# 
# So the MAE would be 3.67. The smaller this is, the better.

# ### RMSE (Root Mean Squared Error)
# 
# RMSE is similar to MAE, but it cares more about big mistakes, because it squares each gap first. A few large misses push RMSE up more than they push MAE up.
# 
# Formula: RMSE = square root of ( (add up the squared gaps) divided by (number of guesses) )
# 
# Same tiny example (the gaps were 2, 3, 6):
# - Square each gap: 4, 9, 36
# - Average them: (4 + 9 + 36) / 3 = 16.33
# - Take the square root: about 4.04
# 
# So RMSE would be about 4.04. It is a little bigger than the MAE (3.67) because of the one larger miss of 6.

# ### R2 (R-squared, the overall quality score)
# 
# R2 tells me how much better my model is than just always guessing the average health. It is on a scale where:
# - 1.0 means perfect,
# - 0 means no better than just guessing the average,
# - below 0 means worse than guessing the average.
# 
# Formula: R2 = 1 - ( my total squared error ) divided by ( the total squared error of just guessing the average )
# 
# In plain words: I compare my mistakes against the mistakes a lazy model would make if it always guessed the average. If my mistakes are much smaller, R2 is close to 1.
# 
# Tiny example (true 80, 90, 70; the average is 80; my guesses 78, 93, 76):
# - My squared errors: 4, 9, 36, which add up to 49.
# - The average-guess squared errors: (80-80), (90-80), (70-80) squared = 0, 100, 100, which add up to 200.
# - R2 = 1 - (49 / 200) = 1 - 0.245 = about 0.76.

# ### Accuracy
# 
# My task predicts a number, not a yes or no, so there is no simple right-or-wrong accuracy. Instead I measure how close my guesses are, in two easy ways.
# 
# Way 1, relative accuracy: on average, how many percent off is each guess?
# - For each battery: the gap divided by the true value, written as a percent.
# - Example (true 80, 90, 70; guesses 78, 93, 76): 2.5%, 3.3%, 8.6%, which averages about 4.8% off.
# - So the relative accuracy is about 100 - 4.8 = 95.2%.
# 
# Way 2, within a tolerance: what share of my guesses land within 5 health points of the truth?
# - In the example, 2 of the 3 guesses are within 5 points, so that is about 67%.

# In[12]:


# I load my saved predictions so I can calculate the scores myself.
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / 'results').exists() and (ROOT.parent / 'results').exists():
    ROOT = ROOT.parent
oof = pd.read_csv(ROOT / 'results' / 'modelling_random_forest_hgb' / 'oof_rf_hgb.csv')

# I focus on my best model (HistGradientBoosting) on the main new-battery test (LOCO).
mine = oof[(oof['model'] == 'HistGradientBoosting') & (oof['scheme'] == 'LOCO')]
true = mine['y_true'].to_numpy()
pred = mine['y_pred'].to_numpy()
print('number of predictions I am scoring:', len(mine))


# In[13]:


# MAE: the average size of my mistakes (ignoring plus or minus).
gaps = np.abs(pred - true)          # how far each guess is from the truth
mae = gaps.mean()                   # the average of those gaps
print('MAE (average error) =', round(mae, 2), 'health points')


# In[14]:


# RMSE: like MAE, but big mistakes count more because I square the gaps first.
squared_gaps = (pred - true) ** 2
rmse = np.sqrt(squared_gaps.mean())
print('RMSE =', round(rmse, 2), 'health points')


# In[15]:


# R2: how much better my model is than just always guessing the average health.
mean_true = true.mean()
ss_res = ((true - pred) ** 2).sum()         # my total squared error
ss_tot = ((true - mean_true) ** 2).sum()    # total squared error of guessing the average
r2 = 1 - ss_res / ss_tot
print('R2 =', round(r2, 2))


# In[16]:


# Accuracy, way 1: on average, how many percent off is each guess?
relative_off = np.abs(pred - true) / true * 100
avg_off = relative_off.mean()
relative_accuracy = 100 - avg_off
print('average off by     =', round(avg_off, 1), '%')
print('relative accuracy  =', round(relative_accuracy, 1), '%')

# Accuracy, way 2: how many of my guesses land close to the truth?
within_5 = (np.abs(pred - true) <= 5).mean() * 100
within_10 = (np.abs(pred - true) <= 10).mean() * 100
print('within 5 points    =', round(within_5), '% of guesses')
print('within 10 points   =', round(within_10), '% of guesses')


# ### All my scores in one table

# In[17]:


# I put all my scores for the best model (HistGradientBoosting, new-battery test) together.
scores = pd.DataFrame({
    'metric': ['MAE (average error)', 'RMSE', 'R2 (quality score)',
               'Relative accuracy', 'Within 5 points', 'Within 10 points'],
    'value': [f'{mae:.2f} points', f'{rmse:.2f} points', f'{r2:.2f} (out of 1)',
              f'{relative_accuracy:.1f} %', f'{within_5:.0f} %', f'{within_10:.0f} %'],
})
scores


# In[18]:


# For completeness, here are MAE, RMSE and R2 for both of my models on the main test.
both = pd.read_csv(ROOT / 'results' / 'modelling_random_forest_hgb' / 'metrics_rf_hgb.csv')
both = both[both['scheme'] == 'LOCO'][['model', 'MAE', 'RMSE', 'R2']].round(2).reset_index(drop=True)
both


# ### Summary of my error and accuracy scores
# 
# - MAE (average error) of my best model is about 4.3 health points. Each guess is usually about 4.3 points away from the true health.
# - RMSE is about 7.7 points. It is bigger than MAE because a few very worn out batteries cause larger misses.
# - R2 is about 0.52, which means my model explains about half of the real ups and downs in battery health, and is clearly better than just guessing the average.
# - Relative accuracy is about 94 percent, meaning on average each guess is only about 6 percent off.
# - About 76 percent of my guesses land within 5 health points of the truth, and about 89 percent within 10 points.
# 
# So my models are quite close on most batteries, with the larger errors coming from the very worn out ones.
