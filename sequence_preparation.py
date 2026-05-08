import numpy as np
import pandas as pd

PREDICTION_DAY_CUTOFF = 90
SEQ_LEN = PREDICTION_DAY_CUTOFF + 1
TARGET_MAP = {'Pass': 0, 'Distinction': 1, 'Fail': 2, 'Withdrawn': 3}

# Load raw data
student_info = pd.read_csv('studentInfo.csv')
student_vle = pd.read_csv('studentVle.csv')
student_assess = pd.read_csv('studentAssessment.csv')
assessments = pd.read_csv('assessments.csv')
student_reg = pd.read_csv('studentRegistration.csv')

# Convert poorly typed columns to numeric
student_vle['date'] = pd.to_numeric(student_vle['date'], errors='coerce')
student_vle['sum_click'] = pd.to_numeric(student_vle['sum_click'], errors='coerce')
student_assess['date_submitted'] = pd.to_numeric(student_assess['date_submitted'], errors='coerce')
student_assess['score'] = pd.to_numeric(student_assess['score'], errors='coerce')
student_reg['date_registration'] = pd.to_numeric(student_reg['date_registration'], errors='coerce')
student_reg['date_unregistration'] = pd.to_numeric(student_reg['date_unregistration'], errors='coerce')

# Filter to the early prediction window
student_vle = student_vle[(student_vle['date'] >= 0) & (student_vle['date'] <= PREDICTION_DAY_CUTOFF)]

# Keep only registered students who are still active at the cutoff day
active_reg = student_reg[
    student_reg['date_unregistration'].isna() |
    (student_reg['date_unregistration'] > PREDICTION_DAY_CUTOFF)
].copy()

# Build the target / static feature frame
master = student_info.merge(
    active_reg[['code_module', 'code_presentation', 'id_student', 'date_registration']],
    on=['code_module', 'code_presentation', 'id_student'],
    how='inner'
)
master = master[master['final_result'].isin(TARGET_MAP)]
master['target'] = master['final_result'].map(TARGET_MAP)

# Build daily sequence features for studentVle
vle_daily = (
    student_vle
    .groupby(['code_module', 'code_presentation', 'id_student', 'date'], as_index=False)
    ['sum_click']
    .sum()
)

# Pivot to day sequence
vle_pivot = vle_daily.pivot_table(
    index=['code_module', 'code_presentation', 'id_student'],
    columns='date',
    values='sum_click',
    aggfunc='sum',
    fill_value=0
)

# Ensure columns are ordered from 0..PREDICTION_DAY_CUTOFF
vle_pivot = vle_pivot.reindex(columns=range(SEQ_LEN), fill_value=0)

# Build daily assessment features
assess_merged = student_assess.merge(
    assessments[['id_assessment', 'code_module', 'code_presentation', 'date']],
    on='id_assessment',
    how='left'
)
assess_early = assess_merged[
    (assess_merged['date_submitted'] >= 0) &
    (assess_merged['date_submitted'] <= PREDICTION_DAY_CUTOFF)
].copy()

assess_daily = (
    assess_early
    .groupby(['code_module', 'code_presentation', 'id_student', 'date_submitted'], as_index=False)
    .agg(
        num_submissions=('id_assessment', 'count'),
        mean_score=('score', 'mean')
    )
    .rename(columns={'date_submitted': 'date'})
)

assess_submissions = assess_daily.pivot_table(
    index=['code_module', 'code_presentation', 'id_student'],
    columns='date',
    values='num_submissions',
    aggfunc='sum',
    fill_value=0
).reindex(columns=range(SEQ_LEN), fill_value=0)

assess_score = assess_daily.pivot_table(
    index=['code_module', 'code_presentation', 'id_student'],
    columns='date',
    values='mean_score',
    aggfunc='mean',
    fill_value=0
).reindex(columns=range(SEQ_LEN), fill_value=0)

# Combine IDs with static features and targets
static_cols = [
    'num_of_prev_attempts',
    'studied_credits',
    'date_registration'
]

master_static = master[[
    'code_module', 'code_presentation', 'id_student', 'target'
] + static_cols].copy()

# Align sequence data to master rows
merged_vle = master_static.merge(
    vle_pivot.reset_index(),
    on=['code_module', 'code_presentation', 'id_student'],
    how='left'
)
merged_vle_columns = [c for c in merged_vle.columns if isinstance(c, int)]
merged_vle[merged_vle_columns] = merged_vle[merged_vle_columns].fillna(0)

merged_assess_sub = master_static.merge(
    assess_submissions.reset_index(),
    on=['code_module', 'code_presentation', 'id_student'],
    how='left'
).fillna(0)
merged_assess_score = master_static.merge(
    assess_score.reset_index(),
    on=['code_module', 'code_presentation', 'id_student'],
    how='left'
).fillna(0)

# Build the final sequence tensor
X_seq = np.zeros((len(master_static), SEQ_LEN, 3), dtype=np.float32)
for i, row in master_static[['code_module', 'code_presentation', 'id_student']].iterrows():
    key = tuple(row)
    idx = i
    X_seq[idx, :, 0] = merged_vle.loc[idx, merged_vle_columns].to_numpy(dtype=np.float32)
    X_seq[idx, :, 1] = merged_assess_sub.loc[idx, merged_vle_columns].to_numpy(dtype=np.float32)
    X_seq[idx, :, 2] = merged_assess_score.loc[idx, merged_vle_columns].to_numpy(dtype=np.float32)

X_static = master_static[static_cols].astype(np.float32).to_numpy()
y = master_static['target'].to_numpy(dtype=np.int64)

np.savez_compressed(
    'sequence_dataset.npz',
    X_seq=X_seq,
    X_static=X_static,
    y=y,
    ids=master_static[['code_module', 'code_presentation', 'id_student']].to_numpy(dtype=object)
)

print('Saved sequence_dataset.npz')
print('X_seq shape:', X_seq.shape)
print('X_static shape:', X_static.shape)
print('y shape:', y.shape)
print('class balance:')
print(pd.Series(y).value_counts())
