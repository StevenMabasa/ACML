import pandas as pd

# =============================================================================
# STEP 0: DEFINE THE PREDICTION CUTOFF
# =============================================================================

PREDICTION_DAY_CUTOFF = 90
print(f"Early Prediction Pipeline (Multi-class) — Cutoff: Day {PREDICTION_DAY_CUTOFF}\n")


# =============================================================================
# STEP 1: LOAD ALL CSVs
# =============================================================================

print("Step 1: Loading raw CSV files...")

assessments    = pd.read_csv("assessments.csv")
courses        = pd.read_csv("courses.csv")
student_info   = pd.read_csv("studentInfo.csv")
student_assess = pd.read_csv("studentAssessment.csv")
student_reg    = pd.read_csv("studentRegistration.csv")
student_vle    = pd.read_csv("studentVle.csv")

# Convert CSV columns that should be numeric but may load as object/string
for col in ['date', 'module_presentation_length', 'id_assessment']:
    if col in assessments.columns:
        assessments[col] = pd.to_numeric(assessments[col], errors='coerce')
for col in ['id_student', 'id_site', 'sum_click', 'date']:
    if col in student_vle.columns:
        student_vle[col] = pd.to_numeric(student_vle[col], errors='coerce')
for col in ['score', 'is_banked', 'date_submitted', 'id_assessment', 'id_student']:
    if col in student_assess.columns:
        student_assess[col] = pd.to_numeric(student_assess[col], errors='coerce')
for col in ['date_registration', 'date_unregistration', 'id_student']:
    if col in student_reg.columns:
        student_reg[col] = pd.to_numeric(student_reg[col], errors='coerce')
for col in ['num_of_prev_attempts', 'studied_credits', 'id_student']:
    if col in student_info.columns:
        student_info[col] = pd.to_numeric(student_info[col], errors='coerce')

# =============================================================================
# STEP 2: CLEAN ASSESSMENTS
# =============================================================================

print("\nStep 2: Cleaning assessments...")

assessments = pd.merge(
    assessments,
    courses[['code_module', 'code_presentation', 'module_presentation_length']],
    on=['code_module', 'code_presentation'],
    how='left'
)

missing_date_mask = assessments['date'].isnull()
assessments.loc[missing_date_mask, 'date'] = (
    assessments.loc[missing_date_mask, 'module_presentation_length']
)

assessments.drop(columns=['module_presentation_length'], inplace=True)
print(f"  Exam dates filled using course length: {missing_date_mask.sum()} rows")


# =============================================================================
# STEP 3: CLEAN STUDENT INFO
# =============================================================================

print("\nStep 3: Cleaning student_info...")

n_missing_imd = student_info['imd_band'].isnull().sum()
student_info['imd_band'] = student_info['imd_band'].fillna('Unknown')
print(f"  imd_band: {n_missing_imd} missing → filled with 'Unknown'")


# =============================================================================
# STEP 4: CLEAN STUDENT ASSESSMENTS
# =============================================================================

print("\nStep 4: Removing banked assessments...")

n_banked = student_assess['is_banked'].sum()
student_assess = student_assess[student_assess['is_banked'] == 0].copy()
n_missing_score = student_assess['score'].isnull().sum()

print(f"  Banked rows removed        : {n_banked}")
print(f"  Missing scores (no submit) : {n_missing_score} — kept as NaN intentionally")


# =============================================================================
# STEP 5: CLEAN STUDENT REGISTRATION
# =============================================================================

print("\nStep 5: Cleaning student_reg...")

# Ensure registration and unregistration dates are numeric day counts
student_reg['date_registration'] = pd.to_numeric(
    student_reg['date_registration'],
    errors='coerce'
)
student_reg['date_unregistration'] = pd.to_numeric(
    student_reg['date_unregistration'],
    errors='coerce'
)

n_missing_reg = student_reg['date_registration'].isnull().sum()
median_registration = student_reg['date_registration'].median()
student_reg['date_registration'] = student_reg['date_registration'].fillna(
    median_registration
)
print(f"  date_registration: {n_missing_reg} missing → filled with median")


# =============================================================================
# STEP 6: AGGREGATE VLE INTERACTIONS (early window only)
# =============================================================================

print(f"\nStep 6: Aggregating VLE data (up to day {PREDICTION_DAY_CUTOFF})...")

student_vle_early = student_vle[student_vle['date'] <= PREDICTION_DAY_CUTOFF].copy()

vle_agg = (
    student_vle_early
    .groupby(['code_module', 'code_presentation', 'id_student'])
    .agg(
        total_clicks     = ('sum_click', 'sum'),
        active_days      = ('date',      'nunique'),   
        unique_resources = ('id_site',   'nunique'),   
        last_click_day   = ('date',      'max')        
    )
    .reset_index()
)

vle_agg['clicks_per_day'] = vle_agg['total_clicks'] / PREDICTION_DAY_CUTOFF
vle_agg['days_since_last_click'] = PREDICTION_DAY_CUTOFF - vle_agg['last_click_day']
vle_agg.drop(columns=['last_click_day'], inplace=True)

print(f"  VLE aggregated: {vle_agg.shape[0]} student-module rows")


# =============================================================================
# STEP 7: AGGREGATE ASSESSMENT SCORES (early window only)
# =============================================================================

print(f"\nStep 7: Aggregating assessments (submitted by day {PREDICTION_DAY_CUTOFF})...")

assess_merged = pd.merge(
    student_assess,
    assessments[['id_assessment', 'code_module', 'code_presentation', 'date']],
    on='id_assessment',
    how='left'
)

assess_early = assess_merged[
    assess_merged['date_submitted'] <= PREDICTION_DAY_CUTOFF
].copy()

assess_agg = (
    assess_early
    .groupby(['code_module', 'code_presentation', 'id_student'])['score']
    .mean()
    .reset_index()
    .rename(columns={'score': 'mean_score'})
)

submission_counts = (
    assess_early
    .dropna(subset=['score'])
    .groupby(['code_module', 'code_presentation', 'id_student'])['id_assessment']
    .count()
    .reset_index()
    .rename(columns={'id_assessment': 'num_submissions'})
)

total_due_before_cutoff = (
    assessments[assessments['date'] <= PREDICTION_DAY_CUTOFF]
    .groupby(['code_module', 'code_presentation'])['id_assessment']
    .count()
    .reset_index()
    .rename(columns={'id_assessment': 'total_assessments_due'})
)

assess_agg = pd.merge(assess_agg, submission_counts, on=['code_module', 'code_presentation', 'id_student'], how='left')
assess_agg = pd.merge(assess_agg, total_due_before_cutoff, on=['code_module', 'code_presentation'], how='left')

assess_agg['num_submissions']   = assess_agg['num_submissions'].fillna(0)
assess_agg['submission_rate']   = (
    assess_agg['num_submissions'] / assess_agg['total_assessments_due']
).fillna(0)

assess_agg.drop(columns=['total_assessments_due'], inplace=True)

print(f"  Assessment aggregated: {assess_agg.shape[0]} student-module rows")


# =============================================================================
# STEP 8: BUILD THE MASTER DATAFRAME
# =============================================================================

print("\nStep 8: Building master DataFrame...")

master_df = student_info.copy()

master_df = pd.merge(master_df, vle_agg, on=['code_module', 'code_presentation', 'id_student'], how='left')
master_df = pd.merge(master_df, assess_agg, on=['code_module', 'code_presentation', 'id_student'], how='left')

master_df = pd.merge(
    master_df,
    student_reg[['code_module', 'code_presentation', 'id_student', 'date_registration', 'date_unregistration']],
    on=['code_module', 'code_presentation', 'id_student'],
    how='left'
)

master_df['total_clicks']           = master_df['total_clicks'].fillna(0)
master_df['clicks_per_day']         = master_df['clicks_per_day'].fillna(0)
master_df['active_days']            = master_df['active_days'].fillna(0)
master_df['unique_resources']       = master_df['unique_resources'].fillna(0)
master_df['days_since_last_click']  = master_df['days_since_last_click'].fillna(PREDICTION_DAY_CUTOFF)

master_df['mean_score']             = master_df['mean_score'].fillna(0)
master_df['num_submissions']        = master_df['num_submissions'].fillna(0)
master_df['submission_rate']        = master_df['submission_rate'].fillna(0)

print(f"  Master DataFrame shape (pre-filter): {master_df.shape}")


# =============================================================================
# STEP 9: DUPLICATE CHECK
# =============================================================================

print("\nStep 9: Checking for duplicate rows...")

key_cols = ['code_module', 'code_presentation', 'id_student']
n_dupes = master_df.duplicated(subset=key_cols).sum()
if n_dupes > 0:
    print(f"  WARNING: {n_dupes} duplicate rows — investigate the merge logic!")
else:
    print("  No duplicate rows found. ✓")


# =============================================================================
# STEP 10: FILTER TO STUDENTS STILL ACTIVE AT THE CUTOFF DAY
# =============================================================================

print(f"\nStep 10: Filtering to students still active at day {PREDICTION_DAY_CUTOFF}...")

active_mask = (
    master_df['date_unregistration'].isnull() |
    (master_df['date_unregistration'] > PREDICTION_DAY_CUTOFF)   
)
early_df = master_df[active_mask].copy()

n_removed = master_df.shape[0] - early_df.shape[0]
print(f"  Original rows     : {master_df.shape[0]}")
print(f"  Rows removed      : {n_removed} (withdrew on or before day {PREDICTION_DAY_CUTOFF})")
print(f"  Prediction sample : {early_df.shape[0]}")


# =============================================================================
# STEP 11: ENCODE THE TARGET VARIABLE
# =============================================================================

print("\nStep 11: Encoding target variable (Multi-class)...")

target_map = {'Pass': 0, 'Distinction': 1, 'Fail': 2, 'Withdrawn': 3}
early_df['target'] = early_df['final_result'].map(target_map)

print(f"  Target distribution (counts):\n{early_df['target'].value_counts()}")
print(f"  Target distribution (ratio) :\n{early_df['target'].value_counts(normalize=True).round(3)}")


# =============================================================================
# STEP 12: DROP REDUNDANT AND LEAKY COLUMNS
# =============================================================================

print("\nStep 12: Dropping redundant and leaky columns...")

cols_to_drop = ['id_student', 'final_result', 'date_unregistration']
df_features = early_df.drop(columns=cols_to_drop)

print(f"  Dropped  : {cols_to_drop}")
print(f"  Kept ({df_features.shape[1]} cols): {list(df_features.columns)}")


# =============================================================================
# STEP 13: ONE-HOT ENCODE CATEGORICAL COLUMNS (TWO VERSIONS)
# =============================================================================

print("\nStep 13: One-hot encoding categorical columns...")

cat_cols = [
    'code_module',
    'code_presentation',
    'gender',
    'region',
    'highest_education',
    'imd_band',
    'age_band',
    'disability'
]

df_encoded_linear = pd.get_dummies(df_features, columns=cat_cols, drop_first=True)
df_encoded_tree   = pd.get_dummies(df_features, columns=cat_cols, drop_first=False)

print(f"  Pre-encoding shape          : {df_features.shape}")
print(f"  Encoded shape (linear/SVM)  : {df_encoded_linear.shape}")
print(f"  Encoded shape (tree-based)  : {df_encoded_tree.shape}")


# =============================================================================
# STEP 14: FINAL SANITY CHECK
# =============================================================================

print("\n" + "=" * 65)
print(f"PIPELINE COMPLETE — Early Prediction (Multi-class) at Day {PREDICTION_DAY_CUTOFF}")
print("=" * 65)

nulls_linear = df_encoded_linear.isnull().sum().sum()
nulls_tree   = df_encoded_tree.isnull().sum().sum()

print(f"\n  df_encoded_linear shape  : {df_encoded_linear.shape}")
print(f"  df_encoded_tree   shape  : {df_encoded_tree.shape}")
print(f"\n  Target balance:")

# FIX: Update the rename dictionary to match the 4 new classes
class_names = {0: 'Pass (0)', 1: 'Distinction (1)', 2: 'Fail (2)', 3: 'Withdrawn (3)'}
print(early_df['target'].value_counts(normalize=True).rename(class_names).round(3))

print(f"\n  Null values (linear) : {nulls_linear}")
print(f"  Null values (tree)   : {nulls_tree}")

if nulls_linear == 0 and nulls_tree == 0:
    print("\n  All clear — no null values detected. Ready for model training. ✓")
else:
    print("\n  WARNING: Null values detected. Investigate before training.")

print(f"""
Features available at prediction time (day {PREDICTION_DAY_CUTOFF}):
  Demographic  : gender, region, highest_education, imd_band, age_band,
                 disability, num_of_prev_attempts, studied_credits
  Registration : date_registration
  VLE          : total_clicks, clicks_per_day, active_days,
                 unique_resources, days_since_last_click
  Assessment   : mean_score, num_submissions, submission_rate
  Course       : code_module, code_presentation
""")

# =============================================================================
# STEP 15: SAVE FINAL DATASETS
# =============================================================================

print("\nStep 15: Saving datasets...")

master_df.to_csv("master_dataset.csv", index=False)
early_df.to_csv("early_prediction_dataset.csv", index=False)

df_encoded_linear.to_csv("ml_dataset_linear.csv", index=False)
df_encoded_tree.to_csv("ml_dataset_tree.csv", index=False)

print("  Datasets saved successfully. ✓")