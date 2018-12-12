---
title: Models
notebook: models.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}


# Models



```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import datetime

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import Imputer

from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm
from statsmodels.api import OLS
```




```python
def display_df(df, nrows=5, ncols=None):
    with pd.option_context('display.max_rows', nrows, 'display.max_columns', ncols):
        display (df)
#gets ratio of NaNs for each column
def stats_NaN(df):
    df_stats = pd.DataFrame(index=[df.columns], columns=["NaN Ratio"])
    for col in df.columns:
        df_stats["NaN Ratio"][col] = df[col].isna().sum()/len(df) #NaN ratio
    return df_stats.sort_values(by=['NaN Ratio'])
```




```python
df_whole = pd.read_csv("../data/data_clean/clean_accepted_2007_to_2018Q2.csv")
```


    /anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (38,48,107,118,119,120,123,124,125,128,134,135,136) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)




```python
stats_nan = stats_NaN(df_whole)
```




```python
display_df(stats_nan, None)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NaN Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>addr_state_DC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_debt_consolidation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_educational</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_home_improvement</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_house</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_major_purchase</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_medical</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_moving</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_other</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_credit_card</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_renewable_energy</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_vacation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_wedding</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AK</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AZ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CO</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_small_business</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_car</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_Late (31-120 days)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_Late (16-30 days)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_ANY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_MORTGAGE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_NONE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_OTHER</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_OWN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_RENT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_recent_revol_delinq</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_recent_inq</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_recent_bc_dlq</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Not Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Source Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_Charged Off</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_Current</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_Default</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_Does not meet the credit policy. Status:Charged Off</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_Does not meet the credit policy. Status:Fully Paid</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_Fully Paid</th>
      <td>0</td>
    </tr>
    <tr>
      <th>loan_status_In Grace Period</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_DE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NM</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NV</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OH</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OK</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_PA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_FL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_RI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_SD</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_TN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_TX</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_UT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_VA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_VT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_SC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NJ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NH</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_GA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_HI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_KS</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_KY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_LA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MD</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ME</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MO</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MS</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ND</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WV</th>
      <td>0</td>
    </tr>
    <tr>
      <th>disbursement_method</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>hardship_flag</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>debt_settlement_flag</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>loan_amnt</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>num_grade</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>total_rec_prncp</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>fico_range_low</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>fico_range_high</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>initial_list_status</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>out_prncp</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>out_prncp_inv</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>total_pymnt</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>pymnt_plan</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>total_pymnt_inv</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>total_rec_int</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>total_rec_late_fee</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>recoveries</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>collection_recovery_fee</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>last_pymnt_amnt</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>last_fico_range_high</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>last_fico_range_low</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>application_type</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>issue_d</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>policy_code</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>sub_grade</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>funded_amnt</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>funded_amnt_inv</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>term</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>zip_code</th>
      <td>1.39714e-05</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>1.54684e-05</td>
    </tr>
    <tr>
      <th>delinq_amnt</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>acc_now_delinq</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>earliest_cr_line</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>2.84419e-05</td>
    </tr>
    <tr>
      <th>last_credit_pull_d</th>
      <td>5.0397e-05</td>
    </tr>
    <tr>
      <th>tax_liens</th>
      <td>6.58653e-05</td>
    </tr>
    <tr>
      <th>chargeoff_within_12_mths</th>
      <td>8.58245e-05</td>
    </tr>
    <tr>
      <th>collections_12_mths_ex_med</th>
      <td>8.58245e-05</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0.000595782</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>0.00069458</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>0.000770425</td>
    </tr>
    <tr>
      <th>last_pymnt_d</th>
      <td>0.00108428</td>
    </tr>
    <tr>
      <th>total_bal_ex_mort</th>
      <td>0.0249774</td>
    </tr>
    <tr>
      <th>total_bc_limit</th>
      <td>0.0249774</td>
    </tr>
    <tr>
      <th>acc_open_past_24mths</th>
      <td>0.0249774</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>0.0249774</td>
    </tr>
    <tr>
      <th>num_bc_sats</th>
      <td>0.0292487</td>
    </tr>
    <tr>
      <th>num_sats</th>
      <td>0.0292487</td>
    </tr>
    <tr>
      <th>mths_since_recent_bc</th>
      <td>0.0350773</td>
    </tr>
    <tr>
      <th>num_accts_ever_120_pd</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_rev_tl_bal_gt_0</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_tl_90g_dpd_24m</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_tl_op_past_12m</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>tot_coll_amt</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>tot_cur_bal</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_op_rev_tl</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_actv_bc_tl</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>total_rev_hi_lim</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>tot_hi_cred_lim</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>total_il_high_credit_limit</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_tl_30dpd</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_actv_rev_tl</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>mo_sin_rcnt_tl</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_il_tl</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_bc_tl</th>
      <td>0.0350798</td>
    </tr>
    <tr>
      <th>num_rev_accts</th>
      <td>0.0350803</td>
    </tr>
    <tr>
      <th>mo_sin_rcnt_rev_tl_op</th>
      <td>0.0350803</td>
    </tr>
    <tr>
      <th>mo_sin_old_rev_tl_op</th>
      <td>0.0350803</td>
    </tr>
    <tr>
      <th>avg_cur_bal</th>
      <td>0.0351022</td>
    </tr>
    <tr>
      <th>pct_tl_nvr_dlq</th>
      <td>0.0351566</td>
    </tr>
    <tr>
      <th>bc_open_to_buy</th>
      <td>0.0357389</td>
    </tr>
    <tr>
      <th>percent_bc_gt_75</th>
      <td>0.035959</td>
    </tr>
    <tr>
      <th>bc_util</th>
      <td>0.0362499</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>0.0622941</td>
    </tr>
    <tr>
      <th>mo_sin_old_il_acct</th>
      <td>0.0648719</td>
    </tr>
    <tr>
      <th>num_tl_120dpd_2m</th>
      <td>0.073635</td>
    </tr>
    <tr>
      <th>next_pymnt_d</th>
      <td>0.42112</td>
    </tr>
    <tr>
      <th>open_rv_12m</th>
      <td>0.432194</td>
    </tr>
    <tr>
      <th>total_bal_il</th>
      <td>0.432194</td>
    </tr>
    <tr>
      <th>open_il_24m</th>
      <td>0.432194</td>
    </tr>
    <tr>
      <th>open_il_12m</th>
      <td>0.432194</td>
    </tr>
    <tr>
      <th>open_act_il</th>
      <td>0.432194</td>
    </tr>
    <tr>
      <th>max_bal_bc</th>
      <td>0.432194</td>
    </tr>
    <tr>
      <th>inq_fi</th>
      <td>0.432194</td>
    </tr>
    <tr>
      <th>open_rv_24m</th>
      <td>0.432194</td>
    </tr>
    <tr>
      <th>inq_last_12m</th>
      <td>0.432195</td>
    </tr>
    <tr>
      <th>total_cu_tl</th>
      <td>0.432195</td>
    </tr>
    <tr>
      <th>open_acc_6m</th>
      <td>0.432195</td>
    </tr>
    <tr>
      <th>all_util</th>
      <td>0.432271</td>
    </tr>
    <tr>
      <th>mths_since_rcnt_il</th>
      <td>0.449512</td>
    </tr>
    <tr>
      <th>il_util</th>
      <td>0.512996</td>
    </tr>
    <tr>
      <th>mths_since_last_major_derog</th>
      <td>0.73965</td>
    </tr>
    <tr>
      <th>annual_inc_joint</th>
      <td>0.957016</td>
    </tr>
    <tr>
      <th>dti_joint</th>
      <td>0.957018</td>
    </tr>
    <tr>
      <th>verification_status_joint</th>
      <td>0.95751</td>
    </tr>
    <tr>
      <th>sec_app_collections_12_mths_ex_med</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_fico_range_high</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_earliest_cr_line</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_inq_last_6mths</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_mort_acc</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_open_acc</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_open_act_il</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_num_rev_accts</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_chargeoff_within_12_mths</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_fico_range_low</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>revol_bal_joint</th>
      <td>0.963348</td>
    </tr>
    <tr>
      <th>sec_app_revol_util</th>
      <td>0.963981</td>
    </tr>
    <tr>
      <th>sec_app_mths_since_last_major_derog</th>
      <td>0.987438</td>
    </tr>
    <tr>
      <th>settlement_term</th>
      <td>0.993311</td>
    </tr>
    <tr>
      <th>settlement_percentage</th>
      <td>0.993311</td>
    </tr>
    <tr>
      <th>settlement_amount</th>
      <td>0.993311</td>
    </tr>
    <tr>
      <th>settlement_date</th>
      <td>0.993311</td>
    </tr>
    <tr>
      <th>settlement_status</th>
      <td>0.993311</td>
    </tr>
    <tr>
      <th>debt_settlement_flag_date</th>
      <td>0.993311</td>
    </tr>
    <tr>
      <th>hardship_end_date</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_last_payment_amount</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_payoff_balance_amount</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>payment_plan_start_date</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_loan_status</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_dpd</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_type</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_reason</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_status</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>deferral_term</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_amount</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_start_date</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>hardship_length</th>
      <td>0.997097</td>
    </tr>
    <tr>
      <th>orig_projected_additional_accrued_interest</th>
      <td>0.997574</td>
    </tr>
  </tbody>
</table>
</div>




```python
our_drop_list = ['funded_amnt','funded_amnt_inv','int_rate','installment','grade',
                 'pymnt_plan','zip_code','initial_list_status','out_prncp', 'application_type', 'policy_code',
                 'out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
                 'total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt',
                 'next_pymnt_d','last_credit_pull_d','last_fico_range_high','last_fico_range_low',
                 'collections_12_mths_ex_med','mths_since_last_major_derog','acc_now_delinq','tot_coll_amt',
                 'tot_cur_bal','open_acc_6m','open_act_il','open_il_12m','open_il_24m','mths_since_rcnt_il',
                 'total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim',
                 'inq_fi','total_cu_tl','inq_last_12m','acc_open_past_24mths','avg_cur_bal','bc_open_to_buy',
                 'bc_util','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct','mo_sin_old_rev_tl_op',
                 'mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc',
                 'mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq',
                 'num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl',
                 'num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m',
                 'num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75',
                 'pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit',
                 'total_il_high_credit_limit','revol_bal_joint','sec_app_fico_range_high',
                 'sec_app_earliest_cr_line','sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc',
                 'sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts',
                 'sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med',
                 'sec_app_mths_since_last_major_derog','hardship_flag','hardship_type','hardship_reason',
                 'hardship_status','deferral_term','hardship_amount','hardship_start_date','hardship_end_date',
                 'payment_plan_start_date','hardship_length','hardship_dpd','hardship_loan_status',
                 'orig_projected_additional_accrued_interest','hardship_payoff_balance_amount',
                 'hardship_last_payment_amount','disbursement_method','debt_settlement_flag',
                 'debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount',
                 'settlement_percentage','settlement_term','loan_status_Charged Off',
                 'loan_status_Current','loan_status_Default',
                 'loan_status_Does not meet the credit policy. Status:Charged Off',
                 'loan_status_Does not meet the credit policy. Status:Fully Paid',
                 'loan_status_Fully Paid','loan_status_In Grace Period','loan_status_Late (16-30 days)',
                 'loan_status_Late (31-120 days)']
```




```python
df_less_feats = df_whole.drop(columns=our_drop_list)
```




```python
display_df(df_less_feats)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>sub_grade</th>
      <th>emp_length</th>
      <th>annual_inc</th>
      <th>issue_d</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
      <th>inq_last_6mths</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>annual_inc_joint</th>
      <th>dti_joint</th>
      <th>verification_status_joint</th>
      <th>sec_app_fico_range_low</th>
      <th>home_ownership_ANY</th>
      <th>home_ownership_MORTGAGE</th>
      <th>home_ownership_NONE</th>
      <th>home_ownership_OTHER</th>
      <th>home_ownership_OWN</th>
      <th>home_ownership_RENT</th>
      <th>verification_status_Not Verified</th>
      <th>verification_status_Source Verified</th>
      <th>verification_status_Verified</th>
      <th>purpose_car</th>
      <th>purpose_credit_card</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_educational</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_renewable_energy</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
      <th>addr_state_AK</th>
      <th>addr_state_AL</th>
      <th>addr_state_AR</th>
      <th>addr_state_AZ</th>
      <th>addr_state_CA</th>
      <th>addr_state_CO</th>
      <th>addr_state_CT</th>
      <th>addr_state_DC</th>
      <th>addr_state_DE</th>
      <th>addr_state_FL</th>
      <th>addr_state_GA</th>
      <th>addr_state_HI</th>
      <th>addr_state_IA</th>
      <th>addr_state_ID</th>
      <th>addr_state_IL</th>
      <th>addr_state_IN</th>
      <th>addr_state_KS</th>
      <th>addr_state_KY</th>
      <th>addr_state_LA</th>
      <th>addr_state_MA</th>
      <th>addr_state_MD</th>
      <th>addr_state_ME</th>
      <th>addr_state_MI</th>
      <th>addr_state_MN</th>
      <th>addr_state_MO</th>
      <th>addr_state_MS</th>
      <th>addr_state_MT</th>
      <th>addr_state_NC</th>
      <th>addr_state_ND</th>
      <th>addr_state_NE</th>
      <th>addr_state_NH</th>
      <th>addr_state_NJ</th>
      <th>addr_state_NM</th>
      <th>addr_state_NV</th>
      <th>addr_state_NY</th>
      <th>addr_state_OH</th>
      <th>addr_state_OK</th>
      <th>addr_state_OR</th>
      <th>addr_state_PA</th>
      <th>addr_state_RI</th>
      <th>addr_state_SC</th>
      <th>addr_state_SD</th>
      <th>addr_state_TN</th>
      <th>addr_state_TX</th>
      <th>addr_state_UT</th>
      <th>addr_state_VA</th>
      <th>addr_state_VT</th>
      <th>addr_state_WA</th>
      <th>addr_state_WI</th>
      <th>addr_state_WV</th>
      <th>addr_state_WY</th>
      <th>num_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15000.0</td>
      <td>1.0</td>
      <td>C1</td>
      <td>10.0</td>
      <td>78000.0</td>
      <td>2014-12-01</td>
      <td>0.1203</td>
      <td>0.0</td>
      <td>1994-08-01</td>
      <td>750.0</td>
      <td>754.0</td>
      <td>0.0</td>
      <td>800.0</td>
      <td>800.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>138008.0</td>
      <td>29.0</td>
      <td>17.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10400.0</td>
      <td>0.0</td>
      <td>A3</td>
      <td>8.0</td>
      <td>58000.0</td>
      <td>2014-12-01</td>
      <td>0.1492</td>
      <td>0.0</td>
      <td>1989-09-01</td>
      <td>710.0</td>
      <td>714.0</td>
      <td>2.0</td>
      <td>42.0</td>
      <td>800.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>6133.0</td>
      <td>31.6</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2004087</th>
      <td>12000.0</td>
      <td>0.0</td>
      <td>B3</td>
      <td>8.0</td>
      <td>36000.0</td>
      <td>2018-01-01</td>
      <td>0.1110</td>
      <td>1.0</td>
      <td>1998-05-01</td>
      <td>685.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>800.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>11648.0</td>
      <td>43.6</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>2004088</th>
      <td>14000.0</td>
      <td>0.0</td>
      <td>C2</td>
      <td>2.0</td>
      <td>80000.0</td>
      <td>2018-01-01</td>
      <td>0.0135</td>
      <td>0.0</td>
      <td>2007-07-01</td>
      <td>660.0</td>
      <td>664.0</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>800.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>1461.0</td>
      <td>4.1</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.2</td>
    </tr>
  </tbody>
</table>
<p>2004089 rows × 98 columns</p>
</div>


## Data Augmentation

There might be some value in having a secondary applicant. We will create a new variable from a check on `sec_app_fico_range_low` to detect a secondary applicant. We will also drop the other high NaN proportion features.



```python
def add_secondary(df, label='sec_app_fico_range_low'):
    df['secondary'] = df[label].apply(lambda x: int(not pd.isnull(x)))
```




```python
add_secondary(df_less_feats)
```




```python
drop_high_nan_feats = ['annual_inc_joint', 'dti_joint','verification_status_joint','sec_app_fico_range_low']
df_less_feats.drop(columns=drop_high_nan_feats, inplace=True)
```




```python
display_df(df_less_feats)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>sub_grade</th>
      <th>emp_length</th>
      <th>annual_inc</th>
      <th>issue_d</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
      <th>inq_last_6mths</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>home_ownership_ANY</th>
      <th>home_ownership_MORTGAGE</th>
      <th>home_ownership_NONE</th>
      <th>home_ownership_OTHER</th>
      <th>home_ownership_OWN</th>
      <th>home_ownership_RENT</th>
      <th>verification_status_Not Verified</th>
      <th>verification_status_Source Verified</th>
      <th>verification_status_Verified</th>
      <th>purpose_car</th>
      <th>purpose_credit_card</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_educational</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_renewable_energy</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
      <th>addr_state_AK</th>
      <th>addr_state_AL</th>
      <th>addr_state_AR</th>
      <th>addr_state_AZ</th>
      <th>addr_state_CA</th>
      <th>addr_state_CO</th>
      <th>addr_state_CT</th>
      <th>addr_state_DC</th>
      <th>addr_state_DE</th>
      <th>addr_state_FL</th>
      <th>addr_state_GA</th>
      <th>addr_state_HI</th>
      <th>addr_state_IA</th>
      <th>addr_state_ID</th>
      <th>addr_state_IL</th>
      <th>addr_state_IN</th>
      <th>addr_state_KS</th>
      <th>addr_state_KY</th>
      <th>addr_state_LA</th>
      <th>addr_state_MA</th>
      <th>addr_state_MD</th>
      <th>addr_state_ME</th>
      <th>addr_state_MI</th>
      <th>addr_state_MN</th>
      <th>addr_state_MO</th>
      <th>addr_state_MS</th>
      <th>addr_state_MT</th>
      <th>addr_state_NC</th>
      <th>addr_state_ND</th>
      <th>addr_state_NE</th>
      <th>addr_state_NH</th>
      <th>addr_state_NJ</th>
      <th>addr_state_NM</th>
      <th>addr_state_NV</th>
      <th>addr_state_NY</th>
      <th>addr_state_OH</th>
      <th>addr_state_OK</th>
      <th>addr_state_OR</th>
      <th>addr_state_PA</th>
      <th>addr_state_RI</th>
      <th>addr_state_SC</th>
      <th>addr_state_SD</th>
      <th>addr_state_TN</th>
      <th>addr_state_TX</th>
      <th>addr_state_UT</th>
      <th>addr_state_VA</th>
      <th>addr_state_VT</th>
      <th>addr_state_WA</th>
      <th>addr_state_WI</th>
      <th>addr_state_WV</th>
      <th>addr_state_WY</th>
      <th>num_grade</th>
      <th>secondary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15000.0</td>
      <td>1.0</td>
      <td>C1</td>
      <td>10.0</td>
      <td>78000.0</td>
      <td>2014-12-01</td>
      <td>0.1203</td>
      <td>0.0</td>
      <td>1994-08-01</td>
      <td>750.0</td>
      <td>754.0</td>
      <td>0.0</td>
      <td>800.0</td>
      <td>800.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>138008.0</td>
      <td>29.0</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10400.0</td>
      <td>0.0</td>
      <td>A3</td>
      <td>8.0</td>
      <td>58000.0</td>
      <td>2014-12-01</td>
      <td>0.1492</td>
      <td>0.0</td>
      <td>1989-09-01</td>
      <td>710.0</td>
      <td>714.0</td>
      <td>2.0</td>
      <td>42.0</td>
      <td>800.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>6133.0</td>
      <td>31.6</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2004087</th>
      <td>12000.0</td>
      <td>0.0</td>
      <td>B3</td>
      <td>8.0</td>
      <td>36000.0</td>
      <td>2018-01-01</td>
      <td>0.1110</td>
      <td>1.0</td>
      <td>1998-05-01</td>
      <td>685.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>800.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>11648.0</td>
      <td>43.6</td>
      <td>18.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2004088</th>
      <td>14000.0</td>
      <td>0.0</td>
      <td>C2</td>
      <td>2.0</td>
      <td>80000.0</td>
      <td>2018-01-01</td>
      <td>0.0135</td>
      <td>0.0</td>
      <td>2007-07-01</td>
      <td>660.0</td>
      <td>664.0</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>800.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>1461.0</td>
      <td>4.1</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2004089 rows × 95 columns</p>
</div>


The FICO scores are defined as a fixed-with range with a range_low and range_high value. Since these two variables are directly dependent, we will combine those into one averaged fico_avg feature.



```python
df_less_feats['fico_avg'] = df_less_feats[['fico_range_low', 'fico_range_high']].mean(axis=1)
df_less_feats.drop(columns=['fico_range_low', 'fico_range_high'], inplace=True)
```


`earliest_cr_line` may not be itself a useful feature (a date with no context). However it can tell us how long a person has had a credit line open, which is probably useful.



```python
def timedelta_to_day(t):
    if not pd.isnull(t):
        if isinstance(t, int):
            return t
        else:
            return t.days
    else:
        return np.nan
```




```python
df_less_feats['issue_d'] = pd.to_datetime(df_less_feats['issue_d'])
df_less_feats['earliest_cr_line'] = pd.to_datetime(df_less_feats['earliest_cr_line'])
df_less_feats['cr_line_hist'] = df_less_feats['issue_d'] - pd.to_datetime(df_less_feats['earliest_cr_line'])
df_less_feats['cr_line_hist'] = df_less_feats['cr_line_hist'].apply(timedelta_to_day)
```




```python
df_less_feats.drop(columns=['earliest_cr_line'],inplace=True)
```


## Data Prep

We need to deal with NaNs. We can drop the samples for NaN features when the NaN ratio of that feature is small. For the rest, which is just `emp_length`, we'll do mean imputation to keep things simple.



```python
stats_nan_less = stats_NaN(df_less_feats)
```




```python
display_df(stats_nan_less,None)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NaN Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>addr_state_DC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_KS</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_HI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_GA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_FL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_DE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_RI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CO</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AZ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_KY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_LA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MD</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OK</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OH</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NV</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NM</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NJ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NH</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MS</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MO</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ME</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ND</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AK</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_wedding</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_vacation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_ANY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_SC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_SD</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_TN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_TX</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_UT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_VA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_VT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WV</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>secondary</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_MORTGAGE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_NONE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_OWN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_small_business</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_renewable_energy</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_other</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_moving</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_medical</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_major_purchase</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_house</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_OTHER</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_home_improvement</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_debt_consolidation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_credit_card</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_car</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Source Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Not Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_RENT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_educational</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_PA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>num_grade</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>loan_amnt</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>issue_d</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>sub_grade</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>term</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>fico_avg</th>
      <td>1.34725e-05</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>1.54684e-05</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>cr_line_hist</th>
      <td>2.79429e-05</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>2.84419e-05</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0.000595782</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>0.000770425</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>0.0622941</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_less_feats.dropna(subset=['issue_d','annual_inc','delinq_2yrs','total_acc','open_acc','pub_rec','cr_line_hist',
               'inq_last_6mths','dti','revol_util'],inplace=True)
```




```python
stats_nan_less = stats_NaN(df_less_feats)
display_df(stats_nan_less,None)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NaN Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ND</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MS</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MO</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ME</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MD</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_LA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_KY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_KS</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_HI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_GA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NH</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NJ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NM</th>
      <td>0</td>
    </tr>
    <tr>
      <th>secondary</th>
      <td>0</td>
    </tr>
    <tr>
      <th>num_grade</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WV</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_VT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_VA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_UT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_FL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_TX</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_SD</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_SC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_RI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_PA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OK</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OH</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NV</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_TN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_DE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_DC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_OWN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_OTHER</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_NONE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_MORTGAGE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_ANY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>0</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>0</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>0</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>0</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>0</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>0</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>0</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0</td>
    </tr>
    <tr>
      <th>issue_d</th>
      <td>0</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>0</td>
    </tr>
    <tr>
      <th>sub_grade</th>
      <td>0</td>
    </tr>
    <tr>
      <th>term</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_RENT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>fico_avg</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Not Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CO</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AZ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AK</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_wedding</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_vacation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_small_business</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Source Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_renewable_energy</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_moving</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_medical</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_major_purchase</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_house</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_home_improvement</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_educational</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_debt_consolidation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_credit_card</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_car</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_other</th>
      <td>0</td>
    </tr>
    <tr>
      <th>cr_line_hist</th>
      <td>0</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>0.0617604</td>
    </tr>
  </tbody>
</table>
</div>


Most models can't handle datetime objects. So we will convert this to a float.



```python
df_less_feats['issue_d']=df_less_feats['issue_d'].map(datetime.datetime.toordinal)
```




```python
target = 'num_grade'
target_class = 'sub_grade'
df_train, df_test = train_test_split(df_less_feats, random_state=9001, test_size=0.2, 
                                     stratify=df_less_feats[target_class])
```




```python
cols_to_imp = ['emp_length']
```




```python
imp_mean = Imputer(copy=True, missing_values=np.nan, strategy='mean').fit(df_train[cols_to_imp]) #fit to training data
df_train[cols_to_imp] = imp_mean.transform(df_train[cols_to_imp])
df_test[cols_to_imp] = imp_mean.transform(df_test[cols_to_imp])
```


    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    /anaconda3/lib/python3.6/site-packages/pandas/core/indexing.py:543: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s
    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until




```python
display_df(stats_NaN(df_train),None) #should be all zeros
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NaN Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ND</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MS</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MO</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ME</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MD</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_MA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_LA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_KY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_KS</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_ID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_IA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_HI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_GA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NH</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NJ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NM</th>
      <td>0</td>
    </tr>
    <tr>
      <th>secondary</th>
      <td>0</td>
    </tr>
    <tr>
      <th>num_grade</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WV</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_WA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_VT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_VA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_UT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_FL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_TX</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_SD</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_SC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_RI</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_PA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OK</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_OH</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_NV</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_TN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_DE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_DC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_OWN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_OTHER</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_NONE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_MORTGAGE</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_ANY</th>
      <td>0</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>0</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>0</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>0</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>0</td>
    </tr>
    <tr>
      <th>home_ownership_RENT</th>
      <td>0</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>0</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>0</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>0</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0</td>
    </tr>
    <tr>
      <th>issue_d</th>
      <td>0</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>0</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>0</td>
    </tr>
    <tr>
      <th>sub_grade</th>
      <td>0</td>
    </tr>
    <tr>
      <th>term</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>0</td>
    </tr>
    <tr>
      <th>fico_avg</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Not Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CO</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_CA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AZ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AL</th>
      <td>0</td>
    </tr>
    <tr>
      <th>addr_state_AK</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_wedding</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_vacation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_small_business</th>
      <td>0</td>
    </tr>
    <tr>
      <th>verification_status_Source Verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_renewable_energy</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_moving</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_medical</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_major_purchase</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_house</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_home_improvement</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_educational</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_debt_consolidation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_credit_card</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_car</th>
      <td>0</td>
    </tr>
    <tr>
      <th>purpose_other</th>
      <td>0</td>
    </tr>
    <tr>
      <th>cr_line_hist</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_cols = list(set([target,target_class]).symmetric_difference(list(df_train.columns)))
x_train, x_test = df_train[x_cols], df_test[x_cols]
y_train, y_test = df_train[target], df_test[target]
y_train_class, y_test_class = df_train[target_class], df_test[target_class]
```


Now we can standardize the data.



```python
def standardize(x, x_ref, labels):
    std = np.std(x_ref[labels])
    mean = np.mean(x_ref[labels])
    x_std = (x[labels] - mean)/std
    return x_std
```




```python
std_labels = ['loan_amnt','total_acc','revol_util','revol_bal','pub_rec','open_acc','mths_since_last_record',
             'mths_since_last_delinq','inq_last_6mths','delinq_2yrs','dti','annual_inc','fico_avg',
             'cr_line_hist','emp_length','issue_d']
x_train_unstand = x_train.copy()
x_train_std = x_train.copy()
x_train_std[std_labels] = standardize(x_train, x_train_unstand, std_labels)

x_test_unstand = x_test.copy()
x_test_std = x_test.copy()
x_test_std[std_labels] = standardize(x_test, x_train_unstand, std_labels)
```




```python
display_df(x_test_std, 10)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>home_ownership_NONE</th>
      <th>home_ownership_OWN</th>
      <th>addr_state_VT</th>
      <th>addr_state_NJ</th>
      <th>issue_d</th>
      <th>verification_status_Not Verified</th>
      <th>purpose_home_improvement</th>
      <th>addr_state_CT</th>
      <th>addr_state_AR</th>
      <th>addr_state_OK</th>
      <th>addr_state_NE</th>
      <th>addr_state_RI</th>
      <th>addr_state_MI</th>
      <th>addr_state_CA</th>
      <th>addr_state_MS</th>
      <th>pub_rec</th>
      <th>addr_state_KY</th>
      <th>annual_inc</th>
      <th>delinq_2yrs</th>
      <th>addr_state_IA</th>
      <th>open_acc</th>
      <th>addr_state_WI</th>
      <th>emp_length</th>
      <th>purpose_educational</th>
      <th>term</th>
      <th>loan_amnt</th>
      <th>dti</th>
      <th>addr_state_DE</th>
      <th>addr_state_HI</th>
      <th>addr_state_NM</th>
      <th>addr_state_KS</th>
      <th>addr_state_IL</th>
      <th>mths_since_last_record</th>
      <th>addr_state_PA</th>
      <th>revol_bal</th>
      <th>purpose_credit_card</th>
      <th>addr_state_WA</th>
      <th>addr_state_OH</th>
      <th>revol_util</th>
      <th>addr_state_TN</th>
      <th>secondary</th>
      <th>home_ownership_OTHER</th>
      <th>addr_state_SC</th>
      <th>inq_last_6mths</th>
      <th>addr_state_ID</th>
      <th>verification_status_Verified</th>
      <th>addr_state_WV</th>
      <th>purpose_moving</th>
      <th>cr_line_hist</th>
      <th>fico_avg</th>
      <th>addr_state_VA</th>
      <th>addr_state_GA</th>
      <th>addr_state_AL</th>
      <th>addr_state_MN</th>
      <th>home_ownership_ANY</th>
      <th>addr_state_AK</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_medical</th>
      <th>addr_state_FL</th>
      <th>addr_state_LA</th>
      <th>home_ownership_MORTGAGE</th>
      <th>addr_state_NC</th>
      <th>addr_state_ME</th>
      <th>purpose_car</th>
      <th>addr_state_IN</th>
      <th>addr_state_MA</th>
      <th>addr_state_MT</th>
      <th>addr_state_DC</th>
      <th>addr_state_OR</th>
      <th>addr_state_ND</th>
      <th>purpose_major_purchase</th>
      <th>addr_state_AZ</th>
      <th>total_acc</th>
      <th>addr_state_MD</th>
      <th>purpose_vacation</th>
      <th>addr_state_UT</th>
      <th>addr_state_WY</th>
      <th>mths_since_last_delinq</th>
      <th>addr_state_NV</th>
      <th>purpose_house</th>
      <th>addr_state_TX</th>
      <th>purpose_renewable_energy</th>
      <th>addr_state_SD</th>
      <th>verification_status_Source Verified</th>
      <th>addr_state_NH</th>
      <th>purpose_wedding</th>
      <th>addr_state_MO</th>
      <th>addr_state_CO</th>
      <th>purpose_small_business</th>
      <th>home_ownership_RENT</th>
      <th>addr_state_NY</th>
      <th>purpose_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1955928</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.218230</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.029497</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>-0.648512</td>
      <td>0</td>
      <td>-2.863524e-01</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.543836</td>
      <td>-0.473508</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>0.038388</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.450860</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.658735</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.418433</td>
      <td>-0.695487</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.369610</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.986977</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1033990</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.625655</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>0.079481</td>
      <td>0.774675</td>
      <td>0</td>
      <td>0.066341</td>
      <td>0</td>
      <td>-1.309037e-12</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.179385</td>
      <td>0.210531</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>0.715214</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.751638</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.446714</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.242055</td>
      <td>-0.695487</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.973425</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.046213</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>668234</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.611449</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.201714</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>-1.542078</td>
      <td>0</td>
      <td>1.131624e+00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.229404</td>
      <td>-1.024518</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>-0.406442</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.893972</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.658735</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.598929</td>
      <td>-0.848969</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.034891</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.913103</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>361666</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.117906</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.126398</td>
      <td>0.774675</td>
      <td>0</td>
      <td>-0.827225</td>
      <td>0</td>
      <td>-1.562531e+00</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.008479</td>
      <td>-0.481066</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>-0.155544</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.109328</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.658735</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.121804</td>
      <td>-0.235039</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.114413</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.082753</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>972280</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-2.095323</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.029497</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>-0.112372</td>
      <td>0</td>
      <td>5.644333e-01</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.737534</td>
      <td>-0.495427</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.441931</td>
      <td>0</td>
      <td>-0.078982</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.016695</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.446714</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.079571</td>
      <td>0.225409</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.616492</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.986977</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1677515</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.117574</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.144308</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>-0.291085</td>
      <td>0</td>
      <td>-1.137138e+00</td>
      <td>0</td>
      <td>0.0</td>
      <td>-1.040919</td>
      <td>0.007209</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>-0.387453</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.260661</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.658735</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.863706</td>
      <td>0.839340</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.030733</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.986977</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>324922</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.117906</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.341702</td>
      <td>0</td>
      <td>-0.109865</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>-0.112372</td>
      <td>0</td>
      <td>5.644333e-01</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.543836</td>
      <td>-0.502986</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-2.285250</td>
      <td>0</td>
      <td>-0.471420</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.195565</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.658735</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.765286</td>
      <td>-1.002452</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.387666</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.889613</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36925</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.760810</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.086903</td>
      <td>0.774675</td>
      <td>0</td>
      <td>-0.827225</td>
      <td>0</td>
      <td>1.131624e+00</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.113107</td>
      <td>-0.485601</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>0.116519</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.089869</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.658735</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.718463</td>
      <td>-0.695487</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.114413</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.067093</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1417463</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.869179</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.341702</td>
      <td>0</td>
      <td>-0.660959</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>-0.827225</td>
      <td>0</td>
      <td>-1.309037e-12</td>
      <td>0</td>
      <td>0.0</td>
      <td>-1.096150</td>
      <td>0.850730</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-2.222259</td>
      <td>0</td>
      <td>-0.611697</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.044284</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.446714</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007396</td>
      <td>-1.155934</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.867531</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.033163</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>839523</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.353386</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.224676</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>0.066341</td>
      <td>0</td>
      <td>5.644333e-01</td>
      <td>0</td>
      <td>0.0</td>
      <td>-1.096150</td>
      <td>-0.236929</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>-0.603989</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.146834</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.446714</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.177274</td>
      <td>0.685857</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.555026</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.996623</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>400276 rows × 92 columns</p>
</div>




```python
display_df(y_train)
```



    806155     4.2
    261548     0.8
              ... 
    1361578    1.6
    1007988    2.8
    Name: num_grade, Length: 1601104, dtype: float64


The regression to classification function:



```python
def myround(x, prec=2, base=0.2):
    return round(base * round(float(x)/base),prec)
    
def num_to_subgrade(num):
    if not pd.isnull(num):
        sub = round((myround(num) - math.floor(num))/0.2)+1
        letter = {
            0:'A',
            1:'B',
            2:'C',
            3:'D',
            4:'E',
            5:'F',
            6:'G',
        }.get(math.floor(num))
        if letter == None:
            if num<0:
                letter = 'A' #if negative num
                sub = 1
            elif num>6:
                letter = 'G'
                sub = 5
        return letter+str(sub)
    else:
        return np.nan

def subgrade_to_num(grade):
    if not pd.isnull(grade):
        letter = grade[0]
        num = int(grade[1])
        num_grade = {
            'A':0,
            'B':1,
            'C':2,
            'D':3,
            'E':4,
            'F':5,
            'G':6
        }.get(letter)  
        num_grade += (num-1)*0.2
        return num_grade
    else:
        return np.nan
```


We want to be able to check for accuracy within a margin of error, so we construct a function to do so:



```python
def acc_within_delta (y1, y2, delta = 0.5 ):
    deltas = pd.Series.abs(y1 - y2)
    in_range = (deltas <= delta)
    accuracy = np.sum(in_range.values)/len(y1)
    return accuracy
```


## Models



```python
x_train = x_train_std.copy()
x_test = x_test_std.copy()
```




```python
display_df(x_train)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>home_ownership_NONE</th>
      <th>home_ownership_OWN</th>
      <th>addr_state_VT</th>
      <th>addr_state_NJ</th>
      <th>issue_d</th>
      <th>verification_status_Not Verified</th>
      <th>purpose_home_improvement</th>
      <th>addr_state_CT</th>
      <th>addr_state_AR</th>
      <th>addr_state_OK</th>
      <th>addr_state_NE</th>
      <th>addr_state_RI</th>
      <th>addr_state_MI</th>
      <th>addr_state_CA</th>
      <th>addr_state_MS</th>
      <th>pub_rec</th>
      <th>addr_state_KY</th>
      <th>annual_inc</th>
      <th>delinq_2yrs</th>
      <th>addr_state_IA</th>
      <th>open_acc</th>
      <th>addr_state_WI</th>
      <th>emp_length</th>
      <th>purpose_educational</th>
      <th>term</th>
      <th>loan_amnt</th>
      <th>dti</th>
      <th>addr_state_DE</th>
      <th>addr_state_HI</th>
      <th>addr_state_NM</th>
      <th>addr_state_KS</th>
      <th>addr_state_IL</th>
      <th>mths_since_last_record</th>
      <th>addr_state_PA</th>
      <th>revol_bal</th>
      <th>purpose_credit_card</th>
      <th>addr_state_WA</th>
      <th>addr_state_OH</th>
      <th>revol_util</th>
      <th>addr_state_TN</th>
      <th>secondary</th>
      <th>home_ownership_OTHER</th>
      <th>addr_state_SC</th>
      <th>inq_last_6mths</th>
      <th>addr_state_ID</th>
      <th>verification_status_Verified</th>
      <th>addr_state_WV</th>
      <th>purpose_moving</th>
      <th>cr_line_hist</th>
      <th>fico_avg</th>
      <th>addr_state_VA</th>
      <th>addr_state_GA</th>
      <th>addr_state_AL</th>
      <th>addr_state_MN</th>
      <th>home_ownership_ANY</th>
      <th>addr_state_AK</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_medical</th>
      <th>addr_state_FL</th>
      <th>addr_state_LA</th>
      <th>home_ownership_MORTGAGE</th>
      <th>addr_state_NC</th>
      <th>addr_state_ME</th>
      <th>purpose_car</th>
      <th>addr_state_IN</th>
      <th>addr_state_MA</th>
      <th>addr_state_MT</th>
      <th>addr_state_DC</th>
      <th>addr_state_OR</th>
      <th>addr_state_ND</th>
      <th>purpose_major_purchase</th>
      <th>addr_state_AZ</th>
      <th>total_acc</th>
      <th>addr_state_MD</th>
      <th>purpose_vacation</th>
      <th>addr_state_UT</th>
      <th>addr_state_WY</th>
      <th>mths_since_last_delinq</th>
      <th>addr_state_NV</th>
      <th>purpose_house</th>
      <th>addr_state_TX</th>
      <th>purpose_renewable_energy</th>
      <th>addr_state_SD</th>
      <th>verification_status_Source Verified</th>
      <th>addr_state_NH</th>
      <th>purpose_wedding</th>
      <th>addr_state_MO</th>
      <th>addr_state_CO</th>
      <th>purpose_small_business</th>
      <th>home_ownership_RENT</th>
      <th>addr_state_NY</th>
      <th>purpose_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>806155</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1.254353</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.603553</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>-1.005938</td>
      <td>0</td>
      <td>-1.420733</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.985687</td>
      <td>-0.585373</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>-0.574287</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.743507</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.658735</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.604178</td>
      <td>-0.388521</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.369610</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.986977</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>261548</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-3.181442</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>0.073833</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>-0.827225</td>
      <td>0</td>
      <td>-0.853543</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.764761</td>
      <td>-0.718401</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>-0.566448</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.150848</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.763061</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.852611</td>
      <td>1.299788</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.722385</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.986977</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1361578</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.919507</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>1.118614</td>
      <td>-0.359691</td>
      <td>0</td>
      <td>0.245054</td>
      <td>0</td>
      <td>1.131624</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.775885</td>
      <td>-0.589152</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>1</td>
      <td>0.743653</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.849203</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.446714</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.925323</td>
      <td>0.839340</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.471346</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.020113</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1007988</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.671113</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.350171</td>
      <td>0</td>
      <td>-0.316525</td>
      <td>0.774675</td>
      <td>0</td>
      <td>-0.827225</td>
      <td>0</td>
      <td>1.131624</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.168262</td>
      <td>-0.326118</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.441931</td>
      <td>0</td>
      <td>-0.725409</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-2.077776</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.658735</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.508441</td>
      <td>1.913718</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.783852</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.048823</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1601104 rows × 92 columns</p>
</div>


### Baseline Model - Linear Regression (OLS)



```python
X_train = sm.add_constant(x_train)
X_test = sm.add_constant(x_test)
model_OLS = OLS(y_train, X_train.astype(float)).fit()
```




```python
OLS_train_pred = model_OLS.predict(X_train)
OLS_test_pred = model_OLS.predict(X_test)
```




```python
score_OLS_train = r2_score(y_train,OLS_train_pred)
score_OLS_test = r2_score(y_test,OLS_test_pred)
# mse_OLS_train = sm.tools.eval_measures.mse( y_train,OLS_train_pred)
# mse_OLS_test = sm.tools.eval_measures.mse( y_test, OLS_test_pred)
acc_OLS_train = accuracy_score(y_train_class,OLS_train_pred.apply(num_to_subgrade))
acc_OLS_test = accuracy_score(y_test_class, OLS_test_pred.apply(num_to_subgrade))
acc_OLS_grade_train = acc_within_delta(y_train,OLS_train_pred, 1)
acc_OLS_grade_test = acc_within_delta(y_test,OLS_test_pred, 1)

print('OLS regression score on the training set is %.6f'%score_OLS_train)
print('OLS regression score on the test set is %.6f'%score_OLS_test)
#print('OLS mean squared error on the training set is %.6f'%mse_OLS_train)
#print('OLS mean squared error on the test set is %.6f'%mse_OLS_test)
print('OLS Classification accuracy of exact sub-grade (A1-G5) on the training set is {0:.1%}'.format(acc_OLS_train))
print('OLS Classification accuracy of exact sub-grade on the test set is {0:.1%}'.format(acc_OLS_test))
print('OLS Classification accuracy within one grade (A-G) on the training set is {0:.1%}'.format(acc_OLS_grade_train))
print('OLS Classification accuracy within one grade on the test set is {0:.1%}'.format(acc_OLS_grade_test))

```


    OLS regression score on the training set is 0.472266
    OLS regression score on the test set is 0.466961
    OLS Classification accuracy of exact sub-grade (A1-G5) on the training set is 8.6%
    OLS Classification accuracy of exact sub-grade on the test set is 8.6%
    OLS Classification accuracy within one grade (A-G) on the training set is 74.9%
    OLS Classification accuracy within one grade on the test set is 75.0%




```python
model_OLS.summary()
```





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>num_grade</td>    <th>  R-squared:         </th>  <td>   0.472</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.472</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.628e+04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Dec 2018</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>04:07:25</td>     <th>  Log-Likelihood:    </th> <td>-2.1303e+06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>1601104</td>     <th>  AIC:               </th>  <td>4.261e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>1601015</td>     <th>  BIC:               </th>  <td>4.262e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    88</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                   <td></td>                      <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                               <td> 4.263e+10</td> <td> 1.81e+10</td> <td>    2.354</td> <td> 0.019</td> <td> 7.14e+09</td> <td> 7.81e+10</td>
</tr>
<tr>
  <th>loan_amnt</th>                           <td>    0.0650</td> <td>    0.001</td> <td>   71.924</td> <td> 0.000</td> <td>    0.063</td> <td>    0.067</td>
</tr>
<tr>
  <th>addr_state_ND</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>pub_rec</th>                             <td>    0.0085</td> <td>    0.001</td> <td>    7.079</td> <td> 0.000</td> <td>    0.006</td> <td>    0.011</td>
</tr>
<tr>
  <th>purpose_wedding</th>                     <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>purpose_house</th>                       <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>cr_line_hist</th>                        <td>   -0.0800</td> <td>    0.001</td> <td>  -98.695</td> <td> 0.000</td> <td>   -0.082</td> <td>   -0.078</td>
</tr>
<tr>
  <th>addr_state_MN</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_NE</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>mths_since_last_record</th>              <td>    0.0203</td> <td>    0.001</td> <td>   16.337</td> <td> 0.000</td> <td>    0.018</td> <td>    0.023</td>
</tr>
<tr>
  <th>home_ownership_MORTGAGE</th>             <td>-9.571e+08</td> <td> 4.07e+08</td> <td>   -2.354</td> <td> 0.019</td> <td>-1.75e+09</td> <td> -1.6e+08</td>
</tr>
<tr>
  <th>revol_util</th>                          <td>    0.1083</td> <td>    0.001</td> <td>  112.692</td> <td> 0.000</td> <td>    0.106</td> <td>    0.110</td>
</tr>
<tr>
  <th>addr_state_RI</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_WI</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_IA</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_OH</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>delinq_2yrs</th>                         <td>    0.0146</td> <td>    0.001</td> <td>   18.564</td> <td> 0.000</td> <td>    0.013</td> <td>    0.016</td>
</tr>
<tr>
  <th>addr_state_VT</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_VA</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_KS</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>purpose_educational</th>                 <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>addr_state_HI</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_AK</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>total_acc</th>                           <td>   -0.1119</td> <td>    0.001</td> <td> -100.100</td> <td> 0.000</td> <td>   -0.114</td> <td>   -0.110</td>
</tr>
<tr>
  <th>addr_state_MA</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_TX</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_AZ</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_PA</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>purpose_debt_consolidation</th>          <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>addr_state_NY</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>purpose_other</th>                       <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>purpose_renewable_energy</th>            <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>home_ownership_OTHER</th>                <td>-9.571e+08</td> <td> 4.07e+08</td> <td>   -2.354</td> <td> 0.019</td> <td>-1.75e+09</td> <td> -1.6e+08</td>
</tr>
<tr>
  <th>emp_length</th>                          <td>    0.0014</td> <td>    0.001</td> <td>    1.898</td> <td> 0.058</td> <td>-4.66e-05</td> <td>    0.003</td>
</tr>
<tr>
  <th>addr_state_MS</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_CO</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_UT</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_WV</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_MT</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_FL</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>open_acc</th>                            <td>    0.0528</td> <td>    0.001</td> <td>   48.399</td> <td> 0.000</td> <td>    0.051</td> <td>    0.055</td>
</tr>
<tr>
  <th>dti</th>                                 <td>    0.1164</td> <td>    0.001</td> <td>  148.827</td> <td> 0.000</td> <td>    0.115</td> <td>    0.118</td>
</tr>
<tr>
  <th>home_ownership_ANY</th>                  <td>-9.571e+08</td> <td> 4.07e+08</td> <td>   -2.354</td> <td> 0.019</td> <td>-1.75e+09</td> <td> -1.6e+08</td>
</tr>
<tr>
  <th>addr_state_DE</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_IN</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_OR</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_GA</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_ID</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_ME</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_NH</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_WA</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_CT</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_AR</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_MO</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>verification_status_Verified</th>        <td> -4.92e+10</td> <td> 2.09e+10</td> <td>   -2.354</td> <td> 0.019</td> <td>-9.02e+10</td> <td>-8.24e+09</td>
</tr>
<tr>
  <th>purpose_home_improvement</th>            <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>fico_avg</th>                            <td>   -0.4586</td> <td>    0.001</td> <td> -452.298</td> <td> 0.000</td> <td>   -0.461</td> <td>   -0.457</td>
</tr>
<tr>
  <th>verification_status_Source Verified</th> <td> -4.92e+10</td> <td> 2.09e+10</td> <td>   -2.354</td> <td> 0.019</td> <td>-9.02e+10</td> <td>-8.24e+09</td>
</tr>
<tr>
  <th>addr_state_TN</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>annual_inc</th>                          <td>   -0.0534</td> <td>    0.001</td> <td>  -68.489</td> <td> 0.000</td> <td>   -0.055</td> <td>   -0.052</td>
</tr>
<tr>
  <th>home_ownership_RENT</th>                 <td>-9.571e+08</td> <td> 4.07e+08</td> <td>   -2.354</td> <td> 0.019</td> <td>-1.75e+09</td> <td> -1.6e+08</td>
</tr>
<tr>
  <th>purpose_car</th>                         <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>purpose_major_purchase</th>              <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>inq_last_6mths</th>                      <td>    0.2210</td> <td>    0.001</td> <td>  291.899</td> <td> 0.000</td> <td>    0.220</td> <td>    0.222</td>
</tr>
<tr>
  <th>addr_state_CA</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_SC</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_NJ</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_DC</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_LA</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>purpose_small_business</th>              <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>addr_state_NV</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>issue_d</th>                             <td>   -0.0252</td> <td>    0.001</td> <td>  -32.307</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.024</td>
</tr>
<tr>
  <th>addr_state_NM</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_NC</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_OK</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_MD</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>verification_status_Not Verified</th>    <td> -4.92e+10</td> <td> 2.09e+10</td> <td>   -2.354</td> <td> 0.019</td> <td>-9.02e+10</td> <td>-8.24e+09</td>
</tr>
<tr>
  <th>addr_state_KY</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>secondary</th>                           <td>    0.0097</td> <td>    0.004</td> <td>    2.352</td> <td> 0.019</td> <td>    0.002</td> <td>    0.018</td>
</tr>
<tr>
  <th>addr_state_SD</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>home_ownership_NONE</th>                 <td>-9.571e+08</td> <td> 4.07e+08</td> <td>   -2.354</td> <td> 0.019</td> <td>-1.75e+09</td> <td> -1.6e+08</td>
</tr>
<tr>
  <th>purpose_credit_card</th>                 <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>revol_bal</th>                           <td>   -0.0618</td> <td>    0.001</td> <td>  -73.575</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.060</td>
</tr>
<tr>
  <th>addr_state_MI</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>addr_state_AL</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>purpose_moving</th>                      <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>addr_state_WY</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>purpose_medical</th>                     <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>addr_state_IL</th>                       <td> 6.543e+08</td> <td> 2.78e+08</td> <td>    2.354</td> <td> 0.019</td> <td>  1.1e+08</td> <td>  1.2e+09</td>
</tr>
<tr>
  <th>mths_since_last_delinq</th>              <td>    0.0266</td> <td>    0.001</td> <td>   30.939</td> <td> 0.000</td> <td>    0.025</td> <td>    0.028</td>
</tr>
<tr>
  <th>term</th>                                <td>    1.0837</td> <td>    0.002</td> <td>  616.072</td> <td> 0.000</td> <td>    1.080</td> <td>    1.087</td>
</tr>
<tr>
  <th>purpose_vacation</th>                    <td>  6.87e+09</td> <td> 2.92e+09</td> <td>    2.354</td> <td> 0.019</td> <td> 1.15e+09</td> <td> 1.26e+10</td>
</tr>
<tr>
  <th>home_ownership_OWN</th>                  <td>-9.571e+08</td> <td> 4.07e+08</td> <td>   -2.354</td> <td> 0.019</td> <td>-1.75e+09</td> <td> -1.6e+08</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>213868.777</td> <th>  Durbin-Watson:     </th>  <td>   2.001</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>702167.449</td>
</tr>
<tr>
  <th>Skew:</th>            <td> 0.684</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>        <td> 5.942</td>   <th>  Cond. No.          </th>  <td>7.61e+15</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 6.78e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.





```python
mpl.rcParams['agg.path.chunksize'] = 10000
plt.scatter(OLS_test_pred)
```



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-260-6403ed2903ff> in <module>()
          1 mpl.rcParams['agg.path.chunksize'] = 10000
    ----> 2 plt.scatter(OLS_test_pred)
    

    TypeError: scatter() missing 1 required positional argument: 'y'


### Regularization Linear Models - Lasso and Ridge



```python
model_Lasso = LassoCV(cv=5, random_state=9001).fit(x_train,y_train)
```




```python
Lasso_train_pred = model_Lasso.predict(x_train)
Lasso_test_pred = model_Lasso.predict(x_test)
```




```python
score_Lasso_train = model_Lasso.score(x_train,y_train)
score_Lasso_test = model_Lasso.score(x_test,y_test)
acc_Lasso_train = accuracy_score(y_train_class,[num_to_subgrade(x) for x in Lasso_train_pred])
acc_Lasso_test = accuracy_score(y_test_class,[num_to_subgrade(x) for x in Lasso_test_pred])

acc_Lasso_grade_train = acc_within_delta(y_train,Lasso_train_pred, 1)
acc_Lasso_grade_test = acc_within_delta(y_test,Lasso_test_pred, 1)
```




```python
print('Lasso regression score on the training set is %.6f'%score_Lasso_train)
print('Lasso regression score on the test set is %.6f'%score_Lasso_test)
print('Lasso classification accuracy of exact sub-grade (A1-G5) on the training set with Lasso is %.6f'%acc_Lasso_train)
print('Lasso classification accuracy of exact sub-grade on the test set is %.6f'%acc_Lasso_test)
print('Classification accuracy within one grade (A-G) on the training set is {0:.1%}'.format(acc_Lasso_grade_train))
print('Classification accuracy within one grade on the test set is {0:.1%}'.format(acc_Lasso_grade_test))
```


    Lasso regression score on the training set is 0.471871
    Lasso regression score on the test set is 0.466649
    Lasso classification accuracy of exact sub-grade (A1-G5) on the training set with Lasso is 0.085869
    Lasso classification accuracy of exact sub-grade on the test set is 0.086086
    Classification accuracy within one grade (A-G) on the training set is 74.9%
    Classification accuracy within one grade on the test set is 74.9%




```python
model_Ridge = RidgeCV(cv=5).fit(x_train,y_train)
```




```python
Ridge_train_pred = model_Ridge.predict(x_train)
Ridge_test_pred = model_Ridge.predict(x_test)
```




```python
score_Ridge_train = model_Ridge.score(x_train,y_train)
score_Ridge_test = model_Ridge.score(x_test,y_test)
acc_Ridge_train = accuracy_score(y_train_class,[num_to_subgrade(x) for x in Ridge_train_pred])
acc_Ridge_test = accuracy_score(y_test_class,[num_to_subgrade(x) for x in Ridge_test_pred])

acc_Ridge_grade_train = acc_within_delta(y_train,Ridge_train_pred, 1)
acc_Ridge_grade_test = acc_within_delta(y_test,Ridge_test_pred, 1)
```




```python
print('Ridge regression score on the training set is %.6f'%score_Ridge_train)
print('Ridge regression score on the test set is %.6f'%score_Ridge_test)
print('Ridge classification accuracy on the training set with Ridge is %.6f'%acc_Ridge_train)
print('Ridge classification accuracy on the test set is %.6f'%acc_Ridge_test)
print('Classification accuracy within one grade (A-G) on the training set is {0:.1%}'.format(acc_Ridge_grade_train))
print('Classification accuracy within one grade on the test set is {0:.1%}'.format(acc_Ridge_grade_test))
```


    Ridge regression score on the training set is 0.472266
    Ridge regression score on the test set is 0.466965
    Ridge classification accuracy on the training set with Ridge is 0.086050
    Ridge classification accuracy on the test set is 0.086298
    Classification accuracy within one grade (A-G) on the training set is 74.9%
    Classification accuracy within one grade on the test set is 75.0%


### Decision Tree



```python
depths = {'max_depth' : [5,10,15,20]}
```




```python
model_DT_grid  = GridSearchCV(DecisionTreeClassifier(), depths, cv = 5)
model_DT_grid.fit(x_train, y_train_class)

model_DT = DecisionTreeClassifier(max_depth=model_DT_grid.best_params_['max_depth'])
model_DT.fit(x_train, y_train_class)
```





    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=15,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')





```python
print("Best DT had max_depth: %i"%model_DT_grid.best_params_['max_depth'])
acc_model_DT_train = model_DT.score(x_train, y_train_class)
print('Accuracy on training set for Decision Tree classifier is %f' %acc_model_DT_train)
acc_model_DT_test = model_DT.score(x_test, y_test_class)
print('Accuracy on test set for Decision Tree classifier is %f' %acc_model_DT_test)

model_DT_sig_feats = np.argsort(model_DT.feature_importances_)[::-1]
print('\nSignificant predictors are: ')

for i in range(10):
    print(x_test.columns[model_DT_sig_feats[i]])

print('\nTotal Number of Predictors is %d' %np.sum(model_DT.feature_importances_> 0))
```


    Best DT had max_depth: 15
    Accuracy on training set for Decision Tree classifier is 0.199871
    Accuracy on test set for Decision Tree classifier is 0.145280
    
    Significant predictors are: 
    fico_avg
    issue_d
    loan_amnt
    term
    revol_util
    dti
    annual_inc
    cr_line_hist
    revol_bal
    open_acc
    
    Total Number of Predictors is 98




```python
model_DTR_grid  = GridSearchCV(DecisionTreeRegressor(), depths, cv = 5)
model_DTR_grid.fit(x_train, y_train)

model_DTR = DecisionTreeRegressor(max_depth=model_DTR_grid.best_params_['max_depth'])
model_DTR.fit(x_train, y_train)
```





    DecisionTreeRegressor(criterion='mse', max_depth=15, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=None, splitter='best')





```python
print("Best DTR had max_depth: %i"%model_DTR_grid.best_params_['max_depth'])
DTR_train_pred = model_DTR.predict(x_train)
acc_model_DTR_train = accuracy_score(y_train_class,[num_to_subgrade(x) for x in DTR_train_pred])
print('Accuracy on training set for Decision Tree classifier is %f' %acc_model_DTR_train)
DTR_test_pred = model_DTR.predict(x_test)
acc_model_DTR_test = accuracy_score(y_test_class,[num_to_subgrade(x) for x in DTR_test_pred])
print('Accuracy on test set for Decision Tree classifier is %f' %acc_model_DTR_test)

acc_DTR_grade_train = acc_within_delta(y_train,DTR_train_pred, 1)
acc_DTR_grade_test = acc_within_delta(y_test,DTR_test_pred, 1)
print('Classification accuracy within one grade (A-G) on the training set is %f'%acc_DTR_grade_train)
print('Classification accuracy within one grade on the test set is %f'%acc_DTR_grade_test)


model_DTR_sig_feats = np.argsort(model_DTR.feature_importances_)[::-1]
print('\nSignificant predictors are: ')

for i in range(10):
    print(x_test.columns[model_DTR_sig_feats[i]])

print('\nTotal Number of Predictors is %d' %np.sum(model_DTR.feature_importances_> 0))
```


    Best DTR had max_depth: 15
    Accuracy on training set for Decision Tree classifier is 0.102280
    Accuracy on test set for Decision Tree classifier is 0.089858
    Classification accuracy within one grade (A-G) on the training set is 0.797906
    Classification accuracy within one grade on the test set is 0.766039
    
    Significant predictors are: 
    fico_avg
    term
    issue_d
    dti
    inq_last_6mths
    annual_inc
    purpose_credit_card
    loan_amnt
    verification_status_Verified
    revol_util
    
    Total Number of Predictors is 90


### Random Forest



```python
params = {'max_depth' : [10,12,15,17,20], 'n_estimators': [15,30,40,45,50]}
```




```python
model_RF_grid  = GridSearchCV(RandomForestClassifier(), params, cv = 5)
model_RF_grid.fit(x_train, y_train_class)
```




```python
model_RF = RandomForestClassifier(n_estimators=model_RF_grid.best_params_['n_estimators'],
                                  max_depth=model_RF_grid.best_params_['max_depth'])
model_RF.fit(x_train, y_train_class)
```




```python
print("Best RF had max_depth: %i"%model_RF_grid.best_params_['max_depth'])
acc_model_RF_train = model_RF.score(x_train, y_train_class)
print('Accuracy on training set for Decision Tree classifier is %f' %acc_model_RF_train)
acc_model_RF_test = model_RF.score(x_test, y_test_class)
print('Accuracy on test set for Decision Tree classifier is %f' %acc_model_RF_test)

model_RF_sig_feats = np.argsort(model_RF.feature_importances_)[::-1]
print('\nSignificant predictors are: ')

for i in range(10):
    print(x_test.columns[model_RF_sig_feats[i]])

print('\nTotal Number of Predictors is %d' %np.sum(model_RF.feature_importances_> 0))
```


### Logistic Regression



```python
model_LR = LogisticRegression().fit(x_train,y_train_class)
```




```python
LR_train_pred = model_LR.predict(x_train)
acc_LR_train = acc_within_delta(y_train, [subgrade_to_num(x) for x in LR_train_pred], 1)
LR_test_pred = model_LR.predict(x_test)
acc_LR_test = acc_within_delta(y_test, [subgrade_to_num(x) for x in LR_test_pred],1 )

print('Classification accuracy within one grade on the training set is %.5f'%acc_LR_train)
print('Classification accuracy within one grade on the test set is %.5f'%acc_LR_test)
```


    Classification accuracy within one grade on the training set is 0.75370
    Classification accuracy within one grade on the test set is 0.75386

