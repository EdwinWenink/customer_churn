from pathlib import Path

CAT_COLUMNS = [
    'gender',
    'education_level',
    'marital_status',
    'income_category',
    'card_category'
]

QUANT_COLUMNS = [
    'customer_age',
    'dependent_count',
    'months_on_book',
    'total_relationship_count',
    'months_inactive_12_mon',
    'contacts_count_12_mon',
    'credit_limit',
    'total_revolving_bal',
    'avg_open_to_buy',
    'total_amt_chng_q4_q1',
    'total_trans_amt',
    'total_trans_ct',
    'total_ct_chng_q4_q1',
    'avg_utilization_ratio'
]

RESPONSE = 'churn'

SELECT_FEATURES = ['customer_age', 'dependent_count', 'months_on_book',
                'total_relationship_count', 'months_inactive_12_mon',
                'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
                'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt',
                'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio',
                f'gender_{RESPONSE}', f'education_level_{RESPONSE}', f'marital_status_{RESPONSE}',
                f'income_category_{RESPONSE}', f'card_category_{RESPONSE}']

IMG_DIR = Path('./images/')
MODEL_DIR = Path('./models/')
DEFAULT_FIG_SIZE = (20, 10)
