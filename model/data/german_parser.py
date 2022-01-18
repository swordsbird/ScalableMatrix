import csv

features = ['Account Balance', 'Duration of Credit (month)', 
'Payment Status of Previous Credit', 'Purpose', 'Credit Amount',
'Value Savings/Stocks', 'Length of current employment', 'Installment per cent',
'Sex & Marital Status', 'Guarantors', 'Duration in Current address',
'Most valuable available asset', 'Age (years)', 'Concurrent Credits',
'Type of apartment', 'No of Credits at this Bank', 'Occupation',
'No of dependents', 'Telephone', 'Foreign Worker', 'Creditability']

reassign = {
    'Creditability' : ['No', 'Yes'], 
    'Payment Status of Previous Credit' : [
        "delay in paying off in the past",
        "critical account/other credits elsewhere",
        "no credits taken/all credits paid back duly",
        "existing credits paid back duly till now",
        "all credits at this bank paid back duly",
    ],
    'Purpose' : [
        "others",
        "car (new)",
        "car (used)",
        "furniture/equipment",
        "radio/television",
        "domestic appliances",
        "repairs",
        "education", 
        "vacation",
        "retraining",
        "business"
    ],
    'Value Savings/Stocks': [
        "unknown/no savings account",
        "... <  100 DM", 
        "100 <= ... <  500 DM",
        "500 <= ... < 1000 DM", 
        "... >= 1000 DM",
    ],
    'Length of current employment': [
        "unemployed", 
        "< 1 yr", 
        "1 <= ... < 4 yrs",
        "4 <= ... < 7 yrs", 
        ">= 7 yrs"
    ],
    #'Installment per cent': [">= 35",  "25 <= ... < 35", "20 <= ... < 25",  "< 20"],
    #'No of Credits at this Bank': ["1", "2-3", "4-5", ">= 6"],
    #'No of dependents': ["3 or more", "0 to 2"],
    'Sex & Marital Status': [
        "male : divorced/separated",
        "female : non-single or male : single",
        "male : married/widowed",
        "female : single"
    ],
    'Guarantors': [
        'none',
        'co-applicant',
        'guarantor'
    ],
    'Most valuable available asset': [
        "unknown / no property",
        "car or other",
        "building soc. savings agr./life insurance", 
        "real estate"
    ],
    'Other installment plans': ['bank', 'stores', 'none'],
    'Type of apartment': ["for free", "rent", "own"],
    'Occupation': [
        'unemployed/ unskilled - non-resident',
        'unskilled - resident',
        'skilled employee / official',
        'management/ self-employed/ highly qualified employee/ officer'
    ],
    'Account Balance': [
        "no checking account",
        "... < 0 DM",
        "0<= ... < 200 DM",
        "... >= 200 DM / salary for at least 1 year",
    ],
    'Telephone': ['No', 'Yes'],
    'Foreign Worker': ['Yes', 'No'],
}
