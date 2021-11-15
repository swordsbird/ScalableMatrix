import os
import numpy as np
import random
import csv
from reader import Reader

class BankReader(Reader):
    @property
    def features(self):
        # Net Income Flag全1，直接剃掉了
        firstrow=r"Bankrupt?, ROA(C) before interest and depreciation before interest, ROA(A) before interest and % after tax, ROA(B) before interest and depreciation after tax, Operating Gross Margin, Realized Sales Gross Margin, Operating Profit Rate, Pre-tax net Interest Rate, After-tax net Interest Rate, Non-industry income and expenditure/revenue, Continuous interest rate (after tax), Operating Expense Rate, Research and development expense rate, Cash flow rate, Interest-bearing debt interest rate, Tax rate (A), Net Value Per Share (B), Net Value Per Share (A), Net Value Per Share (C), Persistent EPS in the Last Four Seasons, Cash Flow Per Share, Revenue Per Share (Yuan ¥), Operating Profit Per Share (Yuan ¥), Per Share Net profit before tax (Yuan ¥), Realized Sales Gross Profit Growth Rate, Operating Profit Growth Rate, After-tax Net Profit Growth Rate, Regular Net Profit Growth Rate, Continuous Net Profit Growth Rate, Total Asset Growth Rate, Net Value Growth Rate, Total Asset Return Growth Rate Ratio, Cash Reinvestment %, Current Ratio, Quick Ratio, Interest Expense Ratio, Total debt/Total net worth, Debt ratio %, Net worth/Assets, Long-term fund suitability ratio (A), Borrowing dependency, Contingent liabilities/Net worth, Operating profit/Paid-in capital, Net profit before tax/Paid-in capital, Inventory and accounts receivable/Net value, Total Asset Turnover, Accounts Receivable Turnover, Average Collection Days, Inventory Turnover Rate (times), Fixed Assets Turnover Frequency, Net Worth Turnover Rate (times), Revenue per person, Operating profit per person, Allocation rate per person, Working Capital to Total Assets, Quick Assets/Total Assets, Current Assets/Total Assets, Cash/Total Assets, Quick Assets/Current Liability, Cash/Current Liability, Current Liability to Assets, Operating Funds to Liability, Inventory/Working Capital, Inventory/Current Liability, Current Liabilities/Liability, Working Capital/Equity, Current Liabilities/Equity, Long-term Liability to Current Assets, Retained Earnings to Total Assets, Total income/Total expense, Total expense/Assets, Current Asset Turnover Rate, Quick Asset Turnover Rate, Working capitcal Turnover Rate, Cash Turnover Rate, Cash Flow to Sales, Fixed Assets to Assets, Current Liability to Liability, Current Liability to Equity, Equity to Long-term Liability, Cash Flow to Total Assets, Cash Flow to Liability, CFO to Assets, Cash Flow to Equity, Current Liability to Current Assets, Liability-Assets Flag, Net Income to Total Assets, Total assets to GNP price, No-credit Interval, Gross Profit to Sales, Net Income to Stockholder's Equity, Liability to Equity, Degree of Financial Leverage (DFL), Interest Coverage Ratio (Interest expense to EBIT), Equity to Liability"

        return firstrow.split(',')[1:]

    @property
    def categories(self):
        return ['No','Yes']

    def val2float(self,feature,val)->float:
        return float(val)

    def getData(self):
        data_file=os.path.join('data','bank.csv')
        dataset=[]
        resultset=[]
        with open(data_file,'r',encoding="utf-8") as f:
            reader=csv.DictReader(f)
            for line in reader:
                obj=[]
                for prop in self.features:
                    obj.append(float(self.val2float(prop,line[prop])))
                dataset.append(obj)
                resultset.append(int(line['Bankrupt?']))
        N=len(dataset)
        vec=[i for i in range(N)]
        random.shuffle(vec)  # randomly shuffles the records
        dataset=[dataset[vec[i]] for i in range(N)]
        resultset=[resultset[vec[i]] for i in range(N)]
        return np.array(dataset),np.array(resultset)
