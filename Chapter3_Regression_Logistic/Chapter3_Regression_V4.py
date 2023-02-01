{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6a603fe9-5932-4e25-8896-d78cef0c47c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import statsmodels.formula.api as sm\n",
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "07bba596-6d9b-4090-8dc7-c9bf1474f3b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>EmpId</th>\n",
       "      <th>Monthly_Income</th>\n",
       "      <th>Monthly_Expenses</th>\n",
       "      <th>Time_Spent_Reading_Books</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>6092</td>\n",
       "      <td>3836.80</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2109</td>\n",
       "      <td>1422.85</td>\n",
       "      <td>14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>7177</td>\n",
       "      <td>4717.05</td>\n",
       "      <td>39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>6665</td>\n",
       "      <td>4330.25</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>8356</td>\n",
       "      <td>5552.40</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   EmpId  Monthly_Income  Monthly_Expenses  Time_Spent_Reading_Books\n",
       "0      1            6092           3836.80                        29\n",
       "1      2            2109           1422.85                        14\n",
       "2      3            7177           4717.05                        39\n",
       "3      4            6665           4330.25                         1\n",
       "4      5            8356           5552.40                        44"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "emp_profile=pd.read_csv(r\"C:\\Users\\SREEHARI\\Desktop\\internship\\my training\\Chapter3_Regression_Logistic\\Datasets\\employee_profile.csv\")\n",
    "\n",
    "#First few rows\n",
    "emp_profile.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f24e0054-3d66-43a5-9c45-ec54d39e7e97",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['EmpId', 'Monthly_Income', 'Monthly_Expenses',\n",
      "       'Time_Spent_Reading_Books'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "#Column names \n",
    "print(emp_profile.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6342ce7b-faf0-4aed-9a6a-35f9d9e90bba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Drawing the Scatter Plotq\n",
    "plt.scatter(emp_profile[\"Monthly_Income\"], emp_profile[\"Monthly_Expenses\"])\n",
    "plt.title('Income vs Expenses Plot')\n",
    "plt.xlabel('Monthly Income')\n",
    "plt.ylabel('Monthly Expenses')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5868fa7d-3b52-4091-9dba-659460c7bb79",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(emp_profile[\"Monthly_Income\"], emp_profile[\"Time_Spent_Reading_Books\"])\n",
    "plt.title('Income vs Time_Spent_Reading_Books Plot')\n",
    "plt.xlabel('Monthly Income')\n",
    "plt.ylabel('Time_Spent_Reading_Books')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5d149e72-3303-466d-bd31-875ef3a1c3fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Passengers_count', 'marketing_cost', 'percent_delayed_flights',\n",
      "       'number_of_trips', 'customer_ratings', 'poor_weather_index',\n",
      "       'percent_male_customers', 'Holiday_week', 'percent_female_customers'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "#Regression Model Building\n",
    "air_pass=pd.read_csv(r\"C:\\Users\\SREEHARI\\Desktop\\internship\\my training\\Chapter3_Regression_Logistic\\Datasets\\Air_Passengers.csv\")\n",
    "\n",
    "print(air_pass.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "170e1c7a-a7d6-4d0e-9dc1-2242fab44e2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Passengers_count</td> <th>  R-squared:         </th> <td>   0.761</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.760</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   830.0</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>4.87e-83</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:25:38</td>     <th>  Log-Likelihood:    </th> <td> -2453.4</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   263</td>      <th>  AIC:               </th> <td>   4911.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   261</td>      <th>  BIC:               </th> <td>   4918.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>      <td> 5186.6868</td> <td>  839.019</td> <td>    6.182</td> <td> 0.000</td> <td> 3534.579</td> <td> 6838.795</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>marketing_cost</th> <td>    6.3901</td> <td>    0.222</td> <td>   28.810</td> <td> 0.000</td> <td>    5.953</td> <td>    6.827</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td> 8.874</td> <th>  Durbin-Watson:     </th> <td>   1.829</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.012</td> <th>  Jarque-Bera (JB):  </th> <td>   9.679</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.342</td> <th>  Prob(JB):          </th> <td> 0.00791</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 3.644</td> <th>  Cond. No.          </th> <td>1.88e+04</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.88e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Passengers_count   R-squared:                       0.761\n",
       "Model:                            OLS   Adj. R-squared:                  0.760\n",
       "Method:                 Least Squares   F-statistic:                     830.0\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):           4.87e-83\n",
       "Time:                        18:25:38   Log-Likelihood:                -2453.4\n",
       "No. Observations:                 263   AIC:                             4911.\n",
       "Df Residuals:                     261   BIC:                             4918.\n",
       "Df Model:                           1                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==================================================================================\n",
       "                     coef    std err          t      P>|t|      [0.025      0.975]\n",
       "----------------------------------------------------------------------------------\n",
       "Intercept       5186.6868    839.019      6.182      0.000    3534.579    6838.795\n",
       "marketing_cost     6.3901      0.222     28.810      0.000       5.953       6.827\n",
       "==============================================================================\n",
       "Omnibus:                        8.874   Durbin-Watson:                   1.829\n",
       "Prob(Omnibus):                  0.012   Jarque-Bera (JB):                9.679\n",
       "Skew:                           0.342   Prob(JB):                      0.00791\n",
       "Kurtosis:                       3.644   Cond. No.                     1.88e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 1.88e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model1 = sm.ols(formula='Passengers_count ~ marketing_cost', data=air_pass)\n",
    "fitted1 = model1.fit()\n",
    "fitted1.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "226bdb29-2888-49d2-be08-386aa84b666e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    33942.146091\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "#prediction from the model\n",
    "new_data=pd.DataFrame({\"marketing_cost\":[4500]})\n",
    "print(fitted1.predict(new_data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "becffdbb-c882-47e9-8f65-11f12b630a7d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    33942.146091\n",
      "1    28191.054238\n",
      "2    24356.993003\n",
      "3    37137.197120\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "new_data1=pd.DataFrame({\"marketing_cost\":[4500,3600, 3000,5000]})\n",
    "print(fitted1.predict(new_data1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8eaefe3f-cd73-4878-929b-d53a0e5a9753",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>marketing_cost</th>\n",
       "      <th>Passengers_count</th>\n",
       "      <th>passengers_count_pred</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3588.1</td>\n",
       "      <td>23291</td>\n",
       "      <td>28115.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3186.3</td>\n",
       "      <td>25523</td>\n",
       "      <td>25547.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3342.0</td>\n",
       "      <td>25620</td>\n",
       "      <td>26542.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2512.5</td>\n",
       "      <td>19625</td>\n",
       "      <td>21242.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3012.1</td>\n",
       "      <td>27231</td>\n",
       "      <td>24434.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>258</th>\n",
       "      <td>2929.8</td>\n",
       "      <td>24238</td>\n",
       "      <td>23908.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>259</th>\n",
       "      <td>4024.0</td>\n",
       "      <td>29600</td>\n",
       "      <td>30900.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>260</th>\n",
       "      <td>3003.8</td>\n",
       "      <td>28578</td>\n",
       "      <td>24381.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>261</th>\n",
       "      <td>3327.6</td>\n",
       "      <td>27426</td>\n",
       "      <td>26450.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>262</th>\n",
       "      <td>2052.1</td>\n",
       "      <td>17591</td>\n",
       "      <td>18300.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>263 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     marketing_cost  Passengers_count  passengers_count_pred\n",
       "0            3588.1             23291                28115.0\n",
       "1            3186.3             25523                25547.0\n",
       "2            3342.0             25620                26542.0\n",
       "3            2512.5             19625                21242.0\n",
       "4            3012.1             27231                24434.0\n",
       "..              ...               ...                    ...\n",
       "258          2929.8             24238                23908.0\n",
       "259          4024.0             29600                30900.0\n",
       "260          3003.8             28578                24381.0\n",
       "261          3327.6             27426                26450.0\n",
       "262          2052.1             17591                18300.0\n",
       "\n",
       "[263 rows x 3 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Predictions for the data\n",
    "air_pass[\"passengers_count_pred\"]=round(fitted1.predict(air_pass))\n",
    "keep_cols=[\"marketing_cost\", \"Passengers_count\", \"passengers_count_pred\"]\n",
    "air_pass[keep_cols]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "690079b4-309e-4d02-b59e-aeccd638c8b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Passengers_count</td> <th>  R-squared:         </th> <td>   0.102</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.099</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   29.72</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>1.16e-07</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:25:38</td>     <th>  Log-Likelihood:    </th> <td> -2627.4</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   263</td>      <th>  AIC:               </th> <td>   5259.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   261</td>      <th>  BIC:               </th> <td>   5266.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>        <td> 2.261e+04</td> <td> 1192.915</td> <td>   18.955</td> <td> 0.000</td> <td> 2.03e+04</td> <td>  2.5e+04</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>customer_ratings</th> <td>  894.4643</td> <td>  164.083</td> <td>    5.451</td> <td> 0.000</td> <td>  571.369</td> <td> 1217.560</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>28.234</td> <th>  Durbin-Watson:     </th> <td>   2.024</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  35.541</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.767</td> <th>  Prob(JB):          </th> <td>1.92e-08</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 3.944</td> <th>  Cond. No.          </th> <td>    27.0</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Passengers_count   R-squared:                       0.102\n",
       "Model:                            OLS   Adj. R-squared:                  0.099\n",
       "Method:                 Least Squares   F-statistic:                     29.72\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):           1.16e-07\n",
       "Time:                        18:25:38   Log-Likelihood:                -2627.4\n",
       "No. Observations:                 263   AIC:                             5259.\n",
       "Df Residuals:                     261   BIC:                             5266.\n",
       "Df Model:                           1                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "====================================================================================\n",
       "                       coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------------\n",
       "Intercept         2.261e+04   1192.915     18.955      0.000    2.03e+04     2.5e+04\n",
       "customer_ratings   894.4643    164.083      5.451      0.000     571.369    1217.560\n",
       "==============================================================================\n",
       "Omnibus:                       28.234   Durbin-Watson:                   2.024\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):               35.541\n",
       "Skew:                           0.767   Prob(JB):                     1.92e-08\n",
       "Kurtosis:                       3.944   Cond. No.                         27.0\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "\"\"\""
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#R-Squared Value\n",
    "model2 = sm.ols(formula='Passengers_count ~ customer_ratings', data=air_pass)\n",
    "fitted2 = model2.fit()\n",
    "fitted2.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "84c8363c-20e7-489f-b97d-bd2acfc453aa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Passengers_count</td> <th>  R-squared:         </th> <td>   0.911</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.908</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   325.3</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>8.93e-129</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:25:38</td>     <th>  Log-Likelihood:    </th> <td> -2323.3</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   263</td>      <th>  AIC:               </th> <td>   4665.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   254</td>      <th>  BIC:               </th> <td>   4697.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "              <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>                <td> 4173.3041</td> <td> 3.71e+04</td> <td>    0.113</td> <td> 0.910</td> <td>-6.88e+04</td> <td> 7.71e+04</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>marketing_cost</th>           <td>    4.4279</td> <td>    0.168</td> <td>   26.287</td> <td> 0.000</td> <td>    4.096</td> <td>    4.760</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>percent_delayed_flights</th>  <td> 2.187e+04</td> <td> 4827.398</td> <td>    4.530</td> <td> 0.000</td> <td> 1.24e+04</td> <td> 3.14e+04</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>number_of_trips</th>          <td>    0.3004</td> <td>    0.270</td> <td>    1.114</td> <td> 0.266</td> <td>   -0.231</td> <td>    0.831</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>customer_ratings</th>         <td>  546.3104</td> <td>   53.897</td> <td>   10.136</td> <td> 0.000</td> <td>  440.168</td> <td>  652.453</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>poor_weather_index</th>       <td> -919.5035</td> <td> 4520.130</td> <td>   -0.203</td> <td> 0.839</td> <td>-9821.210</td> <td> 7982.203</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>percent_female_customers</th> <td>  -15.7188</td> <td>  371.808</td> <td>   -0.042</td> <td> 0.966</td> <td> -747.937</td> <td>  716.499</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Holiday_week</th>             <td> 6804.5389</td> <td>  598.471</td> <td>   11.370</td> <td> 0.000</td> <td> 5625.942</td> <td> 7983.136</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>percent_male_customers</th>   <td>   -7.3113</td> <td>  372.653</td> <td>   -0.020</td> <td> 0.984</td> <td> -741.195</td> <td>  726.572</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td> 0.087</td> <th>  Durbin-Watson:     </th> <td>   1.778</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.957</td> <th>  Jarque-Bera (JB):  </th> <td>   0.082</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.041</td> <th>  Prob(JB):          </th> <td>   0.960</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 2.969</td> <th>  Cond. No.          </th> <td>1.43e+06</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.43e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Passengers_count   R-squared:                       0.911\n",
       "Model:                            OLS   Adj. R-squared:                  0.908\n",
       "Method:                 Least Squares   F-statistic:                     325.3\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):          8.93e-129\n",
       "Time:                        18:25:38   Log-Likelihood:                -2323.3\n",
       "No. Observations:                 263   AIC:                             4665.\n",
       "Df Residuals:                     254   BIC:                             4697.\n",
       "Df Model:                           8                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "============================================================================================\n",
       "                               coef    std err          t      P>|t|      [0.025      0.975]\n",
       "--------------------------------------------------------------------------------------------\n",
       "Intercept                 4173.3041   3.71e+04      0.113      0.910   -6.88e+04    7.71e+04\n",
       "marketing_cost               4.4279      0.168     26.287      0.000       4.096       4.760\n",
       "percent_delayed_flights   2.187e+04   4827.398      4.530      0.000    1.24e+04    3.14e+04\n",
       "number_of_trips              0.3004      0.270      1.114      0.266      -0.231       0.831\n",
       "customer_ratings           546.3104     53.897     10.136      0.000     440.168     652.453\n",
       "poor_weather_index        -919.5035   4520.130     -0.203      0.839   -9821.210    7982.203\n",
       "percent_female_customers   -15.7188    371.808     -0.042      0.966    -747.937     716.499\n",
       "Holiday_week              6804.5389    598.471     11.370      0.000    5625.942    7983.136\n",
       "percent_male_customers      -7.3113    372.653     -0.020      0.984    -741.195     726.572\n",
       "==============================================================================\n",
       "Omnibus:                        0.087   Durbin-Watson:                   1.778\n",
       "Prob(Omnibus):                  0.957   Jarque-Bera (JB):                0.082\n",
       "Skew:                           0.041   Prob(JB):                        0.960\n",
       "Kurtosis:                       2.969   Cond. No.                     1.43e+06\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 1.43e+06. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Multiple regression\n",
    "import statsmodels.formula.api as sm\n",
    "model3 = sm.ols(formula='Passengers_count ~ marketing_cost+percent_delayed_flights+number_of_trips+customer_ratings+poor_weather_index+percent_female_customers+Holiday_week+percent_male_customers', data=air_pass)\n",
    "fitted3 = model3.fit()\n",
    "fitted3.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "074a2b3c-da87-4f7b-b945-76a348a9ed7b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['id', 'Monthly_Income_in_USD', 'Number_of_Credit_cards',\n",
      "       'Number_of_personal_loans', 'Monthly_Income_in_Euro',\n",
      "       'Monthly_Expenses'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "# Multicollinearity\n",
    "income_expenses=pd.read_csv(r\"C:\\Users\\SREEHARI\\Desktop\\internship\\my training\\Chapter3_Regression_Logistic\\Datasets\\customer_income_expenses.csv\")\n",
    "\n",
    "print(income_expenses.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f4bc8c8b-3452-43e2-af98-58f7a589cd09",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Monthly_Expenses</td> <th>  R-squared:         </th> <td>   0.966</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.964</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   483.1</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>1.31e-48</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:25:38</td>     <th>  Log-Likelihood:    </th> <td> -512.77</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>    72</td>      <th>  AIC:               </th> <td>   1036.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>    67</td>      <th>  BIC:               </th> <td>   1047.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "              <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>                <td>  -72.6691</td> <td>  143.534</td> <td>   -0.506</td> <td> 0.614</td> <td> -359.164</td> <td>  213.826</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Monthly_Income_in_USD</th>    <td>    7.5244</td> <td>  121.538</td> <td>    0.062</td> <td> 0.951</td> <td> -235.066</td> <td>  250.115</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Number_of_Credit_cards</th>   <td>   30.2664</td> <td>   53.290</td> <td>    0.568</td> <td> 0.572</td> <td>  -76.100</td> <td>  136.633</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Number_of_personal_loans</th> <td>  149.2454</td> <td>  104.408</td> <td>    1.429</td> <td> 0.158</td> <td>  -59.155</td> <td>  357.645</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Monthly_Income_in_Euro</th>   <td>   -7.6337</td> <td>  135.041</td> <td>   -0.057</td> <td> 0.955</td> <td> -277.178</td> <td>  261.910</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>29.413</td> <th>  Durbin-Watson:     </th> <td>   2.599</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>   5.056</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.104</td> <th>  Prob(JB):          </th> <td>  0.0798</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 1.719</td> <th>  Cond. No.          </th> <td>4.68e+04</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.68e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Monthly_Expenses   R-squared:                       0.966\n",
       "Model:                            OLS   Adj. R-squared:                  0.964\n",
       "Method:                 Least Squares   F-statistic:                     483.1\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):           1.31e-48\n",
       "Time:                        18:25:38   Log-Likelihood:                -512.77\n",
       "No. Observations:                  72   AIC:                             1036.\n",
       "Df Residuals:                      67   BIC:                             1047.\n",
       "Df Model:                           4                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "============================================================================================\n",
       "                               coef    std err          t      P>|t|      [0.025      0.975]\n",
       "--------------------------------------------------------------------------------------------\n",
       "Intercept                  -72.6691    143.534     -0.506      0.614    -359.164     213.826\n",
       "Monthly_Income_in_USD        7.5244    121.538      0.062      0.951    -235.066     250.115\n",
       "Number_of_Credit_cards      30.2664     53.290      0.568      0.572     -76.100     136.633\n",
       "Number_of_personal_loans   149.2454    104.408      1.429      0.158     -59.155     357.645\n",
       "Monthly_Income_in_Euro      -7.6337    135.041     -0.057      0.955    -277.178     261.910\n",
       "==============================================================================\n",
       "Omnibus:                       29.413   Durbin-Watson:                   2.599\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):                5.056\n",
       "Skew:                           0.104   Prob(JB):                       0.0798\n",
       "Kurtosis:                       1.719   Cond. No.                     4.68e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 4.68e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model4=sm.ols(formula='Monthly_Expenses ~ Monthly_Income_in_USD+Number_of_Credit_cards+Number_of_personal_loans+Monthly_Income_in_Euro', data=income_expenses)\n",
    "fitted4 = model4.fit()\n",
    "fitted4.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "e4bc65c6-d94f-4fbf-bba1-816e1087cac3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Monthly_Expenses</td> <th>  R-squared:         </th> <td>   0.966</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.965</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   653.7</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>4.71e-50</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:25:38</td>     <th>  Log-Likelihood:    </th> <td> -512.77</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>    72</td>      <th>  AIC:               </th> <td>   1034.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>    68</td>      <th>  BIC:               </th> <td>   1043.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "              <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>                <td>  -67.9274</td> <td>  120.499</td> <td>   -0.564</td> <td> 0.575</td> <td> -308.379</td> <td>  172.524</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Number_of_Credit_cards</th>   <td>   30.1840</td> <td>   52.882</td> <td>    0.571</td> <td> 0.570</td> <td>  -75.339</td> <td>  135.707</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Number_of_personal_loans</th> <td>  149.1943</td> <td>  103.638</td> <td>    1.440</td> <td> 0.155</td> <td>  -57.611</td> <td>  356.000</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Monthly_Income_in_Euro</th>   <td>    0.7267</td> <td>    0.017</td> <td>   43.434</td> <td> 0.000</td> <td>    0.693</td> <td>    0.760</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>29.870</td> <th>  Durbin-Watson:     </th> <td>   2.600</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>   5.071</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.099</td> <th>  Prob(JB):          </th> <td>  0.0792</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 1.715</td> <th>  Cond. No.          </th> <td>1.87e+04</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.87e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Monthly_Expenses   R-squared:                       0.966\n",
       "Model:                            OLS   Adj. R-squared:                  0.965\n",
       "Method:                 Least Squares   F-statistic:                     653.7\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):           4.71e-50\n",
       "Time:                        18:25:38   Log-Likelihood:                -512.77\n",
       "No. Observations:                  72   AIC:                             1034.\n",
       "Df Residuals:                      68   BIC:                             1043.\n",
       "Df Model:                           3                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "============================================================================================\n",
       "                               coef    std err          t      P>|t|      [0.025      0.975]\n",
       "--------------------------------------------------------------------------------------------\n",
       "Intercept                  -67.9274    120.499     -0.564      0.575    -308.379     172.524\n",
       "Number_of_Credit_cards      30.1840     52.882      0.571      0.570     -75.339     135.707\n",
       "Number_of_personal_loans   149.1943    103.638      1.440      0.155     -57.611     356.000\n",
       "Monthly_Income_in_Euro       0.7267      0.017     43.434      0.000       0.693       0.760\n",
       "==============================================================================\n",
       "Omnibus:                       29.870   Durbin-Watson:                   2.600\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):                5.071\n",
       "Skew:                           0.099   Prob(JB):                       0.0792\n",
       "Kurtosis:                       1.715   Cond. No.                     1.87e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 1.87e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Model after dropping Monthly_Income_in_USD\n",
    "model5=sm.ols(formula='Monthly_Expenses ~Number_of_Credit_cards+Number_of_personal_loans+Monthly_Income_in_Euro', data=income_expenses)\n",
    "fitted5 = model5.fit()\n",
    "fitted5.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f2964084-e1e7-4164-96ef-028c7b842a4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#VIF Function \n",
    "def vif_cal(x_vars):\n",
    "    xvar_names=x_vars.columns\n",
    "    for i in range(0,xvar_names.shape[0]):\n",
    "        y=x_vars[xvar_names[i]] \n",
    "        x=x_vars[xvar_names.drop(xvar_names[i])]\n",
    "        rsq=sm.ols(formula=\"y~x\", data=x_vars).fit().rsquared  \n",
    "        vif=round(1/(1-rsq),2)\n",
    "        print (xvar_names[i], \" VIF = \" , vif)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "30cd2358-9248-4a0f-896c-6b39233c7fd4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id  VIF =  1.07\n",
      "Monthly_Income_in_USD  VIF =  65007299.17\n",
      "Number_of_Credit_cards  VIF =  15.94\n",
      "Number_of_personal_loans  VIF =  16.15\n",
      "Monthly_Income_in_Euro  VIF =  65007347.03\n"
     ]
    }
   ],
   "source": [
    "#Calculating VIF values using that function\n",
    "vif_cal(x_vars=income_expenses.drop([\"Monthly_Expenses\"], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "01485173-cdca-432c-ab4e-0623e525258a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id  VIF =  1.06\n",
      "Monthly_Income_in_USD  VIF =  1.01\n",
      "Number_of_Credit_cards  VIF =  15.94\n",
      "Number_of_personal_loans  VIF =  16.14\n"
     ]
    }
   ],
   "source": [
    "#Calculating VIF values after dropping Monthly_Income_in_Euro\n",
    "vif_cal(x_vars=income_expenses.drop([\"Monthly_Expenses\",\"Monthly_Income_in_Euro\"], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "e7e2dd9c-faa5-4c3b-83e0-ce6381d0d45e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id  VIF =  1.01\n",
      "Monthly_Income_in_USD  VIF =  1.0\n",
      "Number_of_Credit_cards  VIF =  1.01\n"
     ]
    }
   ],
   "source": [
    "#Calculating VIF values after dropping Monthly_Income_in_Euro and Number_of_Credit_cards\n",
    "vif_cal(x_vars=income_expenses.drop([\"Monthly_Expenses\",\"Monthly_Income_in_Euro\",\"Number_of_personal_loans\"], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "f6918250-da96-44d4-a807-860fc46f187e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Monthly_Expenses</td> <th>  R-squared:         </th> <td>   0.965</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.964</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   964.5</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>3.71e-51</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:25:39</td>     <th>  Log-Likelihood:    </th> <td> -513.85</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>    72</td>      <th>  AIC:               </th> <td>   1034.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>    69</td>      <th>  BIC:               </th> <td>   1041.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>              <td>  -56.3653</td> <td>  121.148</td> <td>   -0.465</td> <td> 0.643</td> <td> -298.049</td> <td>  185.319</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Monthly_Income_in_USD</th>  <td>    0.6522</td> <td>    0.015</td> <td>   43.134</td> <td> 0.000</td> <td>    0.622</td> <td>    0.682</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Number_of_Credit_cards</th> <td>  103.7997</td> <td>   13.600</td> <td>    7.632</td> <td> 0.000</td> <td>   76.669</td> <td>  130.931</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>34.625</td> <th>  Durbin-Watson:     </th> <td>   2.544</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>   5.263</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.067</td> <th>  Prob(JB):          </th> <td>  0.0720</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 1.682</td> <th>  Cond. No.          </th> <td>2.06e+04</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.06e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Monthly_Expenses   R-squared:                       0.965\n",
       "Model:                            OLS   Adj. R-squared:                  0.964\n",
       "Method:                 Least Squares   F-statistic:                     964.5\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):           3.71e-51\n",
       "Time:                        18:25:39   Log-Likelihood:                -513.85\n",
       "No. Observations:                  72   AIC:                             1034.\n",
       "Df Residuals:                      69   BIC:                             1041.\n",
       "Df Model:                           2                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==========================================================================================\n",
       "                             coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------------------\n",
       "Intercept                -56.3653    121.148     -0.465      0.643    -298.049     185.319\n",
       "Monthly_Income_in_USD      0.6522      0.015     43.134      0.000       0.622       0.682\n",
       "Number_of_Credit_cards   103.7997     13.600      7.632      0.000      76.669     130.931\n",
       "==============================================================================\n",
       "Omnibus:                       34.625   Durbin-Watson:                   2.544\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):                5.263\n",
       "Skew:                           0.067   Prob(JB):                       0.0720\n",
       "Kurtosis:                       1.682   Cond. No.                     2.06e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 2.06e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#The Final model after removing all the multicollinearity \n",
    "model6=sm.ols(formula='Monthly_Expenses ~ Monthly_Income_in_USD+Number_of_Credit_cards', data=income_expenses)\n",
    "fitted6 = model6.fit()\n",
    "fitted6.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3a47b323-4ed1-420f-95bc-407626d55ea6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "marketing_cost  VIF =  1.51\n",
      "percent_delayed_flights  VIF =  13.4\n",
      "number_of_trips  VIF =  1.03\n",
      "customer_ratings  VIF =  1.06\n",
      "poor_weather_index  VIF =  12.81\n",
      "percent_male_customers  VIF =  990.52\n",
      "Holiday_week  VIF =  1.21\n",
      "percent_female_customers  VIF =  989.91\n"
     ]
    }
   ],
   "source": [
    "#Calculating VIF values for airpassengers data \n",
    "vif_cal(x_vars=air_pass.drop([\"Passengers_count\",\"passengers_count_pred\"], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "35d99ab1-f93d-453c-90e9-a53dba20d4ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "marketing_cost  VIF =  1.51\n",
      "percent_delayed_flights  VIF =  13.34\n",
      "number_of_trips  VIF =  1.03\n",
      "customer_ratings  VIF =  1.06\n",
      "poor_weather_index  VIF =  12.78\n",
      "Holiday_week  VIF =  1.2\n",
      "percent_female_customers  VIF =  1.03\n"
     ]
    }
   ],
   "source": [
    "#Dropped percent_male_customers due to high VIF\n",
    "vif_cal(x_vars=air_pass.drop([\"Passengers_count\",\"passengers_count_pred\", \"percent_male_customers\"], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "e3e95ba7-6af5-43df-b0c4-31da8d3bf9d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "marketing_cost  VIF =  1.45\n",
      "number_of_trips  VIF =  1.02\n",
      "customer_ratings  VIF =  1.06\n",
      "poor_weather_index  VIF =  1.25\n",
      "Holiday_week  VIF =  1.2\n",
      "percent_female_customers  VIF =  1.01\n"
     ]
    }
   ],
   "source": [
    "#Dropped percent_male_customers and percent_delayed_flights due to high VIF\n",
    "vif_cal(x_vars=air_pass.drop([\"Passengers_count\",\"passengers_count_pred\",\"percent_male_customers\", \"percent_delayed_flights\"], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "5e83ae59-8832-41cf-a276-8b3fc0a2986a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Passengers_count</td> <th>  R-squared:         </th> <td>   0.904</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.902</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   401.2</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>4.35e-127</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:26:56</td>     <th>  Log-Likelihood:    </th> <td> -2333.5</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   263</td>      <th>  AIC:               </th> <td>   4681.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   256</td>      <th>  BIC:               </th> <td>   4706.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "              <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>                <td> 3659.6674</td> <td>  984.371</td> <td>    3.718</td> <td> 0.000</td> <td> 1721.172</td> <td> 5598.163</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>marketing_cost</th>           <td>    4.5785</td> <td>    0.171</td> <td>   26.797</td> <td> 0.000</td> <td>    4.242</td> <td>    4.915</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>number_of_trips</th>          <td>    0.4177</td> <td>    0.278</td> <td>    1.503</td> <td> 0.134</td> <td>   -0.130</td> <td>    0.965</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>customer_ratings</th>         <td>  547.0027</td> <td>   55.782</td> <td>    9.806</td> <td> 0.000</td> <td>  437.152</td> <td>  656.853</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>poor_weather_index</th>       <td> 1.855e+04</td> <td> 1461.863</td> <td>   12.691</td> <td> 0.000</td> <td> 1.57e+04</td> <td> 2.14e+04</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>percent_female_customers</th> <td>  -15.5571</td> <td>   12.302</td> <td>   -1.265</td> <td> 0.207</td> <td>  -39.783</td> <td>    8.668</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Holiday_week</th>             <td> 6802.3234</td> <td>  619.101</td> <td>   10.987</td> <td> 0.000</td> <td> 5583.144</td> <td> 8021.503</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td> 1.354</td> <th>  Durbin-Watson:     </th> <td>   1.869</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.508</td> <th>  Jarque-Bera (JB):  </th> <td>   1.134</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.154</td> <th>  Prob(JB):          </th> <td>   0.567</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 3.094</td> <th>  Cond. No.          </th> <td>5.53e+04</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 5.53e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Passengers_count   R-squared:                       0.904\n",
       "Model:                            OLS   Adj. R-squared:                  0.902\n",
       "Method:                 Least Squares   F-statistic:                     401.2\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):          4.35e-127\n",
       "Time:                        18:26:56   Log-Likelihood:                -2333.5\n",
       "No. Observations:                 263   AIC:                             4681.\n",
       "Df Residuals:                     256   BIC:                             4706.\n",
       "Df Model:                           6                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "============================================================================================\n",
       "                               coef    std err          t      P>|t|      [0.025      0.975]\n",
       "--------------------------------------------------------------------------------------------\n",
       "Intercept                 3659.6674    984.371      3.718      0.000    1721.172    5598.163\n",
       "marketing_cost               4.5785      0.171     26.797      0.000       4.242       4.915\n",
       "number_of_trips              0.4177      0.278      1.503      0.134      -0.130       0.965\n",
       "customer_ratings           547.0027     55.782      9.806      0.000     437.152     656.853\n",
       "poor_weather_index        1.855e+04   1461.863     12.691      0.000    1.57e+04    2.14e+04\n",
       "percent_female_customers   -15.5571     12.302     -1.265      0.207     -39.783       8.668\n",
       "Holiday_week              6802.3234    619.101     10.987      0.000    5583.144    8021.503\n",
       "==============================================================================\n",
       "Omnibus:                        1.354   Durbin-Watson:                   1.869\n",
       "Prob(Omnibus):                  0.508   Jarque-Bera (JB):                1.134\n",
       "Skew:                           0.154   Prob(JB):                        0.567\n",
       "Kurtosis:                       3.094   Cond. No.                     5.53e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 5.53e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Model after exclusing the high VIF variables\n",
    "model7 = sm.ols(formula='Passengers_count ~ marketing_cost+number_of_trips+customer_ratings+poor_weather_index+percent_female_customers+Holiday_week', data=air_pass)\n",
    "fitted7 = model7.fit()\n",
    "fitted7.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "ea63fa22-3df5-4b74-b678-e20f12eda84d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Passengers_count</td> <th>  R-squared:         </th> <td>   0.903</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.901</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   597.2</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>4.33e-129</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:27:30</td>     <th>  Log-Likelihood:    </th> <td> -2335.4</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   263</td>      <th>  AIC:               </th> <td>   4681.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   258</td>      <th>  BIC:               </th> <td>   4699.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>          <td> 3366.1891</td> <td>  664.355</td> <td>    5.067</td> <td> 0.000</td> <td> 2057.941</td> <td> 4674.438</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>marketing_cost</th>     <td>    4.6010</td> <td>    0.170</td> <td>   27.002</td> <td> 0.000</td> <td>    4.265</td> <td>    4.936</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>customer_ratings</th>   <td>  539.0514</td> <td>   55.771</td> <td>    9.665</td> <td> 0.000</td> <td>  429.228</td> <td>  648.875</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>poor_weather_index</th> <td> 1.852e+04</td> <td> 1465.404</td> <td>   12.638</td> <td> 0.000</td> <td> 1.56e+04</td> <td> 2.14e+04</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Holiday_week</th>       <td> 6790.5039</td> <td>  618.849</td> <td>   10.973</td> <td> 0.000</td> <td> 5571.866</td> <td> 8009.142</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td> 1.156</td> <th>  Durbin-Watson:     </th> <td>   1.894</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.561</td> <th>  Jarque-Bera (JB):  </th> <td>   0.970</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.145</td> <th>  Prob(JB):          </th> <td>   0.616</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 3.070</td> <th>  Cond. No.          </th> <td>5.16e+04</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 5.16e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Passengers_count   R-squared:                       0.903\n",
       "Model:                            OLS   Adj. R-squared:                  0.901\n",
       "Method:                 Least Squares   F-statistic:                     597.2\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):          4.33e-129\n",
       "Time:                        18:27:30   Log-Likelihood:                -2335.4\n",
       "No. Observations:                 263   AIC:                             4681.\n",
       "Df Residuals:                     258   BIC:                             4699.\n",
       "Df Model:                           4                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "======================================================================================\n",
       "                         coef    std err          t      P>|t|      [0.025      0.975]\n",
       "--------------------------------------------------------------------------------------\n",
       "Intercept           3366.1891    664.355      5.067      0.000    2057.941    4674.438\n",
       "marketing_cost         4.6010      0.170     27.002      0.000       4.265       4.936\n",
       "customer_ratings     539.0514     55.771      9.665      0.000     429.228     648.875\n",
       "poor_weather_index  1.852e+04   1465.404     12.638      0.000    1.56e+04    2.14e+04\n",
       "Holiday_week        6790.5039    618.849     10.973      0.000    5571.866    8009.142\n",
       "==============================================================================\n",
       "Omnibus:                        1.156   Durbin-Watson:                   1.894\n",
       "Prob(Omnibus):                  0.561   Jarque-Bera (JB):                0.970\n",
       "Skew:                           0.145   Prob(JB):                        0.616\n",
       "Kurtosis:                       3.070   Cond. No.                     5.16e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 5.16e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Individual impact of the variables \n",
    "##Drop two variables non-impacful number_of_trips and percent_female_customers\n",
    "model8 = sm.ols(formula='Passengers_count ~ marketing_cost+customer_ratings+poor_weather_index+Holiday_week', data=air_pass)\n",
    "fitted8 = model8.fit()\n",
    "fitted8.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "b86b340c-1fa4-43c0-884e-86650633aab2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>    <td>Passengers_count</td> <th>  R-squared:         </th> <td>   0.627</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.623</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   145.2</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>3.44e-55</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:27:54</td>     <th>  Log-Likelihood:    </th> <td> -2511.8</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   263</td>      <th>  AIC:               </th> <td>   5032.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   259</td>      <th>  BIC:               </th> <td>   5046.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>          <td>  1.45e+04</td> <td> 1016.979</td> <td>   14.257</td> <td> 0.000</td> <td> 1.25e+04</td> <td> 1.65e+04</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>customer_ratings</th>   <td>  834.0677</td> <td>  106.768</td> <td>    7.812</td> <td> 0.000</td> <td>  623.823</td> <td> 1044.312</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>poor_weather_index</th> <td> 3.379e+04</td> <td> 2639.293</td> <td>   12.801</td> <td> 0.000</td> <td> 2.86e+04</td> <td>  3.9e+04</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Holiday_week</th>       <td> 1.223e+04</td> <td> 1142.371</td> <td>   10.705</td> <td> 0.000</td> <td> 9979.306</td> <td> 1.45e+04</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td> 5.814</td> <th>  Durbin-Watson:     </th> <td>   1.887</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.055</td> <th>  Jarque-Bera (JB):  </th> <td>   5.509</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.330</td> <th>  Prob(JB):          </th> <td>  0.0636</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 3.257</td> <th>  Cond. No.          </th> <td>    95.3</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:       Passengers_count   R-squared:                       0.627\n",
       "Model:                            OLS   Adj. R-squared:                  0.623\n",
       "Method:                 Least Squares   F-statistic:                     145.2\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):           3.44e-55\n",
       "Time:                        18:27:54   Log-Likelihood:                -2511.8\n",
       "No. Observations:                 263   AIC:                             5032.\n",
       "Df Residuals:                     259   BIC:                             5046.\n",
       "Df Model:                           3                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "======================================================================================\n",
       "                         coef    std err          t      P>|t|      [0.025      0.975]\n",
       "--------------------------------------------------------------------------------------\n",
       "Intercept            1.45e+04   1016.979     14.257      0.000    1.25e+04    1.65e+04\n",
       "customer_ratings     834.0677    106.768      7.812      0.000     623.823    1044.312\n",
       "poor_weather_index  3.379e+04   2639.293     12.801      0.000    2.86e+04     3.9e+04\n",
       "Holiday_week        1.223e+04   1142.371     10.705      0.000    9979.306    1.45e+04\n",
       "==============================================================================\n",
       "Omnibus:                        5.814   Durbin-Watson:                   1.887\n",
       "Prob(Omnibus):                  0.055   Jarque-Bera (JB):                5.509\n",
       "Skew:                           0.330   Prob(JB):                       0.0636\n",
       "Kurtosis:                       3.257   Cond. No.                         95.3\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "\"\"\""
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Drop an impactful variable\n",
    "model9 = sm.ols(formula='Passengers_count ~  customer_ratings+poor_weather_index+Holiday_week', data=air_pass)\n",
    "fitted9 = model9.fit()\n",
    "fitted9.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "7c2231ca-a006-4eba-a8a7-9b0b3ae7c971",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Income', 'Bought'], dtype='object')\n"
     ]
    }
   ],
   "source": [
    "##################################\n",
    "####Logistic Regression\n",
    "#################################\n",
    "\n",
    "#product_sales Model\n",
    "product_sales=pd.read_csv(r\"C:\\Users\\SREEHARI\\Desktop\\internship\\my training\\Chapter3_Regression_Logistic\\Datasets\\Product_sales.csv\")\n",
    "print(product_sales.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "db1b81e2-64d0-48b2-8e86-66e6e1880f81",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>         <td>Bought</td>      <th>  R-squared:         </th> <td>   0.843</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.842</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2489.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Tue, 31 Jan 2023</td> <th>  Prob (F-statistic):</th> <td>8.50e-189</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>18:28:52</td>     <th>  Log-Likelihood:    </th> <td>  96.245</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   467</td>      <th>  AIC:               </th> <td>  -188.5</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   465</td>      <th>  BIC:               </th> <td>  -180.2</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th> <td>   -0.1805</td> <td>    0.015</td> <td>  -11.712</td> <td> 0.000</td> <td>   -0.211</td> <td>   -0.150</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Income</th>    <td> 2.095e-05</td> <td>  4.2e-07</td> <td>   49.886</td> <td> 0.000</td> <td> 2.01e-05</td> <td> 2.18e-05</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>77.189</td> <th>  Durbin-Watson:     </th> <td>   1.886</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>1010.549</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.076</td> <th>  Prob(JB):          </th> <td>3.65e-220</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td>10.205</td> <th>  Cond. No.          </th> <td>6.20e+04</td> \n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 6.2e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:                 Bought   R-squared:                       0.843\n",
       "Model:                            OLS   Adj. R-squared:                  0.842\n",
       "Method:                 Least Squares   F-statistic:                     2489.\n",
       "Date:                Tue, 31 Jan 2023   Prob (F-statistic):          8.50e-189\n",
       "Time:                        18:28:52   Log-Likelihood:                 96.245\n",
       "No. Observations:                 467   AIC:                            -188.5\n",
       "Df Residuals:                     465   BIC:                            -180.2\n",
       "Df Model:                           1                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==============================================================================\n",
       "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "Intercept     -0.1805      0.015    -11.712      0.000      -0.211      -0.150\n",
       "Income      2.095e-05    4.2e-07     49.886      0.000    2.01e-05    2.18e-05\n",
       "==============================================================================\n",
       "Omnibus:                       77.189   Durbin-Watson:                   1.886\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1010.549\n",
       "Skew:                           0.076   Prob(JB):                    3.65e-220\n",
       "Kurtosis:                      10.205   Cond. No.                     6.20e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 6.2e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model10 = sm.ols(formula='Bought ~  Income', data=product_sales)\n",
    "fitted10 = model10.fit()\n",
    "fitted10.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "04a7cc5a-18ff-4e31-a876-13e982b259ea",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0   -0.096753\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "#prediction from the model\n",
    "new_data=pd.DataFrame({\"Income\":[4000]})\n",
    "print(fitted10.predict(new_data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "7ec4e91d-47e9-4b02-9926-0226fc8e1668",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    1.599893\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "new_data1=pd.DataFrame({\"Income\":[85000]})\n",
    "print(fitted10.predict(new_data1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e0052c41-c72b-4dda-b741-780622726338",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Income  Bought\n",
      "466  57387.4       1\n",
      "267   2468.1       0\n",
      "283  51759.2       1\n",
      "196  10226.8       0\n",
      "36    4643.9       0\n",
      "265  63438.4       1\n",
      "309  42236.6       1\n",
      "86    1679.7       0\n",
      "120  10598.5       0\n",
      "419  12853.8       0\n"
     ]
    }
   ],
   "source": [
    "#product_sales data sample\n",
    "print(product_sales.sample(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "e2d0bd98-218a-477a-99d1-27c4b4d64201",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAAsTAAALEwEAmpwYAAAaqklEQVR4nO3de7hcdX3v8feHnQCBIAGy5ZALJviEILVAdHOxUosgJkErnnMsBFEbHk85FPGRYrmkamuPp1qEY6kCxkgRvJCAGmOwaKBapJUC2SGBEEg0hEt2giQICQIRcvmeP9Zvy2Qye+9JmDWzJ7/P63nmybr8Zq3vmtmZz6zfWrOWIgIzM8vXHq0uwMzMWstBYGaWOQeBmVnmHARmZplzEJiZZc5BYGaWOQeBWZuT9Likd5W07M9K+nYZy7bBw0Fgr1mZH0TtQtINkl6R9IKk30paJOlPBkFdJ0nqGaBNZe3PSrpD0hG7sK7s/w7alYPArHG+GBHDgf2BrwJzJXW0uKZ69dY+BlgH3NDacqyZHATWUJKmS/pPSVdKek7SY5KmVsw/UNI3JK1N8+dVzPsLSSvTt9L5kkZVzAtJ50v6VfrG/TlJb5T0X5Kel3SLpD0r2r9X0hJJGyTdLemoPuqdKenKqmk/lHRRGr5U0pq0zhWSThnoNYiIbcBNwIHAwWk5e0j6tKQnJK2T9E1J+6d5O3xrr/x2LWmYpBvT6/WIpEtqfMs/RtKDkjZKulnS3pL2BX4MjErf9l+ofE37qP2lVPub+3i93idpWXpd75T0pjT9W8ChwK1pPZcM9DrZ4OEgsDIcD6wARgJfBP5FktK8bwH7AH8AvB74JwBJJwNfAM4ADgGeAOZULXcK8FbgBOASYBZwNjCW4oPrrLSstwDXA/8bOAj4GjBf0l41ar0JOLO3PkkHAO8G5kiaCFwAHBsR+wGTgccH2vi0F/AR4DHg6TR5enq8EzgMGA5cPdCykr8DxqXnnQp8qEabMyhen/HAUcD0iHgRmAqsjYjh6bF2gNqHU7ymi2vMOxyYDVwIdAK3UXzw7xkRHwaeBP40reeLdW6bDQIOAivDExHx9YjYCtxI8cF+sKRDKD6YzouI5yJic0T8PD3nbOD6iLg/Il4GZgBvkzSuYrmXR8TzEbEMeAi4PSJWRcRGim++k1K7vwC+FhH3RsTWiLgReJkiQKr9BxDAH6fxDwD/lT4wtwJ7AUdKGhoRj0fEo/1s919L2gC8CFwFfCa9Br3b96VU7wtp+6ZJGtLvK1k4A/h8es16gC/XaPPliFgbEc8CtwLH1LHcWrWvpAip6TXanAn8a0TcERGbgSuBYcAf7eS6bJBxEFgZft07kLoaoPhwGQs8GxHP1XjOKIq9gN7nvQD8Bhhd0ebpiuFNNcaHp+E3AJ9M3Rcb0gfc2LSO7URx1cU5pL0J4IPAd9K8lRTffj8LrJM0Z4CulSsjYgTFh2MXcEVFt9h225eGh5C6jgYwClhdMb66RptfVwy/xKuvRb2ujIgREfHfIuJ9fQRe9Xu0LdUyukZbayMOAmum1cCBkkbUmLeW4gMcgNS/fRCwZhfX8w/pg633sU9EzO6j/WzgA5LeQNGt9f3eGRFxU0ScmGoL4PKBVh6Fh4BfAO9Jk7fbPor+9C0UYfYiRXcZ8Puupc6Ktk9RHMTtNXagGirL2Ym2A6l+j5Rq6X2PfCnjNuUgsKaJiKcounCulXSApKGS3pFm3wScI+mY1Jf/eeDeiHh8F1b1deA8ScersK+k90jar4+6FgPrgeuABRGxAUDSREknp3p+R7HXsbXWMqqpOP3yRGBZmjQb+CtJ41M//OeBmyNiC/BLYO9U41Dg0xRdUr1uAWak12w0xXGLej0NHNR7YPo1ugV4j6RTUp2fpOhyu7tiXYc1YD3WZA4Ca7YPA5uB5RSnKV4IEBE/BT5D8W38KeCNwLRdWUFEdFMcJ7gaeI6i33v6AE+bDbyLIpB67QX8I/AMRdfL64G/6WcZl6QzZl4Ebge+QXGgGoqD198C7qI4iPw74OOp3o3A+RRBtIZiD6HyrKD/k8YfA/4N+B7FB/CAImJ52rZVqZus37OGBljWCooD1V+heE3+lOLg8CupyReAT6f1/PWurseaT74xjVl7kfSXwLSIaPkP1mz34D0Cs0FO0iGS3p5+izCRokvmB62uy3Yf9Zy6ZmattSdFF9N4YAPFWU7XtrIg2724a8jMLHPuGjIzy1zbdQ2NHDkyxo0b1+oyzMzayqJFi56JiM5a89ouCMaNG0d3d3eryzAzayuSnuhrnruGzMwy5yAwM8ucg8DMLHMOAjOzzDkIzMwyV9pZQ5KuB94LrIuIHW57ly5h+8/AaRTXT58eEfeXVU+95i1ewxULVrB2wyZGjRjGxZMn8v5Jzbvcej3rb3WNNnjMW7yGi7+7hM3bGrvcEcOGsnnrNl58pa6LrW5HtMf1qPcAEGzbxWL3GboHL23eRofE1ib/MPdDJxzK/33/HzZseWXuEdxAceu8vkwFJqTHuRQ3+26peYvXMGPuUtZs2EQAazZsYsbcpcxbvCuXxC9n/a2u0QaPeYvXcOHNjQ8BgA2bNu9SCEB7hADANnY9BABeSi98s0MA4Nv3PMmn5y1t2PJKC4KIuAt4tp8mpwPfTDfxuAcYkW5l2DJXLFjBps3b//Fv2ryVKxasGDTrb3WNNnj4Pc/b7Htr3ahu17TyGMFotr/lXg993PJO0rmSuiV1r1+/vrSC1m7YtFPTW7H+Vtdog4ff87w1ck+klUGgGtNqbllEzIqIrojo6uys+Qvphhg1YthOTW/F+ltdow0efs/z1qFaH6G7ppVB0MP2914dQ3FP1Ja5ePJEhg3t2G7asKEdXDx54qBZf6trtMHD73nezjp+Z25d3b9WXmtoPnCBpDkUNwzfmO5p2zK9Z9606oycetbf6hpt8Oh9z33W0K7xWUOvKu1+BJJmAycBIyluav13wFCAiJiZTh+9muLMopeAc9K9ZvvV1dUVvuicmdnOkbQoIrpqzSttjyAizhpgfgAfK2v9ZmZWH/+y2Mwscw4CM7PMOQjMzDLnIDAzy5yDwMwscw4CM7PMOQjMzDLnIDAzy5yDwMwscw4CM7PMOQjMzDLnIDAzy5yDwMwscw4CM7PMOQjMzDLnIDAzy5yDwMwscw4CM7PMOQjMzDLnIDAzy5yDwMwscw4CM7PMOQjMzDLnIDAzy5yDwMwscw4CM7PMOQjMzDLnIDAzy5yDwMwscw4CM7PMOQjMzDJXahBImiJphaSVki6rMX9/SbdKekDSMknnlFmPmZntqLQgkNQBXANMBY4EzpJ0ZFWzjwEPR8TRwEnA/5O0Z1k1mZnZjsrcIzgOWBkRqyLiFWAOcHpVmwD2kyRgOPAssKXEmszMrEqZQTAaWF0x3pOmVboaeBOwFlgKfCIitlUvSNK5krolda9fv76ses3MslRmEKjGtKganwwsAUYBxwBXS3rdDk+KmBURXRHR1dnZ2eg6zcyyVmYQ9ABjK8bHUHzzr3QOMDcKK4HHgCNKrMnMzKqUGQQLgQmSxqcDwNOA+VVtngROAZB0MDARWFViTWZmVmVIWQuOiC2SLgAWAB3A9RGxTNJ5af5M4HPADZKWUnQlXRoRz5RVk5mZ7ai0IACIiNuA26qmzawYXgu8u8wazMysf/5lsZlZ5hwEZmaZcxCYmWXOQWBmljkHgZlZ5hwEZmaZcxCYmWXOQWBmljkHgZlZ5hwEZmaZcxCYmWXOQWBmljkHgZlZ5hwEZmaZcxCYmWXOQWBmljkHgZlZ5hwEZmaZcxCYmWXOQWBmljkHgZlZ5hwEZmaZcxCYmWXOQWBmljkHgZlZ5hwEZmaZcxCYmWXOQWBmljkHgZlZ5hwEZmaZKzUIJE2RtELSSkmX9dHmJElLJC2T9PMy6zEzsx0NKWvBkjqAa4BTgR5goaT5EfFwRZsRwLXAlIh4UtLry6rHzMxqK3OP4DhgZUSsiohXgDnA6VVtPgjMjYgnASJiXYn1mJlZDWUGwWhgdcV4T5pW6XDgAEl3Slok6SO1FiTpXEndkrrXr19fUrlmZnkqMwhUY1pUjQ8B3gq8B5gMfEbS4Ts8KWJWRHRFRFdnZ2fjKzUzy1hpxwgo9gDGVoyPAdbWaPNMRLwIvCjpLuBo4Jcl1mVmZhXK3CNYCEyQNF7SnsA0YH5Vmx8CfyxpiKR9gOOBR0qsyczMqpS2RxARWyRdACwAOoDrI2KZpPPS/JkR8YiknwAPAtuA6yLiobJqMjOzHSmiutu+RiPp8oi4dKBpzdDV1RXd3d3NXq2ZWVuTtCgiumrNq7dr6NQa06bueklmZjZY9Ns1JOkvgfOBwyQ9WDFrP+AXZRZmZmbNMdAxgpuAHwNfACovEfHbiHi2tKrMzKxp+g2CiNgIbATOSpeMODg9Z7ik4b2/CDYzs/ZV11lD6eyfzwJPU5zdA8WPw44qpywzM2uWek8fvRCYGBG/KbEWMzNrgXrPGlpN0UVkZma7mYHOGrooDa4C7pT0r8DLvfMj4ksl1mZmZk0wUNfQfunfJ9Njz/QwM7PdxEBnDf19swoxM7PWqPesoVvZ8RLSG4Fu4GsR8btGF2ZmZs1R78HiVcALwNfT43mKU0kPT+NmZtam6j19dFJEvKNi/FZJd0XEOyQtK6MwMzNrjnr3CDolHdo7koZHptFXGl6VmZk1Tb17BJ8E/lPSoxS3oBwPnC9pX+DGsoozM7Py1RUEEXGbpAnAERRBsLziAPFVJdVmZmZNUO9ZQx+pmnSUJCLimyXUZGZmTVRv19CxFcN7A6cA9wMOAjOzNldv19DHK8cl7Q98q5SKzMysqeo9a6jaS8CERhZiZmatsSu/LO4A3gTcUlZRZmbWPPUeI7iyYngL8ERE9JRQj5mZNVldXUMR8XNgOcXVSA/APyIzM9tt1BUEks4A7gP+DDgDuFfSB8oszMzMmqPerqFPAcdGxDoASZ3AvwHfK6swMzNrjnrPGtqjNwSS3+zEc83MbBCrd4/gJ5IWALPT+JnAbeWUZGZmzVTvD8oulvQ/gBMprjU0KyJ+UGplZmbWFPXuERARc4G5kkZSdA2ZmdluoN9+fkknSLpT0lxJkyQ9BDwEPC1pSnNKNDOzMg20R3A18DfA/sDPgKkRcY+kIyiOF/yk5PrMzKxkA535MyQibo+I7wK/joh7ACJieT0LlzRF0gpJKyVd1k+7YyVt9W8TzMyab6Ag2FYxvKlqXtAPSR3ANcBU4EjgLElH9tHucmDBgNWamVnDDdQ1dLSk5ynOFBqWhknjew/w3OOAlRGxCkDSHOB04OGqdh8Hvs/29zwwM7Mm6TcIIqLjNSx7NLC6YrwHOL6ygaTRwH8HTqafIJB0LnAuwKGHHvoaSjIzs2pl/jpYNaZVdyddBVwaEVv7W1BEzIqIrojo6uzsbFR9ZmbGTvyOYBf0AGMrxscAa6vadAFzJAGMBE6TtCUi5pVYl5mZVSgzCBYCEySNB9YA04APVjaIiPG9w5JuAH7kEDAza67SgiAitki6gOJsoA7g+ohYJum8NH9mWes2M7P6lblHQETcRtXF6foKgIiYXmYtZmZWmy8lbWaWOQeBmVnmHARmZplzEJiZZc5BYGaWOQeBmVnmHARmZplzEJiZZc5BYGaWOQeBmVnmHARmZplzEJiZZc5BYGaWOQeBmVnmHARmZplzEJiZZc5BYGaWOQeBmVnmHARmZplzEJiZZc5BYGaWOQeBmVnmHARmZplzEJiZZc5BYGaWOQeBmVnmHARmZplzEJiZZc5BYGaWOQeBmVnmSg0CSVMkrZC0UtJlNeafLenB9Lhb0tFl1mNmZjsqLQgkdQDXAFOBI4GzJB1Z1ewx4E8i4ijgc8CssuoxM7PaytwjOA5YGRGrIuIVYA5wemWDiLg7Ip5Lo/cAY0qsx8zMaigzCEYDqyvGe9K0vnwU+HGtGZLOldQtqXv9+vUNLNHMzMoMAtWYFjUbSu+kCIJLa82PiFkR0RURXZ2dnQ0s0czMhpS47B5gbMX4GGBtdSNJRwHXAVMj4jcl1mNmZjWUuUewEJggabykPYFpwPzKBpIOBeYCH46IX5ZYi5mZ9aG0PYKI2CLpAmAB0AFcHxHLJJ2X5s8E/hY4CLhWEsCWiOgqqyYzM9uRImp22w9aXV1d0d3d3eoyzMzaiqRFfX3R9i+Lzcwy5yAwM8ucg8DMLHMOAjOzzDkIzMwy5yAwM8ucg8DMLHMOAjOzzDkIzMwy5yAwM8ucg8DMLHMOAjOzzDkIzMwy5yAwM8ucg8DMLHMOAjOzzDkIzMwy5yAwM8ucg8DMLHMOAjOzzDkIzMwy5yAwM8ucg8DMLHMOAjOzzDkIzMwy5yAwM8ucg8DMLHMOAjOzzDkIzMwy5yAwM8ucg8DMLHNDyly4pCnAPwMdwHUR8Y9V85Xmnwa8BEyPiPsbXce8xWv4q5uXEI1esNVtryF7cPn/PIr3Txrd6lKyNG/xGq5YsIK1GzYxasQwLp48kfdPGt3n9GbWtGbDJjoktkYwusk1tEIrX/O+lBYEkjqAa4BTgR5goaT5EfFwRbOpwIT0OB74avq3YeYtXsOFNy9p5CJtF7y8ZRsX3bIEoOV/9LmZt3gNM+YuZdPmrQCs2bCJGXOX0v3Es3x/0ZodpkP571F1TVsjml5DK/T1XkBrt7fMrqHjgJURsSoiXgHmAKdXtTkd+GYU7gFGSDqkkUVcsWBFIxdnr8G28PvRClcsWPH7D55emzZvZfa9q2tOb8Z7VKumZtfQCn29F63e3jKDYDSwumK8J03b2TZIOldSt6Tu9evX71QRazds2qn2Vi6/H83X12ve+y283vaNNNA6dte/k762q9XbW2YQqMa06r+8etoQEbMioisiujo7O3eqiFEjhu1UeyuX34/m6+s171Ct/37NeY8GWsfu+nfS13a1envLDIIeYGzF+Bhg7S60eU0unjyxkYuz12AP+f1ohYsnT2TY0I7tpg0b2sFZx4+tOb0Z71GtmppdQyv09V60envLPGtoITBB0nhgDTAN+GBVm/nABZLmUBwk3hgRTzWyiN4DMD5rqLV81lDr9L7mtc5U6XrDgS05g6WyppzOGurvvWglRR/9hA1ZuHQacBXF6aPXR8Q/SDoPICJmptNHrwamUJw+ek5EdPe3zK6uruju7reJmZlVkbQoIrpqzSv1dwQRcRtwW9W0mRXDAXyszBrMzKx//mWxmVnmHARmZplzEJiZZc5BYGaWuVLPGiqDpPXAE3U2Hwk8U2I5ZWnXusG1t4prb412qv0NEVHzF7ltFwQ7Q1J3X6dLDWbtWje49lZx7a3RzrVXcteQmVnmHARmZpnb3YNgVqsL2EXtWje49lZx7a3RzrX/3m59jMDMzAa2u+8RmJnZABwEZmaZ2y2DQNIUSSskrZR0WQvruF7SOkkPVUw7UNIdkn6V/j2gYt6MVPMKSZMrpr9V0tI078vpqq1I2kvSzWn6vZLGNajusZL+XdIjkpZJ+kQb1b63pPskPZBq//t2qb1ivR2SFkv6UTvVLunxtM4lkrrbrPYRkr4naXn6u39bu9TeEBGxWz0oLnn9KHAYsCfwAHBki2p5B/AW4KGKaV8ELkvDlwGXp+EjU617AePTNnSkefcBb6O4o9uPgalp+vnAzDQ8Dbi5QXUfArwlDe8H/DLV1w61CxiehocC9wIntEPtFdtwEXAT8KN2+ZtJy3scGFk1rV1qvxH4X2l4T2BEu9TekO1vdQEN36DiTVhQMT4DmNHCesaxfRCsAA5Jw4cAK2rVCSxI23IIsLxi+lnA1yrbpOEhFL9wVAnb8EPg1HarHdgHuJ/ipkdtUTvFXfp+CpzMq0HQLrU/zo5BMOhrB14HPFa9rHaovVGP3bFraDSwumK8J00bLA6OdBe29O/r0/S+6h6dhqunb/eciNgCbAQOamSxaRd2EsU367aoPXWtLAHWAXdERNvUTnEjp0uAbRXT2qX2AG6XtEjSuW1U+2HAeuAbqUvuOkn7tkntDbE7BkGtO3K3wzmyfdXd3/aUuq2ShgPfBy6MiOf7a9pHHS2pPSK2RsQxFN+uj5P05n6aD5raJb0XWBcRi+p9Sh91tOpv5u0R8RZgKvAxSe/op+1gqn0IRRfuVyNiEvAiRVdQXwZT7Q2xOwZBDzC2YnwMsLZFtdTytKRDANK/69L0vuruScPV07d7jqQhwP7As40oUtJQihD4TkTMbafae0XEBuBOiluhtkPtbwfeJ+lxYA5wsqRvt0ntRMTa9O864AfAcW1Sew/Qk/YcAb5HEQztUHtD7I5BsBCYIGm8pD0pDszMb3FNleYDf56G/5yi/713+rR0dsF4YAJwX9ol/a2kE9IZCB+pek7vsj4A/CxSJ+RrkdbzL8AjEfGlNqu9U9KINDwMeBewvB1qj4gZETEmIsZR/N3+LCI+1A61S9pX0n69w8C7gYfaofaI+DWwWtLENOkU4OF2qL1hWn2QoowHcBrFmS6PAp9qYR2zgaeAzRTfCD5K0S/4U+BX6d8DK9p/KtW8gnS2QZreRfGf6lHgal79RfjewHeBlRRnKxzWoLpPpNhtfRBYkh6ntUntRwGLU+0PAX+bpg/62qu24yRePVg86Gun6Gd/ID2W9f6/a4fa07KPAbrT38084IB2qb0RD19iwswsc7tj15CZme0EB4GZWeYcBGZmmXMQmJllzkFgZpY5B4EZIOmFVtdg1ioOAjOzzDkIzCpIOknSnRXXpv9OxTXlj5V0t4p7HdwnaT8V9z/4RroG/WJJ70xtp0uaJ+lWSY9JukDSRanNPZIOTO3eKOkn6UJt/yHpiFZuv+VpSKsLMBuEJgF/QHGdmF8Ab5d0H3AzcGZELJT0OmAT8AmAiPjD9CF+u6TD03LenJa1N8UvSi+NiEmS/oni8gNXUdz8/LyI+JWk44FrKS5BbdY0DgKzHd0XET0A6XLW4yguG/xURCwEiHQ1VkknAl9J05ZLegLoDYJ/j4jfUlx/ZiNwa5q+FDgqXd31j4Dvpp0OKG52YtZUDgKzHb1cMbyV4v+JqH3Z4FqXF661nG0V49vSMvcANkRxyWyzlvExArP6LAdGSToWIB0fGALcBZydph0OHEpxIbIBpb2KxyT9WXq+JB1dRvFm/XEQmNUhIl4BzgS+IukB4A6Kvv9rgQ5JSymOIUyPiJf7XtIOzgY+mpa5DDi9sZWbDcxXHzUzy5z3CMzMMucgMDPLnIPAzCxzDgIzs8w5CMzMMucgMDPLnIPAzCxz/x+n1LCAcPBtGwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Drawing the Scatter Plot\n",
    "plt.scatter(product_sales[\"Income\"], product_sales[\"Bought\"])\n",
    "plt.title('Income vs Bought Plot')\n",
    "plt.xlabel('Income')\n",
    "plt.ylabel('Bought')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "d445aa69-5334-4c4d-aef0-3cdc15532500",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYoAAAEWCAYAAAB42tAoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAAsTAAALEwEAmpwYAAArC0lEQVR4nO3deXxU5dn/8c9FSCCsEQFli6Ai7orGfXmsVsWtWH9WUavVR0txe9z3JcUNFVzrVqRudbdFi1Sk2GrrgkoERVAQBIEABZRVCIQk1++POQwzcbKQTObMTL7v1ysv5r7PmXO+MwlzzX3OmXvM3REREalJi7ADiIhIelOhEBGRWqlQiIhIrVQoRESkVioUIiJSKxUKERGplQqFSDNgZt+Z2c+baNu/N7Pnm2Lbkh5UKCQlmvKFKlOY2TNmVm5mP5rZGjP7zMz+Jw1yHWFmpXWsE5t9uZlNMLOdG7CvZv93kIlUKERS6153bwd0BB4HRptZTsiZ6mtT9p7AUuCZcONIqqhQSMqZ2blm9oGZjTCzFWY218yOi1neycyeNrNFwfI3Ypb91sxmB+9qx5hZ95hlbmYXmdms4B377Wa2g5lNNLPVZvaqmeXFrH+imX1uZivN7CMz27OGvE+Y2YhqfX8zsyuD29eZ2cJgnzPN7Ki6ngN3rwJeBDoB2wTbaWFmN5vZPDNbambPmVnHYNlP3vXHvjs3s3wzezZ4vr42s2sTjBL2NrOpZrbKzF4xs9Zm1hYYB3QPRgs/xj6nNWRfF2TfvYbn6xdmNj14Xt8zs12C/j8DhcCbwX6uret5kvSgQiFhOQCYCXQG7gX+ZGYWLPsz0AbYDegKPABgZkcCw4DTgG7APODlatsdAOwLHAhcC4wEzgJ6EXlhOyPY1j7AU8DvgK2BPwJjzKxVgqwvAqdvymdmWwHHAC+bWT/gEmA/d28PHAt8V9eDD0YR5wBzgSVB97nBz8+A7YF2wCN1bStQDPQO7nc08OsE65xG5PnpA+wJnOvua4HjgEXu3i74WVRH9nZEntMpCZbtBLwEXA50Ad4iUhjy3P1sYD5wUrCfe+v52CRkKhQSlnnu/qS7VwLPEnnh38bMuhF54Rri7ivcfaO7/zu4z1nAU+4+2d03ADcAB5lZ75jt3uPuq919OjAN+Ie7z3H3VUTeOfcP1vst8Ed3/8TdK939WWADkQJT3fuAA4cF7VOBicELaiXQCtjVzHLd/Tt3/7aWx321ma0E1gIPArcEz8Gmx3d/kPfH4PENMrOWtT6TEacBdwXPWSnwcIJ1Hnb3Re6+HHgT2Lse202UfTaRInZugnVOB/7u7hPcfSMwAsgHDt7CfUkaUaGQsPx3043gUAZEXnx6AcvdfUWC+3QnMorYdL8fgR+AHjHrLIm5XZag3S64vR1wVXB4ZGXwAtgr2Eccj8yc+TLBaAQ4E3ghWDabyLvn3wNLzezlOg7djHD3AiIvnkXA8JjDbnGPL7jdkuDQVB26Awti2gsSrPPfmNvr2Pxc1NcIdy9w923d/Rc1FMTqv6OqIEuPBOtKhlChkHSzAOhkZgUJli0i8gIPQHB8fWtgYQP3c2fwwrfpp427v1TD+i8Bp5rZdkQOm/110wJ3f9HdDw2yOXBPXTv3iGnAh8AJQXfc4yNyPL+CSLFbS+RwHBA9dNUlZt3FRE4yb9KrrgyxcbZg3bpU/x1ZkGXT70jTVWcgFQpJK+6+mMghosfMbCszyzWzw4PFLwLnmdnewbmEu4BP3P27BuzqSWCImR1gEW3N7AQza19DrinAMmAUMN7dVwKYWT8zOzLIs57IqKUy0Taqs8jlpYcC04Oul4ArzKxPcB7gLuAVd68AvgFaBxlzgZuJHPLa5FXghuA560HkvEl9LQG23nTivJFeBU4ws6OCnFcROaT3Ucy+tk/CfiSFVCgkHZ0NbARmELkM83IAd/8ncAuRd/OLgR2AQQ3ZgbuXEDlP8Qiwgshx93PruNtLwM+JFKxNWgF3A98TObTTFbixlm1cG1zxsxb4B/A0kRPpEDm5/mfgP0ROcq8HLg3yrgIuIlKoFhIZYcRe1XRb0J4LvAP8hcgLdJ3cfUbw2OYEh+Fqveqpjm3NJHIi/Q9EnpOTiJy8Lg9WGQbcHOzn6obuR1LL9MVFItnHzC4EBrl76B/ok8ynEYVIFjCzbmZ2SPBZjH5EDvm8HnYuyQ71uexORNJfHpFDWH2AlUSu0noszECSPXToSUREaqVDTyIiUqusPPTUuXNn7927d9gxREQyxmefffa9u3dJtCwrC0Xv3r0pKSkJO4aISMYws3k1LdOhJxERqZUKhYiI1EqFQkREaqVCISIitVKhEBGRWoVaKMzsqeArH6fVsPys4KsbpwZfVblXqjOKiDR3YY8oniHy1Yw1mQv8j7vvCdxO5GstRUQkhUItFO7+H2B5Lcs/ivmms4+J/2IWEREJPD3lad6Z806TbDuTPnB3PpEvtEnIzAYDgwEKCwtTlUlEJFSL1iyix/2bv2nWi5M/f1/Yh57qxcx+RqRQXFfTOu4+0t2L3L2oS5eEn0IXEckqV7x9RVyR+O9V/61l7YZL+xGFme1J5Fu9jnP3H8LOIyIStlk/zGKnR3aKtu875j6uPOjKJttfWhcKMysERgNnu/s3YecREQmTu3P6X07nta9ei/atun4VHVp1aNL9hloozOwl4Aigs5mVAsVALoC7PwHcCmwNPGZmABXuXhROWhGR8ExePJl9R+4bbT938nOcvdfZKdl3qIXC3c+oY/kFwAUpiiMiknaqvIrDnz6cDxd8CECXNl2Yf8V8WrdsnbIMaX3oSUSkOXt37rsc+dyR0fbYM8Zywk4npDyHCoWISJrZWLmRfo/0Y+7KuQDstc1efDb4M3Ja5ISSR4VCRCSN/PWrv3Lqa6dG2x/+74cc3OvgEBOpUIiIpIW15WvpdG8nyivLATi+7/GMPWMswYU8oVKhEBEJ2R9L/siQvw+JtqddOI3duu4WYqJ4KhQiIiFZXracre/dOto+v//5jPrFqBATJaZCISISgtv/fTu3vndrtP3dZd+xXcF2ISaqmQqFiEgKLVy9kJ4PbJ4I+6bDbuKOI+8IMVHdVChERFLk4r9fzGMlj0XbS69eSpe26T+JqQqFiEgTm/H9DHZ5dJdo++EBD3PpAZeGmGjLqFCIiDQRd+eUV0/hjRlvRPvW3LCGdnntwgvVACoUIiJNYNLCSew/av9o+8VTXuSMPWqd3i5tqVCIiCRRlVdx0J8O4tOFnwLQvX135l42l7ycvJCTNZwKhYhIkkz4dgLHPH9MtP32WW9z7I7HhpgoOVQoREQaqbyynO0f2p6FaxYCsF/3/Zh4/sTQJvFLNhUKEZFGeHnay5zx183nHj654BP277F/LffIPCoUIiIN8GP5j3QY1gHHARjYbyCvn/56Wkzil2wqFCIiW+iRTx/h0nGbPwfx9cVfs3PnnUNM1LRUKERE6un7dd/TZfjmT1IP2XcIj5/4eIiJUkOFQkSkHg57+jA+mP9BtD3/8vn06tgrxESpE2qhMLOngBOBpe6+e4LlBjwEHA+sA85198mpTflTb0xZyPDxM1m0sozuBflcc2w/Tu7fI232H3Y+SS9H3/8es5auTeo2C/JzMYMV6zbSwqDKk7r5tJCf24J9Cgt4d85kFre+JNrfYeP/Y6uK8zhs2FRgao33N8CBrdrk4g4ryzY2eeZNDtmhEy/89qCkbc/cw/sNm9nhwI/AczUUiuOBS4kUigOAh9z9gLq2W1RU5CUlJcmOC0RehG8Y/SVlGyujffm5OQw7ZY+UvBjXtf+w80l6aYoi0ZwsaP1rqmxltN2j7Bla0jm8QFtgS4uFmX3m7kWJlrVIWqoGcPf/AMtrWWUgkSLi7v4xUGBm3VKTLrHh42fGvQgDlG2sZPj4mWmx/7DzSXpRkWiY9S2mMi//xGiRaFtxNNuVjc2YIgHw4be1vbRumXQ/R9EDWBDTLg36Fldf0cwGA4MBCgsLmyzQopVlW9Sf6v2HnU8kkznO/PyT4vp6lr1MDpk1iV+yhTqiqIdEFyQnPFbm7iPdvcjdi7p0abr53bsX5G9Rf6r3H3Y+kUy1tsUHcUWi48Yz2a5sbLMvEpD+haIUiL2soCewKKQsAFxzbD/yc+M/lp+fm8M1x/ZLi/2HnU/SS9+ubcOOkPacSubln8j3re6O9hWWjaag4swQUzXeITt0Stq20r1QjAHOsYgDgVXu/pPDTql0cv8eDDtlD3oU5GNAj4L8lJ4ormv/YeeT9DLhyiOapFgU5OeyVZtcAFpk8AeRV+e8yfz8gdF2p/JL2K5sLEYe+bktGvViu+lp2apNLgX5uY1MumWy7aqnl4AjgM7AEqAYyAVw9yeCy2MfAQYQuTz2PHev83KmprzqSUQy39rytbQbFn9IqeKWiqyZxK8harvqKdST2e5e67d4eKSKXZyiOCLSDFw34Tru/ejeaHvMoDGc1O+kWu4h6X7Vk4hIUlSffgOg6taqrJzEL9nS/RyFiEijDfrLoLgi8dH/foQXu4pEPWlEISJZa+6KuWz/8PbRdt9Offnm0m9CTJSZVChEJCvt9+R+lCzafFFLtk8F3pRUKEQkq0xZPIV9Ru4TbQ/YcQDjzhoXYqLMp0IhIlmj4O4CVm1YFW0vunIR3dqHOj1cVtDJbBHJeP+c809sqEWLxJB9h+DFriKRJBpRiEjGcnda3Bb/fnfV9avo0KpDSImyk0YUIpKRXvzyxbgiceeRd+LFriLRBDSiEJGMsrFyI3l35MX1rb9pPa1atgopUfbTiEJEMsZ9H90XVySeGfgMXuwqEk1MIwoRSXs/lv9I+2Ht4/oqb62khem9biroWRaRtHb525fHFYlxZ43Di11FIoU0ohCRtLTkxyVse9+20XZeTh4bbt4QYqLmSyVZRNLOL1/5ZVyR+PSCT1UkQqQRhYikjdnLZ9P3D32j7d277s6XF34ZYiIBFQoRSRN7PL4H05ZOi7ZnXTqLHTvtGGIi2USFQkRCNWnhJPYftX+0PbDfQN4Y9EZ4geQnVChEJDSt72jNhsrN5x7+e9V/2abdNiEmkkR0MltEUu7t2W9jQy1aJC474DK82FUk0lSoIwozGwA8BOQAo9z97mrLOwLPA4VEso5w96dTHlREkqLKq8i5LSeub/X1q2nfqn0N95B0ENqIwsxygEeB44BdgTPMbNdqq10MfOXuewFHAPeZWR4iknGe/fzZuCIx/OjheLGrSGSAMEcU+wOz3X0OgJm9DAwEvopZx4H2FvkG9HbAcqAi1UFFpOHKK8tpdUf8XEzlN5eTm5MbUiLZUmGeo+gBLIhplwZ9sR4BdgEWAV8Cl7l7VaKNmdlgMysxs5Jly5Y1RV4R2ULD3h8WVySe/+XzeLGrSGSYMEcUlqDPq7WPBT4HjgR2ACaY2fvuvvond3QfCYwEKCoqqr4dEUmh1RtW0/HujnF9VbdWETk4IJkmzBFFKdArpt2TyMgh1nnAaI+YDcwFdk5RPhFpgIv+flFckZhw9gS82FUkMliYI4pJQF8z6wMsBAYBZ1ZbZz5wFPC+mW0D9APmpDSliNTL4jWL6X5/92i7Q6sOrLp+VYiJJFlCKxTuXmFmlwDjiVwe+5S7TzezIcHyJ4DbgWfM7Esih6quc/fvw8osIokd/8LxjJs9LtqePHgy/bv1DzGRJFOon6Nw97eAt6r1PRFzexFwTKpziUj9zPx+Jjs/uvlo8L7d9qVkcEmIiaQpaAoPEWmQfo/045sfvom25/zfHPps1SfERNJUNIWHiGyRj0s/xoZatEictttpeLGrSGQxjShEpF7cnRa3xb+3XHbNMjq36RxSIkkVjShEpE5vznwzrkhcfdDVeLGrSDQTGlGISI0qqyppeXv8y8SPN/xI27y2ISWSMGhEISIJjZo8Kq5IPHjsg3ixq0g0QxpRiEic9RXryb8zP65v4y0badlCLxfNlUYUIhJ1279viysSr576Kl7sKhLNnH77IsLK9SvZ6p6t4vo0iZ9sohGFSDN3wZgL4orEu795V5P4SRyNKESaqdLVpfR6YPMEzl3bdmXJ1UtCTCTpSoVCpBk66rmj+Nfcf0XbXwz5gj232TPERJLOVChEmpGvln3Fbo/tFm0f0usQPvjfD0JMJJlAhUKkmdjuwe2Yv2p+tD3v8nkUdiwMMZFkCp3MFslyH8z/ABtq0SJx9p5n48WuIiH1phGFSJZKNInfD9f+QKf8TiElkkylEYVIFnr969fjisRNh92EF7uKhDSIRhQiWSTRJH7rblxHfm5+DfcQqZtGFCJZ4vFJj8cViceOfwwvdhUJaTSNKEQyXNnGMtrc1Saur+KWCnJa5ISUSLJNvUYUZnZPffq2lJkNMLOZZjbbzK6vYZ0jzOxzM5tuZv9u7D5FssnN/7o5rkiMPm00XuwqEpJU9R1RHA1cV63vuAR99WZmOcCjwbZLgUlmNsbdv4pZpwB4DBjg7vPNrGtD9yeSTZaXLWfre7eO69MkftJUah1RmNmFZvYl0M/Mpsb8zAWmNnLf+wOz3X2Ou5cDLwMDq61zJjDa3ecDuPvSRu5TJOOd8/o5cUXi/fPe1yR+0qTqGlG8CIwDhgGxh4bWuPvyRu67B7Agpl0KHFBtnZ2AXDN7D2gPPOTuzzVyvyIZad7KefR+qHe0XdixkHmXzwsvkDQbtRYKd18FrALOCA4VbRPcp52Ztdv0Tr+BEr398QT59gWOAvKBiWb2sbt/85ONmQ0GBgMUFuoTp5JdDnnqED5a8FG0Pf2i6ezaZdcQE0lzUq9zFGZ2CfB7YAlQFXQ70JjpJkuBXjHtnsCiBOt87+5rgbVm9h9gL+AnhcLdRwIjAYqKiqoXHJGMNHXJVPZ6Yq9o+6g+R/HOOe+EmEiao/qezL4c6OfuPyRx35OAvmbWB1gIDCJyTiLW34BHzKwlkEfk0NQDScwgkra2GbENS9duPi234IoF9OzQM8RE0lzV9wN3C4gcgkoad68ALgHGA18Dr7r7dDMbYmZDgnW+Bt4mcuL8U2CUu09LZg6RdPPed+9hQy1aJM7vfz5e7CoSEhpzr/kojZldGdzcDegH/B3YsGm5u9/fpOkaqKioyEtKSsKOIbJFEk3it+K6FRS0LggnkDQrZvaZuxclWlbXiKJ98DMfmEDk8E/7mB8RSYJXp78aVySGHjEUL3YVCUkLdV31NDRVQUSao4qqCnJvz43rK7upjNYtW4eUSOSn6nvV05v89NLVVUAJ8Ed3X5/sYCLZ7qGPH+Ly8ZdH20+e9CQX7HNBeIFEalDfq57mAF2Al4L26UQuld0JeBI4O/nRRLLT2vK1tBvWLq5Pk/hJOqtvoejv7ofHtN80s/+4++FmNr0pgolko2v+cQ0jJo6ItscMGsNJ/U4KMZFI3epbKLqYWeGmT2KbWSHQOVhW3iTJRLLI9+u+p8vwLtG2YVTeWqn5mSQj1LdQXAV8YGbfEpl6ow9wkZm1BZ5tqnAi2eC0107jta9ei7Ynnj+RA3seGGIikS1Tr0Lh7m+ZWV9gZyKFYkbMCewHmyibSEabu2Iu2z+8fbS909Y7MfOSmSEmEmmY+l71dE61rj3NDM3kKpJY0cgiPlv8WbQ94+IZ9OvcL8REIg1X30NP+8Xcbk1kNtfJgAqFSIwpi6ewz8h9ou3jdjyOt856K8REIo1X30NPl8a2zawj8OcmSSSSoToM68Ca8jXR9qIrF9GtfbcQE4kkR30nBaxuHdA3mUFEMtU7c97Bhlq0SFxYdCFe7CoSkjUa8snsHGAX4NWmCiWSCRJN4rfq+lV0aNUhpEQiTaO+5yhGxNyuAOa5e2kT5BHJCC9MfYFfv/7raPuuI+/ihsNuCDGRSNOp7zmKf5vZNmw+qT2r6SKJpK+NlRvJuyMvrm/DzRvIy8mr4R4ima9e5yjM7DQiXxz0K+A04BMzO7Upg4mkmxEfjYgrEs8MfAYvdhUJyXr1PfR0E7Cfuy8FMLMuwDvAX5oqmEi6WLNhDR3ujj/vUHlrJS2sodeCiGSW+v6lt9hUJAI/bMF9RTLWZeMuiysS484ahxe7ioQ0K/UdUbxtZuOJn2ZcnyKSrLXkxyVse9+20XarnFasv1lfuyLNU31PZl9jZqcAhxKZ62mku7/epMlEQjLw5YGMmTkm2p7020kUdU/4VcIizUJ9RxS4+2hgtJl1JnLoqdHMbADwEJHPZoxy97trWG8/4GPgdHfXeRFpErN+mMVOj+wUbe+5zZ58MeSLEBOJpIdaD7Sa2YFm9p6ZjTaz/mY2DZgGLAle5BvMzHKAR4HjgF2BM8xs1xrWuwcY35j9idRmj8f3iCsSsy6dpSIhEqjrjNwjwF1Ezk38C7jA3bcFDgeGNXLf+wOz3X2Ou5cDLwMDE6x3KfBXYGmCZSKNMmnhJGyoMW3pNABO3vlkvNjZsdOOIScTSR91HXpq6e7/ADCz29z9YwB3n5GEb+bqASyIaZcCB8SuYGY9gF8CRxI/g61Io7W6oxXllZu/oHHJ1Uvo2rZriIlE0lNdI4qqmNtl1ZY5jZOo0lTf5oPAde5eWefGzAabWYmZlSxbtqyR0SSbjZs1Dhtq0SJx+QGX48WuIiFSg7pGFHuZ2WoiL+r5wW2CdutG7rsU6BXT7gksqrZOEfByMHrpDBxvZhXu/kb1jbn7SGAkQFFRUWOLmGShKq8i57acuL41N6yhXV67kBKJZIZaRxTunuPuHdy9vbu3DG5vauc2ct+TgL5m1sfM8oBBwJjYFdy9j7v3dvfeRD4FflGiIiFSl2c+fyauSIw4egRe7CoSIvVQ78tjk83dK8zsEiJXM+UAT7n7dDMbEix/Iqxskj02VGyg9Z3xg9/ym8vJzWns+xyR5iO0QgHg7m9R7RPeNRUIdz83FZkkewx7fxg3/uvGaPuFU17gzD3ODDGRSGYKtVCINIXVG1bT8e6OcX1Vt1aRhCv1RJolzWwmWeXCsRfGFYkJZ0/Ai11FQqQRNKKQrLB4zWK639892u7YqiMrr18ZXiCRLKJCIRnvuBeO4+3Zb0fbU343hb233Tu8QCJZRoVCMtaM72ewy6O7RNtF3YuY9NtJISYSyU4qFJKR+v6hL7OXz4625/zfHPps1SfERCLZSyezJaNMXDARG2rRInH6bqfjxa4iIdKENKKQjODutLgt/n3NsmuW0blN55ASiTQfGlFI2ntz5ptxReLag6/Fi11FQiRFNKKQtFVZVUnL2+P/RNfeuJY2uW1CSiTSPGlEIWnpyc+ejCsSDw14CC92FQmREGhEIWllfcV68u/Mj+vbeMtGWrbQn6pIWDSikLTx+/d+H1ckXj31VbzYVSREQqb/gRK6letXstU9W8X1aRI/kfShEYWE6vy/nR9XJN77zXuaxE8kzWhEIaEoXV1Krwc2fxPutu22ZfFVi0NMJCI1UaGQlDvy2SN597t3o+2pQ6ayxzZ7hJhIRGqjQiEpM33pdHZ/fPdo+9DCQ3n/vPdDTCQi9aFCISlR+EAhC1YviLbnXT6Pwo6FISYSkfrSyWxpUu/Pex8batEicfaeZ+PFriIhkkE0opAmkWgSvx+u/YFO+Z1CSiQiDRXqiMLMBpjZTDObbWbXJ1h+lplNDX4+MrO9wsgpW2b016PjisTNh92MF7uKhEiGCm1EYWY5wKPA0UApMMnMxrj7VzGrzQX+x91XmNlxwEjggNSnlfqoqKog9/bcuL51N64jPze/hnuISCYIc0SxPzDb3ee4eznwMjAwdgV3/8jdVwTNj4GeKc4o9fTYpMfiisTjJzyOF7uKhEgWCPMcRQ9gQUy7lNpHC+cD42paaGaDgcEAhYU6UZoq6zauo+1dbeP6Km6pIKdFTkiJRCTZwhxRJJqjwROuaPYzIoXiupo25u4j3b3I3Yu6dOmSpIhSm5v+eVNckXj99NfxYleREMkyYY4oSoFeMe2ewKLqK5nZnsAo4Dh3/yFF2aQWP6z7gc7D479dTpP4iWSvMEcUk4C+ZtbHzPKAQcCY2BXMrBAYDZzt7t+EkFGq+fXoX8cViQ/O+0CT+IlkudBGFO5eYWaXAOOBHOApd59uZkOC5U8AtwJbA48FL0QV7l4UVubmbN7KefR+qHe03bugN3MvmxteIBFJGXNPeFogoxUVFXlJSUnYMbLGwX86mImlE6Pt6RdNZ9cuu4aYSESSzcw+q+mNuD6ZLTWaumQqez2x+TOOR/U5infOeSfERCISBhUKSajr8K4sW7cs2i69opQeHXqEmEhEwqJJASXOu3PfxYZatEhc0P8CvNhVJESaMY0oBEg8id+K61ZQ0LognEAikjY0ohBemfZKXJG47Yjb8GJXkRARQCOKZm1j5Uby7siL6yu7qYzWLVuHlEhE0pFGFM3Ugx8/GFckRp00Ci92FQkR+QmNKJqZteVraTesXVyfJvETkdpoRNGMXPOPa+KKxNgzxmoSPxGpk0YUzcCytcvoOqJrtN3CWlBxS4XmZxKRetGIIsv96rVfxRWJj8//mMpbK1UkRKTeNKLIUnNWzGGHh3eItvtt3Y8Zl8wIMZGIZCoViiy078h9mbx4crQ94+IZ9OvcL8REIpLJVCiyyOTFk9l35L7R9gl9T2DsmWNDTCQi2UCFIkt0GNaBNeVrou3FVy1m23bbhphIRLKFTmZnuAnfTsCGWrRIXLzfxXixq0iISNJoRJGhqryKnNviP/+w6vpVdGjVIaREIpKtNKLIQM9PfT6uSAw7ahhe7CoSItIkNKLIIOWV5bS6o1Vc34abN5CXk1fDPUREGk8jigwx/MPhcUXi2ZOfxYtdRUJEmpy5e3g7NxsAPATkAKPc/e5qyy1YfjywDjjX3Sf/ZEPVFBUVeUlJyRZlOfr+95i1dO0W3ScVqljHgvzT4voKy8ZgWVbjW7YwRvxqL07ur2/SC8sbUxYyfPxMFq0so3tBPtccG/nsTfW+VP6ONmVauLKMHDMq3ekRQo5USvR7SMVjNbPP3L0o0bLQDj2ZWQ7wKHA0UApMMrMx7v5VzGrHAX2DnwOAx4N/kypdi8Ty3D+ypuWb0XbXDUPJr9q3lntkrooq54pXPgfI2heAdPbGlIXcMPpLyjZWArBwZRnX/OULcNhY5dG+G0Z/CaTmd1Q9U6WHkyOVEv0e0uGxhvm2dH9gtrvPcfdy4GVgYLV1BgLPecTHQIGZdUt2kHQrEpWsYF7+idEiYd6K7crGZm2R2MSJvHuV1Bs+fmb0xWmTjZUeLRKblG2sTNnvKFGmMHKkUqLHnA6PNcyT2T2ABTHtUn46Wki0Tg9gcfWNmdlgYDBAYWFhUoOm0tK82yjL+TTa3nb9A7TyviEmSq1FK8vCjtAsbcnznqrfUV37yca/lZoeU9iPNcwRRaLpS6ufMKnPOpFO95HuXuTuRV26dGl0uFTbaAuZl39itEjkVvVmu7KxzapIAHQvyA87QrO0Jc97qn5Hde0nG/9WanpMYT/WMAtFKdArpt0TWNSAdRqtb9e2yd7kFlnU6iIWtf5dtN19/ZN03/BIiInCYRA9gSqpdc2x/cjPjf8AZ26Okdsi/r1afm5Oyn5HiTKFkSOVEj3mdHisYRaKSUBfM+tjZnnAIGBMtXXGAOdYxIHAKnf/yWGnxppw5RGhFIsNNpN5+SeyscV8APIrD2K7srHketJPw6S9li2MB07fO+tOTmaKk/v3YNgpe9CjIB8DehTkM/zUvRj+q73i+oadskfKfkexmQBygu9QSXWOVEr0e0iHxxr25bHHAw8SuTz2KXe/08yGALj7E8HlsY8AA4hcHnueu9d53WtDLo9NJXcn7448Kqoqon1Lrl5C17Zda7mXiEjTScvLYwHc/S3grWp9T8TcduDiVOdqSm/NeosTXjwh2r7iwCu4/9j7Q0wkIlI7TeGRIokm8Vtzwxra5bULKZGISP1k18d709Qznz8TVyRGHD0CL3YVCRHJCBpRNKENFRtofWfruL7ym8vJzckNKZGIyJbTiKKJ3PmfO+OKxIunvIgXu4qEiGQcjSiSbNX6VRTcUxDXV3VrFWaJPjsoIpL+NKJIoiFjh8QViXfOfgcvdhUJEcloGlEkwaI1i+hx/+YPxBS0LmDFdStCTCQikjwqFI004PkBjP92fLQ95XdT2HvbvcMLJCKSZCoUDTTj+xns8ugu0fb+Pfbnkws+CTGRiEjTUKFogB0f3pFvV3wbbc+9bC69C3qHF0hEpAnpZPYW+GjBR9hQixaJQbsPwotdRUJEsppGFPXg7rS4Lb6mLrtmGZ3bdA4pkYhI6mhEUYeSRSVxReLag6/Fi11FQkSaDY0oalDlVRz61KFMLJ0Y7Vt741ra5LYJMZWISOqpUCTwzpx3OPrPR0fbfz/z7xzf9/gQE4mIhEeFIkZ5ZTk7PrwjC1YvAKD/tv2Z9NtJ5LRI/HWMIiLNgQpFjFZ3tIrennj+RA7seWCIaURE0oMKRYz7jrmPSYsm8eIpL2p+JhGRgApFjCsPujLsCCIiaUeXx4qISK1CKRRm1snMJpjZrODfrRKs08vM3jWzr81supldFkZWEZHmLqwRxfXAP929L/DPoF1dBXCVu+8CHAhcbGa7pjCjiIgQXqEYCDwb3H4WOLn6Cu6+2N0nB7fXAF8DPaqvJyIiTSusQrGNuy+GSEEAuta2spn1BvoDNc7jbWaDzazEzEqWLVuWzKwiIs1ak131ZGbvANsmWHTTFm6nHfBX4HJ3X13Teu4+EhgJUFRU5FuyDxERqVmTFQp3/3lNy8xsiZl1c/fFZtYNWFrDerlEisQL7j66iaKKiEgtwjr0NAb4TXD7N8Dfqq9gkU+8/Qn42t3vT2E2ERGJYe6pP0pjZlsDrwKFwHzgV+6+3My6A6Pc/XgzOxR4H/gSqArueqO7v1WP7S8D5tUjSmfg+4Y8hjSg7OFQ9tTL1NyQWdm3c/cuiRaEUijShZmVuHtR2DkaQtnDoeypl6m5IbOzx9Ins0VEpFYqFCIiUqvmXihGhh2gEZQ9HMqeepmaGzI7e1SzPkchIiJ1a+4jChERqYMKhYiI1KpZFgozG2BmM81stpklmrk2VTmeMrOlZjYtpq/GKdjN7IYg80wzOzamf18z+zJY9nDwYUXMrJWZvRL0fxLMmZWs7AmngU/3/GbW2sw+NbMvgtxDMyF3tceQY2ZTzGxsJmU3s++CfX5uZiUZlr3AzP5iZjOCv/mDMiV7Urh7s/oBcoBvge2BPOALYNeQshwO7ANMi+m7F7g+uH09cE9we9cgayugT/AYcoJlnwIHAQaMA44L+i8CnghuDwJeSWL2bsA+we32wDdBxrTOH+yjXXA7l8hEkweme+5qj+FK4EVgbIb9zXwHdK7WlynZnwUuCG7nAQWZkj0pjz/sACl/wJFf0viY9g3ADSHm6U18oZgJdAtudwNmJsoJjA8eSzdgRkz/GcAfY9cJbrck8glRa6LH8Tfg6EzKD7QBJgMHZEpuoCeR73A5ks2FIlOyf8dPC0XaZwc6AHOrbysTsifrpzkeeuoBLIhpl5Je33NR0xTsNeXuEdyu3h93H3evAFYBWyc7sMVPA5/2+YNDN58TmYxygrtnRO7Ag8C1bJ7WhgzK7sA/zOwzMxucQdm3B5YBTweH/EaZWdsMyZ4UzbFQWIK+TLhGuKbctT2eJn+sVs9p4GvJkvL87l7p7nsTeXe+v5ntXsvqaZPbzE4Elrr7Z/W9Sw05wvqbOcTd9wGOI/KNlYfXsm46ZW9J5BDx4+7eH1hL4m/l3CSdsidFcywUpUCvmHZPYFFIWRJZYpGp17H4Kdhryl0a3K7eH3cfM2sJdASWJyuoJZ4GPmPyu/tK4D1gQIbkPgT4hZl9B7wMHGlmz2dIdtx9UfDvUuB1YP8MyV4KlAYjT4C/ECkcmZA9KZpjoZgE9DWzPmaWR+TE0ZiQM8WqaQr2McCg4OqIPkBf4NNgyLvGzA4MrqA4p9p9Nm3rVOBfHhwEbaxgX4mmgU/r/GbWxcwKgtv5wM+BGemeG8Ddb3D3nu7em8jf7b/c/deZkN3M2ppZ+023gWOAaZmQ3d3/Cywws35B11HAV5mQPWnCPkkSxg9wPJGrdL4Fbgoxx0vAYmAjkXcU5xM5LvlPYFbwb6eY9W8KMs8kuFoi6C8i8p/uW+ARNn/ivjXwGjCbyNUW2ycx+6FEhsZTgc+Dn+PTPT+wJzAlyD0NuDXoT+vcCR7HEWw+mZ322Ykc5/8i+Jm+6f9dJmQPtr03UBL83bwBbJUp2ZPxoyk8RESkVs3x0JOIiGwBFQoREamVCoWIiNRKhUJERGqlQiEiIrVSoRCpJzP7MewMImFQoRARkVqpUIhsITM7wszei/l+ghdivldgPzP7yCLfd/GpmbW3yHdgPB18D8EUM/tZsO65ZvaGmb1pZnPN7BIzuzJY52Mz6xSst4OZvR1Mpve+me0c5uOX5qdl2AFEMlR/YDcic/V8CBxiZp8CrwCnu/skM+sAlAGXAbj7HsGL/D/MbKdgO7sH22pN5FO517l7fzN7gMgUDw8CI4Eh7j7LzA4AHiMyzbhISqhQiDTMp+5eChBMWd6byNTQi919EoAHs+ma2aHAH4K+GWY2D9hUKN519zVE5gBaBbwZ9H8J7BnMznsw8FowaIHIF+KIpIwKhUjDbIi5XUnk/5KReGroRFNIJ9pOVUy7KthmC2ClR6ZFFwmFzlGIJM8MoLuZ7QcQnJ9oCfwHOCvo2wkoJDJZXJ2CUclcM/tVcH8zs72aIrxITVQoRJLE3cuB04E/mNkXwAQi5x4eA3LM7Esi5zDOdfcNNW/pJ84Czg+2OR0YmNzkIrXT7LEiIlIrjShERKRWKhQiIlIrFQoREamVCoWIiNRKhUJERGqlQiEiIrVSoRARkVr9f6iTdoef8SUOAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Drawing the Regression line\n",
    "pred_values= fitted10.predict(product_sales[\"Income\"]) \n",
    "plt.scatter(product_sales[\"Income\"], product_sales[\"Bought\"])\n",
    "plt.plot(product_sales[\"Income\"], pred_values, color='green')\n",
    "plt.title('Income vs Bought Plot')\n",
    "plt.xlabel('Income')\n",
    "plt.ylabel('Bought')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "18f64f6d-9481-47ea-87bc-582bcf6d706f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization terminated successfully.\n",
      "         Current function value: 0.101165\n",
      "         Iterations 9\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>Logit Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>        <td>Bought</td>      <th>  No. Observations:  </th>   <td>   467</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>   <td>   465</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>   <td>     1</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>            <td>Tue, 31 Jan 2023</td> <th>  Pseudo R-squ.:     </th>   <td>0.8525</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                <td>18:32:20</td>     <th>  Log-Likelihood:    </th>  <td> -47.244</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th>  <td> -320.21</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>9.637e-121</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>const</th>  <td>   -7.0288</td> <td>    0.739</td> <td>   -9.505</td> <td> 0.000</td> <td>   -8.478</td> <td>   -5.579</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Income</th> <td>    0.0002</td> <td>  2.1e-05</td> <td>   10.397</td> <td> 0.000</td> <td>    0.000</td> <td>    0.000</td>\n",
       "</tr>\n",
       "</table>"
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                           Logit Regression Results                           \n",
       "==============================================================================\n",
       "Dep. Variable:                 Bought   No. Observations:                  467\n",
       "Model:                          Logit   Df Residuals:                      465\n",
       "Method:                           MLE   Df Model:                            1\n",
       "Date:                Tue, 31 Jan 2023   Pseudo R-squ.:                  0.8525\n",
       "Time:                        18:32:20   Log-Likelihood:                -47.244\n",
       "converged:                       True   LL-Null:                       -320.21\n",
       "Covariance Type:            nonrobust   LLR p-value:                9.637e-121\n",
       "==============================================================================\n",
       "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "const         -7.0288      0.739     -9.505      0.000      -8.478      -5.579\n",
       "Income         0.0002    2.1e-05     10.397      0.000       0.000       0.000\n",
       "==============================================================================\n",
       "\"\"\""
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#3.9 Logistic Regression Model building\n",
    "import statsmodels.api as sm\n",
    "logit_model=sm.Logit(product_sales[\"Bought\"],product_sales[\"Income\"])\n",
    "#Model with intercept\n",
    "logit_model1=sm.Logit(product_sales[\"Bought\"],sm.add_constant(product_sales[\"Income\"]))\n",
    "logit_fit1=logit_model1.fit()\n",
    "logit_fit1.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "a816d73c-bf4d-412d-a289-89a7c1c7edfc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    0.002118\n",
      "1    0.999990\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "#prediction from the model\n",
    "new_data=pd.DataFrame({\"Constant\":[1,1],\"Income\":[4000, 85000]})\n",
    "print(logit_fit1.predict(new_data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "66165409-aede-47fb-b160-25bfbcd85fa4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Drawing the Logistic line\n",
    "\n",
    "new_data=product_sales.drop([\"Bought\"], axis=1)\n",
    "new_data[\"Constant\"]=1\n",
    "new_data=new_data[[\"Constant\",\"Income\"]]\n",
    "#Pass the variables to get the predicted values. Add actual values in a new column \n",
    "new_data[\"pred_values\"]= logit_fit1.predict(new_data)\n",
    "new_data[\"Actual\"]=product_sales[\"Bought\"]\n",
    "#Sort the data and draw the graph\n",
    "new_data=new_data.sort_values([\"pred_values\"])\n",
    "plt.scatter(new_data[\"Income\"], new_data[\"Actual\"])\n",
    "plt.plot(new_data[\"Income\"], new_data[\"pred_values\"], color='green')\n",
    "#Add lables and title \n",
    "plt.title('Predicted vs Actual Plot')\n",
    "plt.xlabel('Income')\n",
    "plt.ylabel('Bought')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "94e83377-3821-4bd5-b1d0-8fd4ace29b75",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Income  Bought\n",
      "0   2380.0       0\n",
      "1   7351.1       0\n",
      "2  48224.4       1\n",
      "3   4833.0       0\n",
      "4  18426.1       0\n",
      "5  52709.0       1\n",
      "6  54926.7       1\n",
      "7  52109.3       1\n",
      "8   8658.6       0\n",
      "9  12227.9       0\n"
     ]
    }
   ],
   "source": [
    "#Accuracy of the model \n",
    "print(product_sales.head(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "4d638d73-7a6e-41ae-8678-6943c8fbe4d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Bought  pred_Bought\n",
      "0         0          0.0\n",
      "1         0          0.0\n",
      "2         1          1.0\n",
      "3         0          0.0\n",
      "4         0          0.0\n",
      "..      ...          ...\n",
      "462       0          0.0\n",
      "463       0          0.0\n",
      "464       1          1.0\n",
      "465       1          1.0\n",
      "466       1          1.0\n",
      "\n",
      "[467 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "#Add a new column for intercept. This will be used in prediction\n",
    "product_sales[\"Constant\"]=1\n",
    "#Get the predicted values into a new column\n",
    "product_sales[\"pred_Bought\"]=logit_fit1.predict(product_sales[[\"Constant\",\"Income\"]])\n",
    "product_sales[\"pred_Bought\"]=round(product_sales[\"pred_Bought\"])\n",
    "\n",
    "#Data after updating with predicted values\n",
    "print(product_sales[[\"Bought\",\"pred_Bought\"]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "6badc652-0eb5-433e-9b45-f10d0cd91149",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[257   5]\n",
      " [  3 202]]\n"
     ]
    }
   ],
   "source": [
    "cm1 = confusion_matrix(product_sales[\"Bought\"],product_sales[\"pred_Bought\"])\n",
    "print(cm1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "1459fa28-f600-48e1-9808-9303b7a311ea",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9828693790149893\n"
     ]
    }
   ],
   "source": [
    "accuracy1=(cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])\n",
    "print(accuracy1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "2d4ee999-3893-475d-a554-59864d8f9cc2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(100000, 11)\n",
      "Index(['Id', 'Active_cust', 'estimated_income', 'months_on_network',\n",
      "       'complaints_count', 'plan_changes_count', 'relocated_new_place',\n",
      "       'monthly_bill_avg', 'CSAT_Survey_Score', 'high_talktime_flag',\n",
      "       'internet_time'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "#3.10 Multiple Logistic Regression Line\n",
    "\n",
    "telco_cust=pd.read_csv(r\"C:\\Users\\SREEHARI\\Desktop\\internship\\my training\\Chapter3_Regression_Logistic\\Datasets\\telco_data.csv\")\n",
    "print(telco_cust.shape)\n",
    "print(telco_cust.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "d516e488-7be9-4e71-8151-6e855928e843",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization terminated successfully.\n",
      "         Current function value: 0.327208\n",
      "         Iterations 11\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>Logit Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>      <td>Active_cust</td>   <th>  No. Observations:  </th>  <td>100000</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 99991</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     8</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>            <td>Tue, 31 Jan 2023</td> <th>  Pseudo R-squ.:     </th>  <td>0.5193</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                <td>18:36:26</td>     <th>  Log-Likelihood:    </th> <td> -32721.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -68074.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "           <td></td>              <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>estimated_income</th>    <td> 5.476e-05</td> <td> 3.15e-05</td> <td>    1.740</td> <td> 0.082</td> <td>-6.93e-06</td> <td>    0.000</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>months_on_network</th>   <td>   -2.1605</td> <td>    2.473</td> <td>   -0.874</td> <td> 0.382</td> <td>   -7.008</td> <td>    2.687</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>complaints_count</th>    <td>   31.7026</td> <td>   37.097</td> <td>    0.855</td> <td> 0.393</td> <td>  -41.006</td> <td>  104.411</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>plan_changes_count</th>  <td>   -0.5828</td> <td>    0.011</td> <td>  -52.166</td> <td> 0.000</td> <td>   -0.605</td> <td>   -0.561</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>relocated_new_place</th> <td>   -2.4047</td> <td>    0.047</td> <td>  -51.554</td> <td> 0.000</td> <td>   -2.496</td> <td>   -2.313</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>monthly_bill_avg</th>    <td>   -0.0035</td> <td>    0.000</td> <td>  -17.173</td> <td> 0.000</td> <td>   -0.004</td> <td>   -0.003</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>CSAT_Survey_Score</th>   <td>    3.3119</td> <td>    3.710</td> <td>    0.893</td> <td> 0.372</td> <td>   -3.959</td> <td>   10.583</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>high_talktime_flag</th>  <td>   -0.0354</td> <td>    0.020</td> <td>   -1.763</td> <td> 0.078</td> <td>   -0.075</td> <td>    0.004</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>internet_time</th>       <td>    0.0079</td> <td> 4.68e-05</td> <td>  168.858</td> <td> 0.000</td> <td>    0.008</td> <td>    0.008</td>\n",
       "</tr>\n",
       "</table>"
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                           Logit Regression Results                           \n",
       "==============================================================================\n",
       "Dep. Variable:            Active_cust   No. Observations:               100000\n",
       "Model:                          Logit   Df Residuals:                    99991\n",
       "Method:                           MLE   Df Model:                            8\n",
       "Date:                Tue, 31 Jan 2023   Pseudo R-squ.:                  0.5193\n",
       "Time:                        18:36:26   Log-Likelihood:                -32721.\n",
       "converged:                       True   LL-Null:                       -68074.\n",
       "Covariance Type:            nonrobust   LLR p-value:                     0.000\n",
       "=======================================================================================\n",
       "                          coef    std err          z      P>|z|      [0.025      0.975]\n",
       "---------------------------------------------------------------------------------------\n",
       "estimated_income     5.476e-05   3.15e-05      1.740      0.082   -6.93e-06       0.000\n",
       "months_on_network      -2.1605      2.473     -0.874      0.382      -7.008       2.687\n",
       "complaints_count       31.7026     37.097      0.855      0.393     -41.006     104.411\n",
       "plan_changes_count     -0.5828      0.011    -52.166      0.000      -0.605      -0.561\n",
       "relocated_new_place    -2.4047      0.047    -51.554      0.000      -2.496      -2.313\n",
       "monthly_bill_avg       -0.0035      0.000    -17.173      0.000      -0.004      -0.003\n",
       "CSAT_Survey_Score       3.3119      3.710      0.893      0.372      -3.959      10.583\n",
       "high_talktime_flag     -0.0354      0.020     -1.763      0.078      -0.075       0.004\n",
       "internet_time           0.0079   4.68e-05    168.858      0.000       0.008       0.008\n",
       "=======================================================================================\n",
       "\"\"\""
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import statsmodels.api as sm\n",
    "logit_model2=sm.Logit(telco_cust['Active_cust'],telco_cust[[\"estimated_income\"]+['months_on_network']+['complaints_count']+['plan_changes_count']+['relocated_new_place']+['monthly_bill_avg']+[\"CSAT_Survey_Score\"]+['high_talktime_flag']+['internet_time']])\n",
    "logit_fit2=logit_model2.fit()\n",
    "logit_fit2.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "b2674ba6-05ae-4d01-9e70-71e13048be3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Confuson Matrix and Accuracy\n",
    "telco_cust[\"pred_Active_cust\"]=logit_fit2.predict(telco_cust.drop([\"Id\",\"Active_cust\"],axis=1))\n",
    "telco_cust[\"pred_Active_cust\"]=round(telco_cust[\"pred_Active_cust\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "0232e62c-9cff-4b98-a7a5-0a52bcf574ae",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[35985  6156]\n",
      " [ 7443 50416]]\n"
     ]
    }
   ],
   "source": [
    "cm2 = confusion_matrix(telco_cust[\"Active_cust\"],telco_cust[\"pred_Active_cust\"])\n",
    "print(cm2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "c7b8f9aa-a656-4032-8dd7-d3d03533f068",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.86401\n"
     ]
    }
   ],
   "source": [
    "accuracy2=(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1])\n",
    "print(accuracy2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "dced7ef4-874a-4bb3-90d7-0f191cd17000",
   "metadata": {},
   "outputs": [],
   "source": [
    "import statsmodels.formula.api as sm1\n",
    "def vif_cal(x_vars):\n",
    "    xvar_names=x_vars.columns\n",
    "    for i in range(0,xvar_names.shape[0]):\n",
    "        y=x_vars[xvar_names[i]] \n",
    "        x=x_vars[xvar_names.drop(xvar_names[i])]\n",
    "        rsq=sm1.ols(formula=\"y~x\", data=x_vars).fit().rsquared  \n",
    "        vif=round(1/(1-rsq),2)\n",
    "        print (xvar_names[i], \" VIF = \" , vif)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "ec0d3350-de00-4c68-807f-2d11fa98ad27",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "estimated_income  VIF =  1.02\n",
      "months_on_network  VIF =  20991947.38\n",
      "complaints_count  VIF =  1105768.47\n",
      "plan_changes_count  VIF =  1.56\n",
      "relocated_new_place  VIF =  1.63\n",
      "monthly_bill_avg  VIF =  1.0\n",
      "CSAT_Survey_Score  VIF =  22885771.92\n",
      "high_talktime_flag  VIF =  1.0\n",
      "internet_time  VIF =  1.07\n"
     ]
    }
   ],
   "source": [
    "#Calculating VIF values using that function\n",
    "vif_cal(x_vars=telco_cust.drop([\"Id\",\"Active_cust\",\"pred_Active_cust\"], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "ee6149aa-4bee-4409-934c-8cb949c3d343",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "estimated_income  VIF =  1.02\n",
      "months_on_network  VIF =  1.03\n",
      "complaints_count  VIF =  1.02\n",
      "plan_changes_count  VIF =  1.56\n",
      "relocated_new_place  VIF =  1.63\n",
      "monthly_bill_avg  VIF =  1.0\n",
      "high_talktime_flag  VIF =  1.0\n",
      "internet_time  VIF =  1.07\n"
     ]
    }
   ],
   "source": [
    "#Drop CSAT_Survey_Score\n",
    "vif_cal(x_vars=telco_cust.drop([\"Id\",\"Active_cust\",\"pred_Active_cust\",\"CSAT_Survey_Score\"], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "258886b9-443b-44e4-bbbd-dbd4af837de7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization terminated successfully.\n",
      "         Current function value: 0.327212\n",
      "         Iterations 10\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>Logit Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>      <td>Active_cust</td>   <th>  No. Observations:  </th>  <td>100000</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 99992</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     7</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>            <td>Tue, 31 Jan 2023</td> <th>  Pseudo R-squ.:     </th>  <td>0.5193</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                <td>18:38:15</td>     <th>  Log-Likelihood:    </th> <td> -32721.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -68074.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "           <td></td>              <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>estimated_income</th>    <td> 5.457e-05</td> <td> 3.15e-05</td> <td>    1.735</td> <td> 0.083</td> <td>-7.08e-06</td> <td>    0.000</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>months_on_network</th>   <td>    0.0474</td> <td>    0.001</td> <td>   60.421</td> <td> 0.000</td> <td>    0.046</td> <td>    0.049</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>complaints_count</th>    <td>   -1.4164</td> <td>    0.024</td> <td>  -59.194</td> <td> 0.000</td> <td>   -1.463</td> <td>   -1.370</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>plan_changes_count</th>  <td>   -0.5827</td> <td>    0.011</td> <td>  -52.166</td> <td> 0.000</td> <td>   -0.605</td> <td>   -0.561</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>relocated_new_place</th> <td>   -2.4051</td> <td>    0.047</td> <td>  -51.561</td> <td> 0.000</td> <td>   -2.496</td> <td>   -2.314</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>monthly_bill_avg</th>    <td>   -0.0035</td> <td>    0.000</td> <td>  -17.172</td> <td> 0.000</td> <td>   -0.004</td> <td>   -0.003</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>high_talktime_flag</th>  <td>   -0.0354</td> <td>    0.020</td> <td>   -1.760</td> <td> 0.078</td> <td>   -0.075</td> <td>    0.004</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>internet_time</th>       <td>    0.0079</td> <td> 4.68e-05</td> <td>  168.861</td> <td> 0.000</td> <td>    0.008</td> <td>    0.008</td>\n",
       "</tr>\n",
       "</table>"
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                           Logit Regression Results                           \n",
       "==============================================================================\n",
       "Dep. Variable:            Active_cust   No. Observations:               100000\n",
       "Model:                          Logit   Df Residuals:                    99992\n",
       "Method:                           MLE   Df Model:                            7\n",
       "Date:                Tue, 31 Jan 2023   Pseudo R-squ.:                  0.5193\n",
       "Time:                        18:38:15   Log-Likelihood:                -32721.\n",
       "converged:                       True   LL-Null:                       -68074.\n",
       "Covariance Type:            nonrobust   LLR p-value:                     0.000\n",
       "=======================================================================================\n",
       "                          coef    std err          z      P>|z|      [0.025      0.975]\n",
       "---------------------------------------------------------------------------------------\n",
       "estimated_income     5.457e-05   3.15e-05      1.735      0.083   -7.08e-06       0.000\n",
       "months_on_network       0.0474      0.001     60.421      0.000       0.046       0.049\n",
       "complaints_count       -1.4164      0.024    -59.194      0.000      -1.463      -1.370\n",
       "plan_changes_count     -0.5827      0.011    -52.166      0.000      -0.605      -0.561\n",
       "relocated_new_place    -2.4051      0.047    -51.561      0.000      -2.496      -2.314\n",
       "monthly_bill_avg       -0.0035      0.000    -17.172      0.000      -0.004      -0.003\n",
       "high_talktime_flag     -0.0354      0.020     -1.760      0.078      -0.075       0.004\n",
       "internet_time           0.0079   4.68e-05    168.861      0.000       0.008       0.008\n",
       "=======================================================================================\n",
       "\"\"\""
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import statsmodels.api as sm\n",
    "logit_model3=sm.Logit(telco_cust['Active_cust'],telco_cust[[\"estimated_income\"]+['months_on_network']+['complaints_count']+['plan_changes_count']+['relocated_new_place']+['monthly_bill_avg']+['high_talktime_flag']+['internet_time']])\n",
    "logit_fit3=logit_model3.fit()\n",
    "logit_fit3.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "cfd06e12-fa2e-43bb-bb6f-71953e9c6cbb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Confuson Matrix and Accuracy\n",
    "telco_cust[\"pred_Active_cust\"]=logit_fit3.predict(telco_cust.drop([\"Id\",\"Active_cust\",\"pred_Active_cust\",\"CSAT_Survey_Score\"],axis=1))\n",
    "telco_cust[\"pred_Active_cust\"]=round(telco_cust[\"pred_Active_cust\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "df726aa8-d110-4023-aa42-1d0cab87cafd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[35983  6158]\n",
      " [ 7442 50417]]\n"
     ]
    }
   ],
   "source": [
    "cm3 = confusion_matrix(telco_cust[\"Active_cust\"],telco_cust[\"pred_Active_cust\"])\n",
    "print(cm3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "1b0285b6-4065-4d1f-8337-a188d4fb0edb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.864\n"
     ]
    }
   ],
   "source": [
    "accuracy3=(cm3[0,0]+cm3[1,1])/(cm3[0,0]+cm3[0,1]+cm3[1,0]+cm3[1,1])\n",
    "print(accuracy3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "2f863e30-ba20-44f3-b241-8df2478511e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization terminated successfully.\n",
      "         Current function value: 0.327522\n",
      "         Iterations 8\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>Logit Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>      <td>Active_cust</td>   <th>  No. Observations:  </th>  <td>100000</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 99994</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     5</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>            <td>Tue, 31 Jan 2023</td> <th>  Pseudo R-squ.:     </th>  <td>0.5189</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                <td>18:39:59</td>     <th>  Log-Likelihood:    </th> <td> -32752.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -68074.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "           <td></td>              <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>months_on_network</th>   <td>    0.0463</td> <td>    0.001</td> <td>   69.979</td> <td> 0.000</td> <td>    0.045</td> <td>    0.048</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>complaints_count</th>    <td>   -1.3804</td> <td>    0.013</td> <td> -105.503</td> <td> 0.000</td> <td>   -1.406</td> <td>   -1.355</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>plan_changes_count</th>  <td>   -0.5807</td> <td>    0.011</td> <td>  -52.132</td> <td> 0.000</td> <td>   -0.603</td> <td>   -0.559</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>relocated_new_place</th> <td>   -2.3988</td> <td>    0.046</td> <td>  -51.593</td> <td> 0.000</td> <td>   -2.490</td> <td>   -2.308</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>monthly_bill_avg</th>    <td>   -0.0034</td> <td>    0.000</td> <td>  -17.166</td> <td> 0.000</td> <td>   -0.004</td> <td>   -0.003</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>internet_time</th>       <td>    0.0079</td> <td> 4.67e-05</td> <td>  169.255</td> <td> 0.000</td> <td>    0.008</td> <td>    0.008</td>\n",
       "</tr>\n",
       "</table>"
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                           Logit Regression Results                           \n",
       "==============================================================================\n",
       "Dep. Variable:            Active_cust   No. Observations:               100000\n",
       "Model:                          Logit   Df Residuals:                    99994\n",
       "Method:                           MLE   Df Model:                            5\n",
       "Date:                Tue, 31 Jan 2023   Pseudo R-squ.:                  0.5189\n",
       "Time:                        18:39:59   Log-Likelihood:                -32752.\n",
       "converged:                       True   LL-Null:                       -68074.\n",
       "Covariance Type:            nonrobust   LLR p-value:                     0.000\n",
       "=======================================================================================\n",
       "                          coef    std err          z      P>|z|      [0.025      0.975]\n",
       "---------------------------------------------------------------------------------------\n",
       "months_on_network       0.0463      0.001     69.979      0.000       0.045       0.048\n",
       "complaints_count       -1.3804      0.013   -105.503      0.000      -1.406      -1.355\n",
       "plan_changes_count     -0.5807      0.011    -52.132      0.000      -0.603      -0.559\n",
       "relocated_new_place    -2.3988      0.046    -51.593      0.000      -2.490      -2.308\n",
       "monthly_bill_avg       -0.0034      0.000    -17.166      0.000      -0.004      -0.003\n",
       "internet_time           0.0079   4.67e-05    169.255      0.000       0.008       0.008\n",
       "=======================================================================================\n",
       "\"\"\""
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#3.12 Individual impact of the variables \n",
    "\n",
    "#Drop estimated_income and high_talktime_flag \n",
    "import statsmodels.api as sm\n",
    "logit_model4=sm.Logit(telco_cust['Active_cust'],telco_cust[['months_on_network']+['complaints_count']+['plan_changes_count']+['relocated_new_place']+['monthly_bill_avg']+['internet_time']])\n",
    "logit_fit4=logit_model4.fit()\n",
    "logit_fit4.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "383001a6-335d-4bb9-9c82-0ce5f2a8d74b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Confuson Matrix and Accuracy\n",
    "telco_cust[\"pred_Active_cust\"]=logit_fit4.predict(telco_cust.drop([\"Id\",\"Active_cust\",\"pred_Active_cust\",\"CSAT_Survey_Score\",\"estimated_income\",\"high_talktime_flag\"],axis=1))\n",
    "telco_cust[\"pred_Active_cust\"]=round(telco_cust[\"pred_Active_cust\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "af58c55e-0a80-4b8b-ac9a-f22aae119205",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[36010  6131]\n",
      " [ 7474 50385]]\n"
     ]
    }
   ],
   "source": [
    "cm4= confusion_matrix(telco_cust[\"Active_cust\"],telco_cust[\"pred_Active_cust\"])\n",
    "print(cm4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "f62887c4-e50c-45be-b1fe-666722426aea",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.86395\n"
     ]
    }
   ],
   "source": [
    "accuracy4=(cm4[0,0]+cm4[1,1])/(cm4[0,0]+cm4[0,1]+cm4[1,0]+cm4[1,1])\n",
    "print(accuracy4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d6f504f7-3da7-4bc1-ac32-b0ee0309e476",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
