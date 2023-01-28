{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ba460254-9012-4a8a-a64a-7c4ddc80d43a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d5a0f4b2-5efa-4895-8e36-720a8d3fbf46",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the wholesale customers dataset\n",
    "cust_data = pd.read_csv(r\"C:\\Users\\SREEHARI\\Desktop\\internship\\ML_DL_py_TF-master\\Chapter6_Cluster\\Datasets\\Wholesale\\Wholesale_customers_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e63346ff-9364-47cb-9a8b-3556c59d003c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(440, 9)\n",
      "['Cust_id' 'Channel' 'Region' 'Fresh' 'Milk' 'Grocery' 'Frozen'\n",
      " 'Detergents_Paper' 'Delicatessen']\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 440 entries, 0 to 439\n",
      "Data columns (total 9 columns):\n",
      " #   Column            Non-Null Count  Dtype\n",
      "---  ------            --------------  -----\n",
      " 0   Cust_id           440 non-null    int64\n",
      " 1   Channel           440 non-null    int64\n",
      " 2   Region            440 non-null    int64\n",
      " 3   Fresh             440 non-null    int64\n",
      " 4   Milk              440 non-null    int64\n",
      " 5   Grocery           440 non-null    int64\n",
      " 6   Frozen            440 non-null    int64\n",
      " 7   Detergents_Paper  440 non-null    int64\n",
      " 8   Delicatessen      440 non-null    int64\n",
      "dtypes: int64(9)\n",
      "memory usage: 31.1 KB\n"
     ]
    }
   ],
   "source": [
    "#Rows and Columns\n",
    "print(cust_data.shape)\n",
    "print(cust_data.columns.values)\n",
    "#Dataset Information\n",
    "cust_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3af173e4-2a10-4699-907f-5e47481b387c",
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
       "      <th>Cust_id</th>\n",
       "      <th>Channel</th>\n",
       "      <th>Region</th>\n",
       "      <th>Fresh</th>\n",
       "      <th>Milk</th>\n",
       "      <th>Grocery</th>\n",
       "      <th>Frozen</th>\n",
       "      <th>Detergents_Paper</th>\n",
       "      <th>Delicatessen</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>45</th>\n",
       "      <td>46</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>5181</td>\n",
       "      <td>22044</td>\n",
       "      <td>21531</td>\n",
       "      <td>1740</td>\n",
       "      <td>7353</td>\n",
       "      <td>4985</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>223</th>\n",
       "      <td>224</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2790</td>\n",
       "      <td>2527</td>\n",
       "      <td>5265</td>\n",
       "      <td>5612</td>\n",
       "      <td>788</td>\n",
       "      <td>1360</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64</th>\n",
       "      <td>65</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4760</td>\n",
       "      <td>1227</td>\n",
       "      <td>3250</td>\n",
       "      <td>3724</td>\n",
       "      <td>1247</td>\n",
       "      <td>1145</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>366</th>\n",
       "      <td>367</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>9561</td>\n",
       "      <td>2217</td>\n",
       "      <td>1664</td>\n",
       "      <td>1173</td>\n",
       "      <td>222</td>\n",
       "      <td>447</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>288</th>\n",
       "      <td>289</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>16260</td>\n",
       "      <td>594</td>\n",
       "      <td>1296</td>\n",
       "      <td>848</td>\n",
       "      <td>445</td>\n",
       "      <td>258</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Cust_id  Channel  Region  Fresh   Milk  Grocery  Frozen  \\\n",
       "45        46        2       3   5181  22044    21531    1740   \n",
       "223      224        2       1   2790   2527     5265    5612   \n",
       "64        65        1       3   4760   1227     3250    3724   \n",
       "366      367        1       3   9561   2217     1664    1173   \n",
       "288      289        1       3  16260    594     1296     848   \n",
       "\n",
       "     Detergents_Paper  Delicatessen  \n",
       "45               7353          4985  \n",
       "223               788          1360  \n",
       "64               1247          1145  \n",
       "366               222           447  \n",
       "288               445           258  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Head\n",
    "pd.set_option('display.max_columns', None) #This option displays all the columns \n",
    "cust_data.sample(n=5, random_state=77)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5a755cb5-f6ab-47fd-a683-ef6cb25b7f3a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3    316\n",
       "1     77\n",
       "2     47\n",
       "Name: Region, dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Frequency Counts\n",
    "cust_data[\"Channel\"].value_counts()\n",
    "cust_data[\"Region\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "064d4132-0cb1-4ddb-a953-9dcd8bb2e023",
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
       "      <th>Cust_id</th>\n",
       "      <th>Channel</th>\n",
       "      <th>Region</th>\n",
       "      <th>Fresh</th>\n",
       "      <th>Milk</th>\n",
       "      <th>Grocery</th>\n",
       "      <th>Frozen</th>\n",
       "      <th>Detergents_Paper</th>\n",
       "      <th>Delicatessen</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>440.00</td>\n",
       "      <td>440.00</td>\n",
       "      <td>440.00</td>\n",
       "      <td>440.00</td>\n",
       "      <td>440.00</td>\n",
       "      <td>440.00</td>\n",
       "      <td>440.00</td>\n",
       "      <td>440.00</td>\n",
       "      <td>440.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>220.50</td>\n",
       "      <td>1.32</td>\n",
       "      <td>2.54</td>\n",
       "      <td>12000.30</td>\n",
       "      <td>5796.27</td>\n",
       "      <td>7951.28</td>\n",
       "      <td>3071.93</td>\n",
       "      <td>2881.49</td>\n",
       "      <td>1524.87</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>127.16</td>\n",
       "      <td>0.47</td>\n",
       "      <td>0.77</td>\n",
       "      <td>12647.33</td>\n",
       "      <td>7380.38</td>\n",
       "      <td>9503.16</td>\n",
       "      <td>4854.67</td>\n",
       "      <td>4767.85</td>\n",
       "      <td>2820.11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>3.00</td>\n",
       "      <td>55.00</td>\n",
       "      <td>3.00</td>\n",
       "      <td>25.00</td>\n",
       "      <td>3.00</td>\n",
       "      <td>3.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>110.75</td>\n",
       "      <td>1.00</td>\n",
       "      <td>2.00</td>\n",
       "      <td>3127.75</td>\n",
       "      <td>1533.00</td>\n",
       "      <td>2153.00</td>\n",
       "      <td>742.25</td>\n",
       "      <td>256.75</td>\n",
       "      <td>408.25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>220.50</td>\n",
       "      <td>1.00</td>\n",
       "      <td>3.00</td>\n",
       "      <td>8504.00</td>\n",
       "      <td>3627.00</td>\n",
       "      <td>4755.50</td>\n",
       "      <td>1526.00</td>\n",
       "      <td>816.50</td>\n",
       "      <td>965.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>330.25</td>\n",
       "      <td>2.00</td>\n",
       "      <td>3.00</td>\n",
       "      <td>16933.75</td>\n",
       "      <td>7190.25</td>\n",
       "      <td>10655.75</td>\n",
       "      <td>3554.25</td>\n",
       "      <td>3922.00</td>\n",
       "      <td>1820.25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>440.00</td>\n",
       "      <td>2.00</td>\n",
       "      <td>3.00</td>\n",
       "      <td>112151.00</td>\n",
       "      <td>73498.00</td>\n",
       "      <td>92780.00</td>\n",
       "      <td>60869.00</td>\n",
       "      <td>40827.00</td>\n",
       "      <td>47943.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Cust_id  Channel  Region      Fresh      Milk   Grocery    Frozen  \\\n",
       "count   440.00   440.00  440.00     440.00    440.00    440.00    440.00   \n",
       "mean    220.50     1.32    2.54   12000.30   5796.27   7951.28   3071.93   \n",
       "std     127.16     0.47    0.77   12647.33   7380.38   9503.16   4854.67   \n",
       "min       1.00     1.00    1.00       3.00     55.00      3.00     25.00   \n",
       "25%     110.75     1.00    2.00    3127.75   1533.00   2153.00    742.25   \n",
       "50%     220.50     1.00    3.00    8504.00   3627.00   4755.50   1526.00   \n",
       "75%     330.25     2.00    3.00   16933.75   7190.25  10655.75   3554.25   \n",
       "max     440.00     2.00    3.00  112151.00  73498.00  92780.00  60869.00   \n",
       "\n",
       "       Detergents_Paper  Delicatessen  \n",
       "count            440.00        440.00  \n",
       "mean            2881.49       1524.87  \n",
       "std             4767.85       2820.11  \n",
       "min                3.00          3.00  \n",
       "25%              256.75        408.25  \n",
       "50%              816.50        965.50  \n",
       "75%             3922.00       1820.25  \n",
       "max            40827.00      47943.00  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Summary\n",
    "round(cust_data.describe(),2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1dc9f114-0587-49b1-8707-15a765a54e5a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Box Plots\n",
    "plt.figure(figsize=(10,10))\n",
    "plt.title(\"All the variables box plots\", size=20)\n",
    "sns.boxplot(x=\"variable\", y=\"value\", data=pd.melt(cust_data[['Fresh', 'Milk', 'Grocery','Frozen', 'Detergents_Paper', 'Delicatessen']]))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7ee0eb81-ab3a-4120-b2ec-b97777a26522",
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
       "      <th>Cust_id</th>\n",
       "      <th>Fresh</th>\n",
       "      <th>Grocery</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>45</th>\n",
       "      <td>46</td>\n",
       "      <td>5181</td>\n",
       "      <td>21531</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>223</th>\n",
       "      <td>224</td>\n",
       "      <td>2790</td>\n",
       "      <td>5265</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64</th>\n",
       "      <td>65</td>\n",
       "      <td>4760</td>\n",
       "      <td>3250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>366</th>\n",
       "      <td>367</td>\n",
       "      <td>9561</td>\n",
       "      <td>1664</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>288</th>\n",
       "      <td>289</td>\n",
       "      <td>16260</td>\n",
       "      <td>1296</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Cust_id  Fresh  Grocery\n",
       "45        46   5181    21531\n",
       "223      224   2790     5265\n",
       "64        65   4760     3250\n",
       "366      367   9561     1664\n",
       "288      289  16260     1296"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Distance Measure\n",
    "cust_data_sample=cust_data.sample(n=5, random_state=77)\n",
    "cust_data_sample[[\"Cust_id\",\"Fresh\", \"Grocery\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2a2bb258-6c73-42f7-a2c9-c6c1baa30a0b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAn4AAAJiCAYAAABZxM+AAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABOi0lEQVR4nO3debxVVf3/8deHURCZBIEABZxIzQFufjXRSBzABjWHNEv9piFkZb9v/X5qfi1TGyy/WfZNjURFMzVMcy5JUtOcLmiJMw4EiqiIgBPj+v2x98XD5dwJ77mXy349H4/zOOesvfbaa+97uLzv2nvtEyklJEmStPFr19odkCRJUssw+EmSJBWEwU+SJKkgDH6SJEkFYfCTJEkqCIOfJElSQRj8pBYQEWdFRIqI0a3dl+aQ78vdrd0PbbzKfcY2ln9HEXF8vh/Ht3ZfVDwGP20U8l+i9T2Ob+0+CiJiu4j4eUTMjIg3I2JF/vxQRJwfESNbu49SWxERV+S/34a0dl/UdnRo7Q5IzewHdZQ/1pKd0NoiIoDv5Y92wEzgOuBNYDNgZ+AbwLcj4usppV+3Vl+1Qftf4Frg363dEamtMvhpo5JSOqu1+6CyvgecBcwFjk4p3V+7QkRsAXwL6NGiPVObkVJ6A3ijtfshtWWe6lVhRMSQ/LTIFfkpx+si4rWIWF16zVBEHBgRt0fEGxGxLCKej4ifRUTPMm3uHBHXRMRLed3X89OYv4iIjnX04/CIeDgi3s1Pc14bEQObsB89IuL/RsT0iJgXEcvz7d4cEXvUsU6KiLsjok9ETIqI+Xl/n4iI/6xjnU4RcWa+/8si4sWIODciOje2r3k7w4D/BpYD48qFPoCU0msppe8CP621fs3prGER8Y2I+FdEvFd6/VdEbBsRV0bEy/nxeCV/v20dfWofERMi4v6IWJy3NzsiLq29TkR0iIivRcSDEbEk/7k9GhFfj4h2terW+xnL21hV16m5iPhOvv63G3FcN8t/PrPyfi3Nf1bXlZ4yr9Wn4RHxp/xz905E3BcRB9SzjaMj4m8RsSgi3o+IpyLiv8t9BlriMxZ1XOO3ntvunLf3Qu1tRxOuYf0wx7eO9kZGxB/zz82yiJgTERdFxIDa+wwcl799MT64rOWlpmxPxeOIn4poa+Ah4FngaqALsAQgIr5Hdrr4TeBW4DWy05DfAQ6KiD1TSjV1d87bScDNwItAd2Ab4GtkYWdFrW1/DfhcXv8e4D+ALwC7RMSuKaVljej/R4EfAvcCtwGLgC3zdsdFxGdTSn8us15P4H6yAHY9sAlwOHBZRKxOKU2pqRgRAfwBOBh4nuwUWyfgK8DHGtHHUv9J9rvm9ymlJxqqnFJaWceiXwJ7k+3z7cCqvK8fB/5Kdsr4ZuBJYDhwDHBwRIxJKVWX7FunvI39yEYgf0/28x8CHArcBzyX1+0I3AIcCDyT130f+BTwK7Kf35fL9LWuz9hFwBTgq8AZZdY7EViW16lT/vP5M/AJ4AHgUmAlMBgYDfwdmFFrtaF53VnAb4ABZJ+9OyLiiyml62ptYzLZz3secAPwFrAHcA4wJiL2L/Oz6knrfMbWZ9t/BD5N9rP+X6AjcDyw43psG5p4fMuJiM/k/Yp8H+YAI4GJZJ/lvVJKL+XVfwAcAuxC9m/jrbz8LaT6pJR8+GjzD7LwlchOJ9Z+HJ/XGVJS70dl2vhUvuwfQM9ay47Pl11QUvY/ednBZdrqBbQreX9WXncJ8LFadX+fLzuykfvaA+hTpnwQ8ArwVD3H51KgfUn5DmSB4cla9b+Y138A2KSkvDfZf9IJuLuR/Z2e1z9hPX+2V+TrvwwMrbUsgKfy5cfUWvaFvPzpWj+LH/FBWO9ca53OQN8yP7df1Tpu7YHJtX/+jfiMdSY7VTkf6Fhr2eh8vasbcUw+lte9scyydkCvOvr0s1p1q8j+OFkEdC/zeb8B6FJrnZpjckpLf8ZKtj36Q277y3n9e4FOJeU9889LUz7fH+b4Hl9S1i3/bKwC9q7Vzql5/Tvr+LcxZH3+bfko5qPVO+DDR3M8Sn7xlnvcndep+QX9KrX+w8+X35gv37GObTwKvFbyvib4HdCI/tX8h3VumWU1gfP8ZjgOF+ZtbVnm+LxT+p9PybJ78uWblZRNy8s+VaZ+zX9adzeyT0/m9ceWWTaEdYP6t2rVqfnP7ZQy6++VL/tHHdv+e758n/x9e7IRkXeBjzTQ73Z8ENI6lFneE1gN/KHW/tT5Gcvr/Cyvc1it8mtK+9pA32qC3+8bUbemT2+V/ozLHN/jan3WV1DrD6CSY/gG8HBLf8aoP/g1Zdt/retYk40Ur0/wa8rxrdm/48tsd52fKdmI+YvU+reNwc/Hejw81auNSkopGlHtn6n8KdU9yf6zOyIijiizvBPQNyI2TyktJJuVegrwp4i4nuw/k/tTSs/Xs+3qMmVz8+dejeg7ABGxV77tPYEt8r6VGsi6Mx+fS/lp6jq23xNYmr8eQRZq7itT/+7G9rOmu/lzKrNsCPD9WmVzgF+UqftwmbIR+fP0OrY9HRgF7EY2ujOcbMT0oZTSK3X2OLMdsDnZqcD/zs4OruM9slPvtdX1GQO4GPg2cBLZaT0iog/ZaeanUkr3NtAvyML0Y8DREbEVcBPZz6o6pbS8jnVmppSWlim/m+xasd2AKRHRlez04RvAt+rY72WU3+/W+ow1ddu75dv+R5n65frTGI06vvWsX+dnOaW0MiLuJfv3shvOataHYPBTEb1aR/nmZP8mageR2roBC1NKD0fE3mTXah1Ofq1XRDwD/CCldE2Zdd8qU1ZznVT7BrZL3v6hZNf/vE82avI82WjHarLThZ8kO6XYmG3Xtf0ewJsppdrXKELdx68u88kC1zoTWFJKd5MHw4jowLrXRDa03R4l26hr25D9p1/6/HI926mxef68LfV/JrqVKavzGKWUXoiIvwAHRsTW+R8Kx5P9zH7TiH6RUloVEfuSzZY+HDgvX7Q0IqYAp6eU3q612oI6mqvpa82x7EX2M+lLw/8WanurjvJKf8bWd9vlriet6zg1pLHHty5N/SxL68VZvSqiciNPAIuBRSmlaOAxZ01DKT2QUvoM2X+We5Fd+N4P+H1E7Feh/p9DdgF7VUrpkJTSt1NK30vZrWyeaaZtLAZ6R/mZyf2b2FbNLN4xH65LZX9ui/Pnuvo0oFa9t/LnxsyirlnnxgY+D0Mb2ddSF5OFq6/m708kC/JXNqJf2QZSWpRS+j8ppcFk4fREsuvTvp63X1u/OpqqOXaLaz0/2tC/hcb2tQ7N+RlrqiX5tssNftR1nBrS2ONbl6Z+lqX1YvCTPvAg0CsimjyrL6W0LKX0j5TS94Bv5sUHN2vvPrAN2cXqT5UWRnZrkVHNtI2ZZL8fyrU3uoltXUE26nJ4RJQ7PfhhPJo/j65jeU35zPz5abLwt3NEfKSBtmvq7lFHOPkwbiU7Xfef+e0+tie7VnDR+jSWUpqdUppMNtr7NuU/eyMiYrMy5aPz50fztt4GngB2jIje69OfRmrOz1hTPZpv+xNllq3vv6FGHd8G+lRaf408oNb0a2bJolX5c6POFkhg8JNKXZA//7ZcKIiITaPkPnkRsXdElDt9U/OX/7sV6CPAS8C2pX3Mb0/xfbJZjM3h8vz5hxGxScl2epPdpqbR8lOZ55Jdh3hHRJT7zxbW7xTW/WSjnKMi4vDSBfn7fchuqXJf3pdVZLdU6QJcUvt+cfl95frmdVeSzeYdAFwYEV1qbzwiBkREk495Smk1MIns+szL8uJLGrt+RAyt4w+UXmSnjN8rs6wH2anh0naqyCYVLCab3FTj52Q/r8ui/P0re0XEiNrlTdRsn7H1UDOyem5+e5+abfcAzlzPNptyfMv5E9ltpI6Ode/H+S1gGPDXlFLp9X0L8+ct16/LKiKv8ZNyKaW7IuI04MfAcxFxO9lMum7AVmSjKfcBY/NVvg0ckN/o9QWykZYdgXFkt2+YVKGuXkAWEh6NiD+SXRe3F1nouwX4bDNs4xqy26F8DpgVETeR3efscOARsvvUNcXZZKc2zwTuj4gZZJM13iQLfEPI7qsH2SSMRkkppYg4juxax+vyfj5NNoJ2CNnF/MfmQavGD8juv/dZ4NmIuDWvNxg4APi/ZKOUkJ1W3wWYAHw2IqaTXR+4Bdnp1b3IrvF8srF9LnEpWVAYCDyeUnqgCevuAtyYH8dZZLfx6Us20teRD675K3UvcGJE/AdZYK65z1w74KTSiREppcsiuwn014Dn82sS/012q5WhZIH6crLjsr6a+zPWFFcCR5H9W54VETfn2z6MbALW9mTXzDZFo49vOSmltyPiK8BU4J6ImEp2zEeSfS5fJZsQVOouss/rb/MJZm8Db6WU/reJfVeRtPa0Yh8+muNBfuuWBuoMyetd0UC9UWQ3ln2F7Fq618lmUP6c7Lq6mnoHkP3n9yTZX/TvkI0+XQhsVavNsyhzG4qm9KvWOsfnfXqHbPbljWS3+Ci7Heq5PQV13BKCbMTne2ShdhnZSOMPyUaUGn27i1ptbk8WXB8jO426giz8PZKXj2hs/8q0exXZBfAr8uffAdvXUb8D2bVwD5P9Z/kO2ezdScA2teoG2cSdu/K+LicLf/cB3wUGr+/Pkg9uIXRyE4/jILL7Ed5PFgiWkd1o+Q6yb0cp+/kim4l7E9kfJu/m6x9Yz3Y+wwc3Ml+eb+thshHc4S39GWvmz/cmZH+QvFhr2wPz+n9q5M+iyceXMrdzKVn28fxz8Xp+zP9Nds1m2dsPAf9Fdi/LZXmbLzX136WPYj0ipYauQZYkNbf8mszZZJcGDEgNjAh9iO0MIQs3U1JKx1diGxuTiNgfuBP4SUrp9EbUH4LHV22I1/hJUus4nOy06ZWVCn2qWx3X8W4O/CR/29A1eVKb5DV+ktSC8utIewPjyU4x/6T+NVQhP4+IXchu4vw62enzcWQ/m9+klMrdMFxq8yo24hcRgyPibxHxVEQ8ERGn5OU/i4inI+JfEXFjzYyxiBgSEe9FxGP545KStkZGxOMRMTsiLsxnMBIRnSPiurz8oXzIXZI2ZD8mm6X5EnBoKrkvpFrUDWTXLH6W7Dq5z5HNAB8PTGzFfkkVVbFr/CJiANl1KzPzexvNIJtlNwiYnrKvoDkPIKV0ah7abk0p7VSmrYfJvp7qQeB24MKU0h0R8TVg55TShIg4iuyX6BcqskOSJEltXMVG/FJK81NKM/PXS8lmHQ1MKd2ZPvianAfJgmCd8gDZPWXfkJDIpuEfki8+mA+++/B6YEzNaKAkSZLW1iLX+OWjebsBD9Va9BWyL7qvMTQiHiX7Op3/Tin9nWxq/bySOvP44OuWBpJ/AXc+griY7Ps136irL3369ElDhgxZ732RJElqKTNmzHgjpdS3udqrePCLiG7AH4Fvlc5ci4gzyL7G6eq8aD6wZUppYX7j0D/ld6YvN4JXc366vmWlfRhPdt0GW265JdXV1eu7O5IkSS0mIpr1OuCK3s4l/37LPwJXp5RuKCk/juzGoMfkp29J2XedLsxfzwCeB7YjG+ErPR08iOzGuuTLBudtdiD7ypw3a/cjpTQppVSVUqrq27fZQrMkSVKbUslZvQFMBp5KKf28pHwscCrwuZTSuyXlfSOiff56GNnXIb2QUpoPLI2IPfI2jyW7MzrAzcBx+evDySaNeEdqSZKkMip5qncvsq85ejwiHsvLvkv2dVadgWn5PIwHU0oTyL778eyIWAmsAiaklGpG7yaSfR1OF7KvJLojL58MXBURs8lG+o6q4P5IkiS1aYX7yraqqqrkNX6SJKktiIgZKaWq5mrPr2yTJEkqCIOfJElSQRj8JEmSCsLgJ0mSVBAGP0mSpIIw+GmjdNVVVxERRASXXnpp2TopJaZMmcLo0aPp3bs3Xbp0YejQoRx55JE8++yzLdxjSZIqr0W+q1dqSXPnzuUb3/gG3bp14+233y5b5/333+eII47g1ltvZfvtt+eLX/wim222Ga+88gp///vfefbZZ9luu+1auOeSJFWWwU8blZQS//mf/8nmm2/O5z//ec4///yy9b797W9z6623cvrpp3PuuefSrt3ag98rVqxoie5KktSiPNWrNqOhm42nlLjwwguZPn06l19+OZtuumnZes8//zyXXHIJH//4x/nhD3+4TugD6NixY7P0WZKkDYkjftrgpZSICFauTkx78lVmzFnE068u4b3lq+jSqT3D+3dn5Fa9GBRvctppp/HNb36TffbZh+nTp5dt75prrmH16tUcd9xxLFmyhFtuuYW5c+ey+eabs++++7LNNtu08B5KktQyDH7a4K1OiUn3PM/k+17kjbeXr7P8/tkLufTe2bxx9Xfo3ncA5/7wh/W298gjjwCwePFitt56axYuXLhmWUQwceJELrzwQtq3b9+8OyJJUivzVK82aHPffJeDf30/5/35mbKhr8bi+6/h3fnP0+6TX+Ooy2Yy981366z72muvAfC9732PqqoqHn/8cZYuXcpdd93F1ltvzUUXXcQ555zT7PsiSVJrM/hpgzX3zXc5/JJ/MOvlJfXWW/bKMyx+4A90//ghdB74UWa9vIQjLnmAxe+Vn6CxatUqAAYMGMCNN97ITjvtRLdu3dh33325/vrradeuHT//+c9ZvrzuoClJUltk8NMGJ6XEqtWrmXj1DBYsWVZ/3dWreOO2n9Ox90B67v3lNeWvLnmf2/41f017pXr16gXA2LFj6dKly1rLdtllF4YOHcrSpUt56qmnmmN3JEnaYHiNnzY4EcGke55vcKQPIC1/j5VvvgzAv//n0LJ1xo8fz/jx4znllFP4xS9+wfbbb8+dd95Jz549y9avCYbvvffe+u2AJEkbKIOfNigpJVauTky+78XGrdC+I912PqDsouULnmf5gufZa69RbL/9duy5554AjBkzhl/96lfMmjVrnXWWLVvGc889B8CQIUPWax8kSdpQGfy0QYkIpj35ar0TOUq169iZzcd9s+yyt+67muULnmfk/gfzy+9/Z035uHHjGDZsGH/5y1+YNm0a+++//5pl55xzDosXL+aTn/wk/fv3/3A7I0nSBsbgpw3OjDmLmrW9OQvXnuHbqVMnpkyZwgEHHMC4ceM49NBD2WqrrXjkkUe499576du3L5MmTWrWPkiStCFwcoc2OE+/2vC1fU3x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OeywwxgwYACdO3dmwIABHHDAAdx+++1r6rz00ktERJ2Po446qhX3QJKkDVeH1u6AVOPcc8/lzDPPpE+fPnzmM59hwIABvPHGGzz66KPcfffdHHTQQWvV32WXXTjkkEPWaWennXZqoR5LktS2GPy0QZg6dSpnnnkm++23HzfccAObbbbZWstXrFixzjq77rorZ511Vgv1UJKkts9TvWpRKaV1ylavXs2pp55K165dufrqq9cJfQAdO3Zsie5JkrRRc8RPLSKlRESwcnVi2pOvMmPOIp5+dQnvLV/F0jmzePHFF9lr/8+wWY+e3HbbbTz++ON06dKF3XffnT333LNsm6+88gq/+c1vWLhwIZtvvjl77rknO++8cwvvmSRJbYfBTy1idUpMuud5Jt/3Im+8vXytZUuqqwH418LE5lsN570FL661fJ999uH666+nb9++a5VPmzaNadOmrVU2evRopkyZwpZbblmBvZAkqW3zVK8qbu6b73Lwr+/nvD8/s07oA1j17lsALH30DlYsf58tvnAuB/7sz0y7/xEOPPBA7r33Xo444og19bt27cqZZ57JjBkzWLRoEYsWLeKee+7hU5/6FHfffTdjxozhnXfeaandkySpzYhy11xtzKqqqlJ1PsKkypv75rscfsk/WLBkWZ11Fv3tMpY8fANEOwYc/ws6bTEMgP7dN+Gq43Zl3z12Zd68efzjH/+o87QvwMqVKxk1ahQPPfQQv/jFLzjllFOafX8kSWpJETEjpVTVXO054qeKSCmxavVqJl49o97QB9Buk24AdOjZb03oA3h1yfv81w1PccABBwDw8MMP19tOhw4dOPHEEwG49957P0z3JUnaKBn8VBERwaR7X2DWy0sarNux9yAA2nXuts6yx19ezL/fyT6m7733XoNt1VwH6KleSZLWZfBTs0spsWLVaibf92LDlYHOg3eEdu1ZsegV0qp179f38MzHANhqq60abOvBBx8EYNiwYQ3UlCSpeAx+anYRwbQnF5SdyFFO+6492HT43qRl7/DW/deutey9Fx9lyXMz2HSz7owbNw6Ahx56iOXL1217+vTpXHDBBQB86Utf+pB7IUnSxsfbuagiZsxZ1KT6vfY9kWXzn2HJA9exbO4sOg/YjpVLXuPdZx+Adu34zMln0bNnTwBOPfVUnnjiCUaPHs2gQdlp4n/9619Mnz4dgHPOOYdPfOITzbo/kiRtDAx+qoinX2342r5S7TftSf8v/5zF/7iWd597kGWvPEO7Tl3osvXH6bHHEXTY+oMg9+Uvf5kbb7yRRx55hDvuuIMVK1bQr18/jjzySL7+9a+z9957N/fuSJK0UTD4qSLeW76qyeu077IZvcd8ld5jvrrOsneXr1zz+oQTTuCEE074UP2TJKmIvMZPFdGlU/tmba9rJ/9GkSTpwzL4qSKG9+/evO0N2KxZ25MkqYgMfqqIkVv12qDbkySpiAx+anYpJfbfoR99unVqlvb6dOvEfh/tR9G+XlCSpOZm8FOziwg6tm/HCaOGNkt7J44aRsf27YiIZmlPkqSiMvipIlJKjN9nGDsN/HDX+n1sYA++us8wR/skSWoGBj9VRETQvl07Lj5mJP27b7JebfTvvgkXHTOC9u3C0T5JkpqBwU8VNbh3V6ZO2LPJI38fG9iDqRP2ZHDvrhXqmSRJxWPwU8UN7t2Vm07ei1PHbt/ghI8+3Tpx2tjh/OnkvQx9kiQ1M++KqxbRLoKJo7fhxL2H8denFjBjziKenr+Ud5evpGunDgwfsBkjt+rFfh/tR8f27bymT5KkCjD4qUXUXKPXoV0wbqcBjNtpQNl6NYHPa/okSWp+nupVi2oo0Bn4JEmqHIOfJElSQRj8JEmSCqLFg19EDI6Iv0XEUxHxRESckpefFREvR8Rj+eOgknVOj4jZEfFMRBxYUj4yIh7Pl10YnieUJEmqU2tM7lgJfDulNDMiNgNmRMS0fNkFKaXzSytHxA7AUcCOwEeAv0bEdimlVcDFwHjgQeB2YCxwRwvthyRJUpvS4iN+KaX5KaWZ+eulwFPAwHpWORi4NqW0LKX0IjAb2D0iBgDdU0oPpGwq6JXAIZXtvSRJUtvVqtf4RcQQYDfgobzo6xHxr4i4LCJ65WUDgbklq83Lywbmr2uXS5IkqYxWC34R0Q34I/CtlNISstO2WwO7AvOB/6mpWmb1VE95uW2Nj4jqiKh+/fXXP2zXJUmS2qRWCX4R0ZEs9F2dUroBIKW0IKW0KqW0GvgtsHtefR4wuGT1QcArefmgMuXrSClNSilVpZSq+vbt27w7I0mS1Ea0xqzeACYDT6WUfl5SXvpVDocCs/LXNwNHRUTniBgKbAs8nFKaDyyNiD3yNo8FbmqRnZAkSWqDWmNW717Al4HHI+KxvOy7wNERsSvZ6dqXgJMAUkpPRMQfgCfJZgSfnM/oBZgIXAF0IZvN64xeSZKkOkTNd6MWRVVVVaqurm7tbkiSJDUoImaklKqaqz2/uUOSJKkgDH6SVAGnnnoqY8aMYfDgwXTp0oXevXuz22678YMf/ICFCxeWXSelxJQpUxg9ejS9e/emS5cuDB06lCOPPJJnn312rbpDhgwhIup9nHPOOS2xq5LaEE/1SlIFdOrUiREjRrDDDjuwxRZb8M477/Dggw9SXV3NRz7yER588EEGD/7ghgXvv/8+RxxxBLfeeivbb789++23H5ttthmvvPIKf//737nwwgv5zGc+s6b+L37xC9566611tptS4sc//jErVqzgkUceoaqq2c4QSWoFzX2qtzUmd0jSRm/JkiVssskm65SfccYZ/OhHP+LHP/4xF1100Zryb3/729x6662cfvrpnHvuubRrt/YJmRUrVqz1/lvf+lbZ7f7lL39hxYoV7LbbboY+SevwVK8kraf6zphssskmZZcfeeSRADz33HNryp5//nkuueQSPv7xj/PDH/5wndAH0LFjx0b1adKkSQCcdNJJjaovqVgc8ZOkJkopERGsXJ2Y9uSrzJiziKdfXcJ7y1fRpVN7hvfvzsiterH/Dv3o2D7W1Ae45ZZbANh5553XtHfNNdewevVqjjvuOJYsWcItt9zC3Llz2Xzzzdl3333ZZpttGtWvBQsWcMstt9CtWze++MUvNv+OS2rzDH6S1ESrU2LSPc8z+b4XeePt5essv3/2Qibf9yJ9unViy5fvYqctOrN0yRKqq6u577772HnnnTnttNPW1H/kkUcAWLx4MVtvvfVakz8igokTJ3LhhRfSvn37evt12WWXsWLFCo4//ng222yzZtpbSRsTg58kNcHcN99l4tUzmPXykgbrvvH2ch696jfc+M5ba8rGjh3LFVdcQenXR7722msAfO9732O//fbj/PPPZ8iQITz88MOcdNJJXHTRRfTt25ezzjqrzm2llLj00ksBGD9+/PrtnKSNnrN6JamR5r75Lodf8g8WLFnW5HU3b/ceE4av5Pwffp+lS5dy6623MmLECAB23313HnnkEQYNGsSzzz5Lly5d1qz3z3/+kxEjRrDpppvyxhtv0KlTp7LtT5s2jQMOOIARI0YwY8aM9dtBSRscb+AsSS0spcSq1auZePWM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\n",
      "text/plain": [
       "<Figure size 720x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Scatter plot of customers\n",
    "plt.figure(figsize=(10,10))\n",
    "plt.title(\"Fresh and Grocery spending plot\", size=20)\n",
    "plot=sns.scatterplot(x=\"Fresh\",y=\"Grocery\", data=cust_data_sample, s=500)\n",
    "for i in list(cust_data_sample.index):\n",
    "    plot.text(cust_data_sample.Fresh[i],cust_data_sample.Grocery[i],cust_data_sample.Cust_id[i],size=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c422a6fc-46a3-4567-aa36-eaf9f8678d84",
   "metadata": {},
   "outputs": [],
   "source": [
    "#############\n",
    "## Distance Matrix\n",
    "\n",
    "def distance_cal(data_frame):\n",
    "    distance_matrix=np.zeros((data_frame.shape[0],data_frame.shape[0]))\n",
    "    for i in range(0 , data_frame.shape[0]):\n",
    "        for j in range(0 , data_frame.shape[0]):\n",
    "            distance_matrix[i,j]=round(np.sqrt(sum((data_frame.iloc[i] - data_frame.iloc[j])**2)))\n",
    "    return(distance_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f0de71f7-cc7a-4501-ac74-96a2b582814a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[    0. 16441. 18286. 20344. 23069.]\n",
      " [16441.     0.  2818.  7669. 14043.]\n",
      " [18286.  2818.     0.  5056. 11665.]\n",
      " [20344.  7669.  5056.     0.  6709.]\n",
      " [23069. 14043. 11665.  6709.     0.]]\n",
      "0.0\n",
      "16441.0\n",
      "2818.0\n"
     ]
    }
   ],
   "source": [
    "distance_matrix=distance_cal(cust_data_sample[[\"Fresh\", \"Grocery\"]])\n",
    "print(distance_matrix)\n",
    "print(distance_matrix[0,0])\n",
    "print(distance_matrix[1,0])\n",
    "print(distance_matrix[2,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "93900bc4-4c7a-473d-84cc-6673e8496c54",
   "metadata": {},
   "outputs": [],
   "source": [
    "############################\n",
    "## Building clusters in Python    \n",
    "from sklearn.cluster import KMeans\n",
    "kmeans = KMeans(n_clusters=5, random_state=333) # Mention the Number of clusters\n",
    "X=cust_data.drop(['Cust_id', 'Channel', 'Region'],axis=1) # Custid is not needed\n",
    "kmeans = kmeans.fit(X) #Model building"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5c2a2e02-3f7d-4ac4-8450-08888199b5d6",
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
       "      <th>Cust_id</th>\n",
       "      <th>Channel</th>\n",
       "      <th>Region</th>\n",
       "      <th>Fresh</th>\n",
       "      <th>Milk</th>\n",
       "      <th>Grocery</th>\n",
       "      <th>Frozen</th>\n",
       "      <th>Detergents_Paper</th>\n",
       "      <th>Delicatessen</th>\n",
       "      <th>Cluster_id</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>12669</td>\n",
       "      <td>9656</td>\n",
       "      <td>7561</td>\n",
       "      <td>214</td>\n",
       "      <td>2674</td>\n",
       "      <td>1338</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>7057</td>\n",
       "      <td>9810</td>\n",
       "      <td>9568</td>\n",
       "      <td>1762</td>\n",
       "      <td>3293</td>\n",
       "      <td>1776</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>6353</td>\n",
       "      <td>8808</td>\n",
       "      <td>7684</td>\n",
       "      <td>2405</td>\n",
       "      <td>3516</td>\n",
       "      <td>7844</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>13265</td>\n",
       "      <td>1196</td>\n",
       "      <td>4221</td>\n",
       "      <td>6404</td>\n",
       "      <td>507</td>\n",
       "      <td>1788</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>22615</td>\n",
       "      <td>5410</td>\n",
       "      <td>7198</td>\n",
       "      <td>3915</td>\n",
       "      <td>1777</td>\n",
       "      <td>5185</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>9413</td>\n",
       "      <td>8259</td>\n",
       "      <td>5126</td>\n",
       "      <td>666</td>\n",
       "      <td>1795</td>\n",
       "      <td>1451</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>12126</td>\n",
       "      <td>3199</td>\n",
       "      <td>6975</td>\n",
       "      <td>480</td>\n",
       "      <td>3140</td>\n",
       "      <td>545</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>7579</td>\n",
       "      <td>4956</td>\n",
       "      <td>9426</td>\n",
       "      <td>1669</td>\n",
       "      <td>3321</td>\n",
       "      <td>2566</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>5963</td>\n",
       "      <td>3648</td>\n",
       "      <td>6192</td>\n",
       "      <td>425</td>\n",
       "      <td>1716</td>\n",
       "      <td>750</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>6006</td>\n",
       "      <td>11093</td>\n",
       "      <td>18881</td>\n",
       "      <td>1159</td>\n",
       "      <td>7425</td>\n",
       "      <td>2098</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Cust_id  Channel  Region  Fresh   Milk  Grocery  Frozen  Detergents_Paper  \\\n",
       "0        1        2       3  12669   9656     7561     214              2674   \n",
       "1        2        2       3   7057   9810     9568    1762              3293   \n",
       "2        3        2       3   6353   8808     7684    2405              3516   \n",
       "3        4        1       3  13265   1196     4221    6404               507   \n",
       "4        5        2       3  22615   5410     7198    3915              1777   \n",
       "5        6        2       3   9413   8259     5126     666              1795   \n",
       "6        7        2       3  12126   3199     6975     480              3140   \n",
       "7        8        2       3   7579   4956     9426    1669              3321   \n",
       "8        9        1       3   5963   3648     6192     425              1716   \n",
       "9       10        2       3   6006  11093    18881    1159              7425   \n",
       "\n",
       "   Delicatessen  Cluster_id  \n",
       "0          1338           0  \n",
       "1          1776           0  \n",
       "2          7844           0  \n",
       "3          1788           4  \n",
       "4          5185           4  \n",
       "5          1451           0  \n",
       "6           545           0  \n",
       "7          2566           0  \n",
       "8           750           0  \n",
       "9          2098           2  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Getting the cluster labels and attaching them to the original data\n",
    "cust_data_clusters=cust_data\n",
    "cust_data_clusters[\"Cluster_id\"]= kmeans.predict(X)\n",
    "cust_data_clusters.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "bd6eb699-dc85-4078-9087-f950700823c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Final Results\n",
    "cluster_counts=cust_data_clusters['Cluster_id'].value_counts(sort=False)\n",
    "cluster_means= cust_data_clusters.groupby(['Cluster_id']).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3d6751e8-6cd8-4dd2-9663-c683d97506e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    217\n",
      "4    107\n",
      "2     82\n",
      "1     27\n",
      "3      7\n",
      "Name: Cluster_id, dtype: int64\n",
      "            Cust_id  Channel  Region    Fresh     Milk  Grocery  Frozen  \\\n",
      "Cluster_id                                                                \n",
      "0             230.0      1.2     2.5   5834.0   3322.4   4096.3  2635.2   \n",
      "1             225.1      1.1     2.7  46916.6   7033.6   6205.3  9757.0   \n",
      "2             207.1      1.9     2.5   5057.0  12105.1  18414.1  1580.7   \n",
      "3             127.9      2.0     2.6  20031.3  38084.0  56126.1  2564.6   \n",
      "4             216.4      1.2     2.5  20490.7   3554.1   5040.1  3446.8   \n",
      "\n",
      "            Detergents_Paper  Delicatessen  \n",
      "Cluster_id                                  \n",
      "0                     1234.7         995.5  \n",
      "1                      936.4        4199.3  \n",
      "2                     8092.0        1828.7  \n",
      "3                    27644.6        2548.1  \n",
      "4                     1099.0        1623.9  \n"
     ]
    }
   ],
   "source": [
    "print(cluster_counts)\n",
    "print(round(cluster_means,1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3c30c432-1270-48f7-b2fa-a9dbee77984d",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Cluster wise Spendings box plot\n",
    "df_melt = pd.melt(cust_data_clusters.drop(['Cust_id', 'Channel', 'Region'],axis=1), \"Cluster_id\", var_name=\"Prod_type\", value_name=\"Spend\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d52c43b2-e71b-4101-bfd8-b50163d8959f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Cluster wise Spendings')"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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eedJko499lj9+c9/9u/GZCCSUgAAAAAAIGvstttuLb8753TzzTfrmmuu+dR5K1as0KJFi3TzzTerpKREt912myRp0KBBkuKTqDc0NPgTdIZi+B4AAAAAAMhK06dP169//Wtt375dkvTPf/5TmzZt0gcffKChQ4fqK1/5im644Qa9+uqrAUeamaiUAgAAAAAAabdv/vBurZiX1PV6qKSkRGvXrtUJJ5wgSRo2bJh+//vfq7q6WjfeeKNycnKUl5en+++/v8d94dPMORd0DH3CxIkT3fLly4MOAwAAAACAjLB27VodfvjhQYcBHyX6m5vZCufcxETnM3wPAAAAAAAAviMpBQAAAAAAAN+RlAIAAAAAAIDvSEoBAAAAAADAdySlAAAAAAAA4DuSUgAAAAAAAPDdgKADAAAAAAAAme/m716rbbUf9tr19hwxSv/vpz/r9Bwz01e+8hX97ne/kyQ1NDRov/320/HHH68nn3xSCxcu1Jo1a3TTTTdp1qxZGjZsmG644QZNnTpVd999tyZOnNhr8eLTSEoBAAAAAIC021b7oW4qXNdr17uruutzdtttN61evVr19fUaMmSIKisrNXr06JbjZ599ts4+++xeiwnJYfgeAAAAAADIWKWlpXrqqackSQ899JAuvvjilmPz58/Xdddd12HbpqYmhUIh3XrrrWmPMxuRlAIAAAAAABnroosu0sMPP6wdO3Zo1apVOv7447vVrqGhQZdeeqkOPfRQzZkzJ81RZieSUgAAAAAAIGMVFRWppqZGDz30kM4444xut7vmmms0btw4fe9730tjdNmNpBQAAAAAAMhoZ599tm644YY2Q/e68rnPfU5Lly7Vjh070hhZdiMpBQAAAAAAMtrXvvY13XbbbRo/fny321x55ZU644wzdMEFF6ihoSGN0WUvVt8DAAAAAABpt+eIUd1aMS+Z63VXQUGBvv3tbyfdx/XXX69t27bpsssu04MPPqicHGp7epM554KOoU+YOHGiW758edBhAAAAAACQEdauXavDDz886DDgo0R/czNb4ZybmOh8UnwAAAAAAADwHUkpAAAAAAAA+I6kFAAAAAAAAHxHUgoAAAAAAAC+IykFAAAAAAAA35GUAgAAAAAAgO8GBB0AAAAAAADIfNf9x3XaWLux166374h9dd9P7uvyvI0bN+q73/2u/v73v2vvvffWwIED9Z//+Z8699xzey0WpIakFAAAAAAASLuNtRv1wbEf9N4FV3R9inNOX/ziFxUKhfSHP/xBkvSPf/xDCxcubHNeQ0ODBgzo/RRJY2OjcnNze/26mYLhewAAAAAAICM988wzGjhwoL7xjW+07DvooIM0Y8YMzZ8/XxdccIHOOusslZSU6KOPPtIXv/hFFRUVafLkyVq1apUkafv27briiis0fvx4FRUV6U9/+pMkqaKiQieccIKOOeYYXXDBBdq+fbskacyYMbrjjjt00kkn6a677tIxxxzT0ndVVZWOPfZYH++Bvo1KKQAAAAAAkJHefPPNNkmh9l588UWtWrVKw4cP14wZM3T00Ufrscce0zPPPKOvfvWrWrlypX7wgx9ozz331BtvvCFJ2rJli6LRqObMmaPFixdrt9120w9/+EPdc889uu222yRJgwcP1vPPPy9JWrx4sVauXKkJEyboN7/5jS6//PK03+7+gkopAAAAAACQFa699lodddRROu644yRJxcXFGj58uCTp+eef12WXXSZJOvXUU1VbW6tt27Zp8eLFuvbaa1uusffee+vvf/+71qxZoxNPPFETJkxQOBzWP/7xj5ZzLrzwwpbfr7rqKv3mN79RY2OjFixYoEsuucSPm9ovUCkFAAAAAAAy0pFHHtky3E6SfvaznykajWrixImSpN12263lmHPuU+3NTM45mVmb/c45FRcX66GHHkrYb+vrfulLX9Ls2bN16qmn6thjj9WIESN6dJsyCZVSAAAAAAAgI5166qnasWOH7r///pZ9dXV1Cc+dMmWKHnzwQUnSsmXLlJ+frz322EMlJSW6775/r/K3ZcsWTZ48WS+88IKqq6tbrrlu3bqE1x08eLCmT5+ub37zm7riiit666ZlBCqlAAAAAABA2u07Yt9urZiX1PW6YGZ67LHH9N3vflc/+tGPNHLkyJY5oOrr69ucO2vWLF1xxRUqKirS0KFDFQ6HJUm33nqrrr32Wo0bN065ubm6/fbbdd5552n+/Pm6+OKLtXPnTknSnDlzdOihhyaM49JLL9Wf//xnlZSU9PBWZxZLVJ6WjSZOnOiWL18edBgAAAAAAGSEtWvX6vDDDw86jD7h7rvv1rZt2/SDH/wg6FDSKtHf3MxWOOcmJjqfSikAAAAAAIA0Offcc/XOO+/omWeeCTqUPoekFAAAAAAA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3HHH6aijjtKXvvQl1dXVSZIuv/xyfeMb39DJJ5+sQw89VE8++aQkdfs29xTD9wAAAAAAQFY45JBD1NTUpE2bNunxxx/XnnvuqVdeeUU7d+7UiSeeqJKSEt111126++67WxI0DzzwQMLzJOnll1/W6tWrdfDBB+vRRx9VTU2N3njjDW3atEmHH364vva1rykWi2nGjBl6/PHHNXLkSC1YsEDf+9739Otf/1qS1NDQoJdffrmlimvx4sW64447tHz5ct13332SpLPOOks/+9nPdOKJJ2r79u0aPHhwl7e1rq5OS5Ys0R133KFJkybp61//uiTp1ltv1a9+9SvNmDFDUrwy7Nlnn9U777yjadOmqbq6Wr/97W+7dZt7iqQUAAAAAADIGs45SVJFRYVWrVrVUs20bds2VVVVaeDAgW3O7+y8SZMmtSRnnn/+eV1wwQXKycnRqFGjNG3aNEnS22+/rdWrV6u4uFhSvAppv/32a7n+eeedJ0k69thjVVNTkzDmE088Uddff70uvfRSnXfeeSooKOjw9r3zzjuaMGGCzEznnHOOSktL9eyzz+rWW2/V1q1btX37dk2fPr3l/C9/+cvKycnR2LFjdcghh+itt97q9m3uKZJSAAAAAAAgK7z77rvKzc3VPvvsI+ec5s6d2yZBI8WHqLXW2Xm77bZbm/MScc7pyCOP1Isvvpjw+KBBgyTFJ2VvaGhIeM5NN92kL3zhC1q0aJEmT56sxYsX67Of/WzCc5vnlGrt8ssv12OPPaajjjpK8+fPb3MbzazNuWbW7dvcUySlAAAAAABA2o3I37tbK+Ylc71kbN68Wd/4xjd03XXXycw0ffp03X///Tr11FOVl5endevWafTo0dp9993bzNnU0XntnXTSSQqHwwqFQtq8ebOWLVumSy65RIcddpg2b96sF198USeccIJisZjWrVunI488ssNY28fwzjvvaPz48Ro/frxefPFFvfXWWx0mpRL5+OOPtd9++ykWi+nBBx9sE/8f//hHhUIhvffee3r33Xd12GGHdfs29xRJKQAAAAAAkHY/uffHvvdZX1+vCRMmKBaLacCAAbrssst0/fXXS5Kuuuoq1dTU6JhjjpFzTiNHjtRjjz2moqIiDRgwQEcddZQuv/xyffvb3054Xntf+tKXtGTJEo0bN06HHnqojj/+eO25554aOHCgHn30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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(20,10))\n",
    "sns.boxplot(x='Cluster_id', hue=\"Prod_type\", y=\"Spend\", data=df_melt)\n",
    "plt.title(\"Cluster wise Spendings\", size=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "6586a5ef-f018-4b09-963f-e4b3fa5a8370",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Cluster4')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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GqCzMOiS/d7MBgT3Gy+qUafiWGcDb3ikhuuKiEjCieRzkh2B3GCEahLwvLu/amYLgOfN7FLga2AUgLt/YmbPA8uBUmCCu5cFxxrPfx8BmRW8AB4HnjKzAkItyPMieD0ikgBuuHoCmzZ802jHa9vuQH73wMN17mNm/OIXv+Cpp54CoLS0lI4dO3L00Ufz6quvMmPGDJYvX87111/Prbfeyl577cW1117LkCFDuPfee+nX70dPLkgjiViSdPelQO9ayjcAw8LUuRO4s5byRcCP7me6+zaCJCsi0hg2bfiG67t+0WjHu6tg5/vsueeeLFu2jK1bt9KmTRvy8vLo1KlT5eennXYap512WqPFJPWnaelERGLAiBEjeO211wB49tlnOf/88ys/mzZtGldccUXYuuXl5WRnZ3PTTTdFPM5EoyQpIhIDzjvvPJ577jm2bdvG0qVLOfroo+tVr7S0lNGjR9O9e3fuuOOOCEeZeJQkRURiQEZGBoWFhTz77LOcfPLJ9a43btw4evbsyY033hjB6BKXkqSISIw47bTTuPbaa6t1te7Mz372M+bNm8e2bdsiGFniUpIUEYkRl1xyCTfffDPp6en1rnPppZdy8sknM2rUKK0bGQFaKktEpIq27Q6s14jUhhyvvtLS0rjyyisbfI5rrrmGTZs2ceGFF/L000+TlKT2T2OJ2FJZsUpLZWmpLJGqtFRW4omVpbJERETimpKkiIhIGEqSIiIiYShJioiIhKEkKSIiEoaSpIiISBh6TlJEpIorfnUFazesbbTjdWjXgYfue6jOfdauXcvVV1/Ne++9x3777UfLli353//9X0aOHNlocciuUZIUEali7Ya1rO67uvEOuLjuj92dM844g+zsbJ555hkA/vWvfzFjxoxq+5WWltKiReP/k11WVkZycnKjH7e5UHeriEgUvfHGG7Rs2ZLLLrussqxz585MnDiRadOmMWrUKE499VSGDx/Ot99+yxlnnEFGRgYDBgxg6dKlAGzevJmLL76Y9PR0MjIy+Otf/wrA7NmzGThwIH369GHUqFFs3rwZgC5dujB58mSOOeYY7rrrLvr06VN57hUrVtC3b98m/A3ENrUkRUSi6JNPPqmWpGpasGABS5cuZf/992fixIn07t2bl19+mTfeeIOLLrqI/Px8br/9dtq2bcvHH38MwMaNGykuLuaOO+5g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7LHHat0PYOHChSxbtoxDDjmEF198kcLCQj7++GPWrVvHEUccwSWXXEJJSQkTJ07klVdeoX379kyfPp0bb7yRJ554AoDS0lIWLlxY2cqdM2cOkydPZtGiRTz00EMAnHrqqTz88MMMGjSIzZs307p1651e65YtW5g7dy6TJ0+mf//+/M///A8AN910E48//jgTJ04EQi3nN998k5UrVzJ06FAKCgp48skn63XNuyNiSdLd1wBrgp+/N7NPgU51VDkdeM7dtwNfmVkB0N/MCoF93H0BgJk9CZxBKEmeDtwa1H8ReMjMzBNtQloRaXYq/hmbPXs2S5curWztbdq0iRUrVtCyZctq+9e1X//+/SuTxTvvvMOoUaNISkriwAMPZOjQoUBokedly5aRlZUFhFppHTt2rDz+mWeeCYTWoSwsLKw15kGDBnHNNdcwevRozjzzTNLS0sJe38qVK8nMzMTMOP300xkxYgRvvvkmN910E9999x2bN2/mxBMrOxQ555xzSEpKolu3bhx66KF89tln9b7m3dEkA3eCbtDewPvAIOAKM7sIWESotbmRUAJ9r0q1oqCsJPi5ZjnB+yoAdy81s01AO6C4xvnHEmqJ8pOf/KQxL01EpNF9+eWXJCcnc8ABB+DuTJkypVrCgFCXYlV17VeftSndnR49erBgwYJaP2/VqhVQ9zqU119/PT//+c+ZOXMmAwYMYM6cORx++OG17ltxT7KqMWPG8PLLL9OrVy+mTZtW7Ror1sGsul3fa94dEU+SZrYX8FfgKnf/j5k9CtwOePB+H3AJYLVU9zrK2cln/y1wfwx4DEKrgDT0GkQkcbRL3a9eI1IbcryGWL9+PZdddhlXXHEFZsaJJ57Io48+yvHHH09KSgpffPEFnTp1Yu+99652zy/cfjUdc8wx5Obmkp2dzfr165k/fz4XXHABhx12GOvXr2fBggUMHDiQkpISvvjiC3r06BE21poxrFy5kvT0dNLT01mwYAGfffZZ2CRZm++//56OHTtSUlLC008/XS3+F154gezsbL766iu+/PJLDjvssHpf8+6IaJI0sxRCCfJpd/8bgLuvrfL5H4FXg80i4OAq1dOA1UF5Wi3lVesUmVkLoC3wbeNfiYgkimg807h161YyMzMrHwG58MILueaaawD45S9/SWFhIX369MHdad++PS+//DIZGRm0aNGC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\n",
      "text/plain": [
       "<Figure size 504x504 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "##Cluster 0 to 4\n",
    "Cluster=df_melt[(df_melt['Cluster_id']==4)] #Change this from 0  to 4 \n",
    "plt.figure(figsize=(7,7))\n",
    "sns.boxplot(x='Cluster_id', hue=\"Prod_type\", y=\"Spend\", data=Cluster)\n",
    "plt.title(\"Cluster4\", size=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ef7d19d4-82e1-439f-9bc2-1f79f75a3f64",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Cluster 2 and 3 only')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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rT/CiO23ZlZE8Pv6XaAdwVWmvwT+r5YFDNH2NMENpV/xFmmItBjmXLx2JxAREYmMmc0lmEB/0zn3epzDEWlSSuJERMSXzOwrBHtqPwbOcvqDJi2MhlNFRMRXzOwqgvM1ryC4D99PlMBJS6SeOBER8RVvMc1wgvPvfu2ceySuAYnEiZI4ERERER9qccOpaWlprmfPnvEOQ0RERKReq1atKnHOdQ31WotL4nr27El+fjhbZomIiIjEl5l9WttrOrFBRERExIeUxImIiIj4kJI4ERERER9qcXPiRERE/KCsrIxAIMCBAw05WU38qm3btqSnp5OSkhL2NUriREREElAgEOCYY46hZ8+emFm8w5EYcs6xc+dOAoEAp5xySv0XeDScKiIikoAOHDhAly5dlMC1AGZGly5dGtzrqiROREQkQSmBazka871WEiciIiLiQ0riRERERHxISZyIiIg0SHJyMpmZmfTt25fx48ezf//+Rrc1YsSIOk9S+r//+79Gt93cKYkTERGRBklNTaWgoIB169bRunVrfve73x3x+uHDh6P2XkriaqckTkRERBrt61//OkVFRSxfvpyRI0dy1VVX0a9fPw4cOMD1119Pv379OPvss1m2bBkApaWlXHHFFWRkZHD55ZdTWlpaa9vTpk2jtLSUzMxMrr76an7yk5/wm9/8pur1e+65h5ycHJYvX87w4cMZN24cZ555JrfccgsVFRUALF68mKFDhzJgwADGjx/Pvn37YvsFaUJK4kRERKRRysvLycvLo1+/fgCsXLmSBx54gA0bNvDYY48BsHbtWl544QWys7M5cOAAjz/+OO3atWPNmjXcc889rFq1qtb2H3zwwapev+eee44bb7yR3NxcACoqKpg3bx5XX3111Xv/6le/Yu3atWzevJk///nPlJSUMGvWLJYuXcr7779PVlYWDz/8cIy/Kk1Hm/2KiIhIg1T2jkGwJ+7GG2/kH//4B4MGDararHbFihVMnjwZgDPOOIOTTz6ZTZs28dZbbzFlyhQAMjIyyMjICPt9e/bsSZcuXVi9ejXbtm3j7LPPpkuXLgAMGjSIU089FYArr7ySFStW0LZtWzZs2MCwYcMAOHToEEOHDo3K1yARKIkTERGRBqnsHaupffv2VZ8752q9PpL9777//e8zd+5c/v3vf3PDDTfU2qaZ4Zxj1KhRvPDCC41+v0SmJE5ERFq0nJwcioqK6qwTCAQASE9Pr7Ner169qnqZWrrhw4fz3HPPcd5557Fp0yY+++wzTj/99KrykSNHsm7dOtasWVNnOykpKZSVlVWdKTpu3Djuu+8+ysrKeP7556vqrVy5kk8++YSTTz6Z+fPnc9NNNzFkyBAmTpxIUVERvXr1Yv/+/QQCAU477bSY3ntT0Zw4ERGRepSWltY5AV+O9oMf/IDDhw/Tr18/Lr/8cubOnUubNm249dZb2bdvHxkZGfziF79g0KBBdbZz0003kZGRUTX3rXXr1owcOZLvfe97JCcnV9UbOnQo06ZNo2/fvpxyyimMGzeOrl27MnfuXK688koyMjIYMmQIH374YUzvuylZXd2dzVFWVparaz8aERGRmip713JycprsPTdu3EifPn2a7P38oqKiggEDBvCnP/2J3r17A7B8+XIeeughXn311ThHF5lQ33MzW+WcywpVXz1xIiIi4gsbNmygV69enH/++VUJXEumOXEiIiISd4MHD+bgwYNHlD377LNV25cAnHnmmXz88cdHXTtixAhGjBgR6xATjpI4ERERibt333033iH4joZTRURERHxISZyIiIiIDymJExEREfEhzYkTERHxgYm338m2ki+i1t4JaZ157JGH6qyTnJx8xMKCBQsW0LNnz0a/Z8+ePcnPzyctLa3Rbch/KYkTERHxgW0lX/BJ9xHRa/Dz5fVWqe14LQgeq+WcIylJg3rxoq+8iIiIhKW4uJg+ffrwgx/8gAEDBrBlyxZ++ctfcs4555CRkcH9998PwJdffsm3vvUt+vfvT9++fZk/f35VG7Nnz2bAgAH069evWZ2eEA9K4kRERCSk0tJSMjMzyczMZNy4cQB89NFHXHfddaxevZqPPvqIwsJCVq5cSUFBAatWreKtt97ir3/9KyeeeCIffPAB69at44ILLqhqMy0tjffff59bb72Vhx6qezhX6qYkTkREREKqHE4tKCjgL3/5CwAnn3wyQ4YMAWDx4sUsXryYs88+mwEDBvDhhx9SWFhIv379WLp0KT/60Y/4+9//TseOHavavPTSSwEYOHAgxcXFTX5PzYnmxImIiEjY2rdvX/W5c467776bm2+++ah6q1atYtGiRdx9992MHj2a++67D4A2bdoAwUUT5eXlTRN0M6WeOBEREWmUMWPG8NRTT7Fv3z4A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RP266g0aNKgqmVmxYgXjx48nKSmJbt26MXLkSAA++ugj1q1bx6hRo4DgMGT37t2r2r/00ksBGDhwIMXFxSFjHjZsGFOnTuXqq6/m0ksvrbOXd/PmzWRmZmJmXHLJJYwdO5Y333yTe++9l927d7Nv3z7GjBlTVf973/seSUlJ9O7dm1NPPZUPP/ww7HuOlJI4ERERCcvHH39McnIyxx9/PM45Zs+efURCA7B8+fIjntdVr3379kfUC8U5x1lnncU777wT8vU2bdoAwUUY5eXlIetMmzaNb33rWyxatIghQ4awdOlSzjjjjJB1K+fEVTdhwgQWLFhA//79mTt37hH3aGZH1DWzsO85UkriREREfKBLWqewVpQ2pL2G2LFjB7fccguTJk3CzBgzZgyPP/445513HikpKWzatIkePXpwzDHHsHfv3qrraqtX07nnnktubi7Z2dns2LGD5cuXc9VVV3H66aezY8cO3nnnHYYOHUpZWRmbNm3irLPOqjXWmjFs3ryZfv360a9fP9555x0+/PDDWpO4UPbu3Uv37t0pKyvjueeeOyL+P/3pT2RnZ/PJJ5/w8ccfc/rpp4d9z5FSEiciIuID8djTrbS0lMzMzKotRq699lqmTp0KwPe//32Ki4sZMGAAzjm6du3KggULyMjIoFWrVvTv358JEyZw2223haxX03e/+11ef/11+vbty2mnncbgwYPp2LEjrVu35qWXXmLKlCns2bOH8vJybr/99jqTuJEjR/Lggw+SmZnJ3XffzYoVK1i2bBnJycmceeaZjB07tkFfh5/+9KcMHjyYk08+mX79+h2RIJ5++ul84xvfYNu2bfzud7+jbdu2tX5tos1q675srrKyslx+fn68wxARkVok4py4eMS0ceNG+vTp02Tvlwj27dtHhw4d2LlzJ4MGDeLtt9+mW7du8Q6rVhMmTOCiiy7isssui0p7ob7nZrbKOZcVqr564kRERCQhXHTRRezevZtDhw7xk5/8JKETuESgJE5EREQSQs1FEbGydu1arr322iPK2rRpw7vvvtugdubOnRvFqBpOSZyIiIi0KP369TtqBaof6dgtERERER9SEiciIiLiQ0riRERERHxIc+JERER8YOqUyZTs2BG19tK6duXhnNl11klOTqZfv35V+8RlZ2dz++23k5RUex9QcXExF110EevWrSM/P59nnnmmUVuzPPLII9x00020a9euwde2FEriREREfKBkxw5OTw59rFRjfBRGQpiamlq1AGD79u1cddVV7NmzhxkzZoT1HllZWWRlhdzirF6PPPII11xzjZK4Omg4VUREROp1/PHHM2fOHB599FGccxw+fJi77rqLc845h4yMDJ544omjrlm+fDkXXXQRENzI9/rrr6dfv35kZGTw8ssvA3DrrbeSlZXFWWedxf333w8EN1XeunUrI0eOZOTIkQAsXryYoUOHMmDAAMaPH8++ffuA4LmoZ555JhkZGdx5551A8Cisvn370r9/f4YPHw5Qa7zLly9nxIgRXHbZZZxxxhlcffXVtZ7jmmjUEyciIiJhOfXUU6moqGD79u288sordOzYkffee4+DBw8ybNgwRo8efdSB8JV++tOf0rFjR9auXQvArl27AHjggQfo3Lkzhw8f5vzzz2fNmjVMmTKFhx9+mGXLlpGWlkZJSQmzZs1i6dKltG/fnp///Oc8/PDDTJo0ib/85S98+OGHmBm7d+8GYObMmfztb3+jR48eVWVPPvlkyHgBVq9ezfr16znxxBMZNmwYb7/9Nueee25sv5hRoCROREREwlbZS7V48WLWrFnDSy+9BMCePXsoLCzktNNOC3nd0qVLmTdvXtXzTp06AfDiiy8yZ84cysvL+fzzz9mwYQMZGRlHXPvPf/6TDRs2MGzYMAAOHTrE0KFDOfbYY6vOKv3Wt75V1es3bNgwJkyYwPe+9z0uvfTSOuNt3bo1gwYNIj09HYDMzEyKi4uVxImIiEjz8fHHH5OcnMzxxx+Pc47Zs2czZsyYI+oUFxeHvNY5d1Qv3SeffMJDDz3Ee++9R6dOnZgwYQIHDhwIee2oUaN44YUXjnpt5cqVvP7668ybN49HH32UN954g9/97ne8++67vPbaa2RmZlJQUFBrvMuXL6dNmzZVz5OTkykvj97cw1jSnDgRERGp144dO7jllluYNGkSZsaYMWN4/PHHKSsrA2DTpk18+eWXtV4/evRoHn300arnu3bt4j//+Q/t27enY8eObNu2jby8vKrXjznmGPbu3QvAkCFDePvttykqKgJg//79bNq0iX379rFnzx4uvPBCHnnkkapFGJs3b2bw4MHMnDmTtLQ0tmzZ0uB4/UA9cSIiIj6Q1rVrWCtKG9JefUpLS8nMzKzaYuTaa69l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\n",
      "text/plain": [
       "<Figure size 720x504 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "##Objective 1\n",
    "Cluster_2and3=df_melt[(df_melt['Cluster_id']==2)| (df_melt['Cluster_id']==3)]\n",
    "plt.figure(figsize=(10,7))\n",
    "sns.boxplot(x='Cluster_id', hue=\"Prod_type\", y=\"Spend\", data=Cluster_2and3)\n",
    "plt.title(\"Cluster 2 and 3 only\", size=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a152a954-155b-4aed-ba83-914f17a94d9f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[10, 11, 17, 24, 29, 38, 39, 43, 44, 46, 47, 48, 50, 54, 57, 58, 62, 64, 66, 78, 82, 83, 86, 87, 93, 95, 101, 102, 107, 108, 110, 112, 146, 156, 157, 160, 164, 166, 171, 172, 174, 176, 183, 189, 190, 194, 201, 202, 206, 210, 212, 215, 216, 217, 219, 246, 252, 265, 266, 267, 269, 294, 302, 304, 305, 306, 307, 310, 313, 316, 320, 332, 334, 344, 347, 350, 352, 354, 358, 377, 385, 397, 408, 417, 419, 421, 427, 431, 438]\n"
     ]
    }
   ],
   "source": [
    "obj1_data= cust_data_clusters[(cust_data_clusters['Cluster_id']==2)| (cust_data_clusters['Cluster_id']==3)]\n",
    "print(list(obj1_data[\"Cust_id\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "14e6bd49-e09f-4ed8-bf0e-c75500a903b7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[4, 5, 13, 14, 15, 19, 21, 23, 25, 26, 28, 31, 33, 34, 37, 41, 42, 55, 59, 68, 71, 72, 74, 76, 84, 90, 105, 106, 113, 114, 115, 119, 121, 127, 128, 133, 139, 141, 142, 145, 150, 151, 153, 158, 163, 191, 192, 196, 203, 211, 218, 221, 227, 233, 235, 238, 241, 242, 243, 248, 249, 254, 256, 263, 268, 270, 277, 280, 284, 288, 289, 295, 297, 301, 308, 312, 323, 324, 325, 329, 333, 335, 336, 337, 348, 355, 357, 361, 369, 372, 374, 381, 382, 388, 394, 402, 403, 404, 405, 407, 422, 423, 424, 425, 433, 435, 436]\n"
     ]
    }
   ],
   "source": [
    "##Objective 2\n",
    "obj2_data= cust_data_clusters[cust_data_clusters['Cluster_id']==4]\n",
    "print(list(obj2_data[\"Cust_id\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "6cdb2f67-c2a0-4171-b6df-fc22a3f96272",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[30, 40, 53, 88, 104, 125, 126, 130, 143, 177, 182, 184, 197, 240, 259, 260, 274, 283, 285, 286, 290, 326, 371, 378, 383, 428, 437]\n"
     ]
    }
   ],
   "source": [
    "##Objective 3\n",
    "obj3_data= cust_data_clusters[cust_data_clusters['Cluster_id']==1]\n",
    "print(list(obj3_data[\"Cust_id\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "08095457-e10a-42c4-9f70-f99f5dd2d114",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "53138778131.157585\n"
     ]
    }
   ],
   "source": [
    "##Inertia\n",
    "print(kmeans.inertia_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b36681b2-c8ce-4b5a-84f8-aec9c37fb00b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\SREEHARI\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1036: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=2.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       K       Inertia\n",
      "1    1.0  15759.585837\n",
      "2    2.0  11321.752961\n",
      "3    3.0   8034.216798\n",
      "4    4.0   6485.574107\n",
      "5    5.0   5313.877813\n",
      "6    6.0   4727.387665\n",
      "7    7.0   4131.258478\n",
      "8    8.0   3627.142213\n",
      "9    9.0   3281.378566\n",
      "10  10.0   3006.729241\n",
      "11  11.0   2837.224803\n",
      "12  12.0   2659.655791\n",
      "13  13.0   2455.059454\n",
      "14  14.0   2258.887159\n",
      "15  15.0   2133.564861\n"
     ]
    }
   ],
   "source": [
    "###########################\n",
    "####Elbow Method\n",
    "elbow_data=pd.DataFrame()\n",
    "for i in range(1,16):\n",
    "    kmeans_m2 = KMeans(n_clusters=i, random_state=333) # Mention the Number of clusters\n",
    "    X=cust_data.drop(['Cust_id', 'Channel', 'Region'],axis=1) # Custid is not needed\n",
    "    model= kmeans_m2.fit(X)\n",
    "    elbow_data.at[i,\"K\"]=i\n",
    "    elbow_data.at[i,\"Inertia\"]=round(model.inertia_)/10000000 #To lower the values\n",
    "print(elbow_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "4aa08c6f-e522-4400-a1af-dc20cd16e4e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Inertia')"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "##Elbow Plot\n",
    "plt.figure(figsize=(15,8))\n",
    "plt.title(\"Elbow Plot\", size=20)\n",
    "plt.plot(elbow_data[\"K\"],elbow_data[\"Inertia\"],'--bD')\n",
    "plt.xticks(elbow_data[\"K\"])\n",
    "plt.xlabel(\"K\", size=15)\n",
    "plt.ylabel(\"Inertia\", size=15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "826af7ff-c669-43b2-aa9b-7488b1be5990",
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
