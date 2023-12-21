# LOAN_ANALYSIS
This Python code utilizes the NumPy library to preprocess a loan dataset ("loan-data.csv"). The preprocessing involves handling missing values, transforming string data, creating checkpoints, converting data types, and organizing the dataset for subsequent analysis.

<img src="https://i.imgur.com/Ah4wGve.jpeg"> 

## Importing the Packages: The code starts by importing the NumPy package.


```python
import numpy as np
```


```python
np.set_printoptions(suppress = True, linewidth = 100, precision = 2)
```

## Importing the Data: Raw data is loaded from the CSV file using the NumPy np.genfromtxt function.


```python
raw_data_np = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, autostrip = True)
raw_data_np
```

## Checking for Incomplete Data: Checks the amount of missing values in the raw data and create temporary files to substitute them.


```python
np.isnan(raw_data_np).sum()
```


```python
temporary_fill = np.nanmax(raw_data_np) + 1
temporary_mean = np.nanmean(raw_data_np, axis = 0)
```


```python
temporary_mean
```


```python
temporary_stats = np.array([np.nanmin(raw_data_np, axis = 0),
                           temporary_mean,
                           np.nanmax(raw_data_np, axis = 0)])
```


```python
temporary_stats
```

## Splitting the Dataset: String and numeric datasets are separated based on the presence of missing values.

### Splitting the Columns


```python
columns_strings = np.argwhere(np.isnan(temporary_mean)).squeeze()
columns_strings
```


```python
columns_numeric = np.argwhere(np.isnan(temporary_mean) == False).squeeze()
columns_numeric
```

### Re-importing the Dataset


```python
loan_data_strings = np.genfromtxt("loan-data.csv",
                                  delimiter = ';',
                                  skip_header = 1,
                                  autostrip = True, 
                                  usecols = columns_strings,
                                  dtype = np.str)
loan_data_strings
```


```python
loan_data_numeric = np.genfromtxt("loan-data.csv",
                                  delimiter = ';',
                                  autostrip = True,
                                  skip_header = 1,
                                  usecols = columns_numeric,
                                  filling_values = temporary_fill)
loan_data_numeric
```

### The Names of the Columns


```python
header_full = np.genfromtxt("loan-data.csv",
                            delimiter = ';',
                            autostrip = True,
                            skip_footer = raw_data_np.shape[0],
                            dtype = np.str)
header_full
```


```python
header_strings, header_numeric = header_full[columns_strings], header_full[columns_numeric]
```


```python
header_strings
```


```python
header_numeric
```

## Creating Checkpoints: A checkpoint function is defined to save checkpoints of string data.


```python
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)
```


```python
checkpoint_test = checkpoint("checkpoint-test", header_strings, loan_data_strings)
```


```python
checkpoint_test['data']
```


```python
np.array_equal(checkpoint_test['data'], loan_data_strings)
```

## Manipulating String Columns:

```python
header_strings
```


```python
header_strings[0] = "issue_date"
```


```python
loan_data_strings
```

### Issue Date


```python
np.unique(loan_data_strings[:,0])
```


```python
loan_data_strings[:,0] = np.chararray.strip(loan_data_strings[:,0], "-15")
```


```python
np.unique(loan_data_strings[:,0])
```


```python
months = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
```


```python
for i in range(13):
        loan_data_strings[:,0] = np.where(loan_data_strings[:,0] == months[i],
                                          i,
                                          loan_data_strings[:,0])
```


```python
np.unique(loan_data_strings[:,0])
```

### Loan Status


```python
header_strings
```


```python
np.unique(loan_data_strings[:,1])
```


```python
np.unique(loan_data_strings[:,1]).size
```


```python
status_bad = np.array(['','Charged Off','Default','Late (31-120 days)'])
```


```python
loan_data_strings[:,1] = np.where(np.isin(loan_data_strings[:,1], status_bad),0,1)
```


```python
np.unique(loan_data_strings[:,1])
```

### Term


```python
header_strings
```


```python
np.unique(loan_data_strings[:,2])
```


```python
loan_data_strings[:,2] = np.chararray.strip(loan_data_strings[:,2], " months")
loan_data_strings[:,2]
```


```python
header_strings[2] = "term_months"
```


```python
loan_data_strings[:,2] = np.where(loan_data_strings[:,2] == '', 
                                  '60', 
                                  loan_data_strings[:,2])
loan_data_strings[:,2]
```


```python
np.unique(loan_data_strings[:,2])
```

### Grade and Subgrade


```python
header_strings
```


```python
np.unique(loan_data_strings[:,3])
```


```python
np.unique(loan_data_strings[:,4])
```

#### Filling Sub Grade


```python
for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') & (loan_data_strings[:,3] == i),
                                      i + '5',
                                      loan_data_strings[:,4])
```


```python
np.unique(loan_data_strings[:,4], return_counts = True)
```


```python
loan_data_strings[:,4] = np.where(loan_data_strings[:,4] == '',
                                  'H1',
                                  loan_data_strings[:,4])
```


```python
np.unique(loan_data_strings[:,4])
```

#### Removing Grade


```python
loan_data_strings = np.delete(loan_data_strings, 3, axis = 1)
```


```python
loan_data_strings[:,3]
```


```python
header_strings = np.delete(header_strings, 3)
```


```python
header_strings[3]
```

#### Converting Sub Grade


```python
np.unique(loan_data_strings[:,3])
```


```python
keys = list(np.unique(loan_data_strings[:,3]))                         
values = list(range(1, np.unique(loan_data_strings[:,3]).shape[0] + 1)) 
dict_sub_grade = dict(zip(keys, values))
```


```python
dict_sub_grade
```


```python
for i in np.unique(loan_data_strings[:,3]):
        loan_data_strings[:,3] = np.where(loan_data_strings[:,3] == i, 
                                          dict_sub_grade[i],
                                          loan_data_strings[:,3])
```


```python
np.unique(loan_data_strings[:,3])
```

### Verification Status


```python
header_strings
```


```python
np.unique(loan_data_strings[:,4])
```


```python
loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') | (loan_data_strings[:,4] == 'Not Verified'), 0, 1)
```


```python
np.unique(loan_data_strings[:,4])
```

### URL


```python
loan_data_strings[:,5]
```


```python
np.chararray.strip(loan_data_strings[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")
```


```python
loan_data_strings[:,5] = np.chararray.strip(loan_data_strings[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")
```


```python
header_full
```


```python
loan_data_numeric[:,0].astype(dtype = np.int32)
```


```python
loan_data_strings[:,5].astype(dtype = np.int32)
```


```python
np.array_equal(loan_data_numeric[:,0].astype(dtype = np.int32), loan_data_strings[:,5].astype(dtype = np.int32))
```


```python
loan_data_strings = np.delete(loan_data_strings, 5, axis = 1)
header_strings = np.delete(header_strings, 5)
```


```python
loan_data_strings[:,5]
```


```python
header_strings
```


```python
loan_data_numeric[:,0]
```


```python
header_numeric
```

### State Address


```python
header_strings
```


```python
header_strings[5] = "state_address"
```


```python
states_names, states_count = np.unique(loan_data_strings[:,5], return_counts = True)
states_count_sorted = np.argsort(-states_count)
states_names[states_count_sorted], states_count[states_count_sorted]
```


```python
loan_data_strings[:,5] = np.where(loan_data_strings[:,5] == '', 
                                  0, 
                                  loan_data_strings[:,5])
```


```python
states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])
```




```python
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_west), 1, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_south), 2, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_midwest), 3, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_east), 4, loan_data_strings[:,5])
```


```python
np.unique(loan_data_strings[:,5])
```

## Converting to Numbers: String data is converted to numeric types.


```python
loan_data_strings
```


```python
loan_data_strings = loan_data_strings.astype(np.int)
```


```python
loan_data_strings
```

### Checkpoint 1: Strings


```python
checkpoint_strings = checkpoint("Checkpoint-Strings", header_strings, loan_data_strings)
```


```python
checkpoint_strings["header"]
```


```python
checkpoint_strings["data"]
```


```python
np.array_equal(checkpoint_strings['data'], loan_data_strings)
```

## Manipulating Numeric Columns: Missing values in numeric data are filled with temporary statistics(An exchange rate is applied to certain columns, converting from USD to EUR).


```python
loan_data_numeric
```


```python
np.isnan(loan_data_numeric).sum()
```

### Substitute "Filler" Values


```python
header_numeric
```

#### ID


```python
temporary_fill
```


```python
np.isin(loan_data_numeric[:,0], temporary_fill)
```


```python
np.isin(loan_data_numeric[:,0], temporary_fill).sum()
```


```python
header_numeric
```

#### Temporary Stats


```python
temporary_stats[:, columns_numeric]
```

#### Funded Amount


```python
loan_data_numeric[:,2]
```


```python
loan_data_numeric[:,2] = np.where(loan_data_numeric[:,2] == temporary_fill, 
                                  temporary_stats[0, columns_numeric[2]],
                                  loan_data_numeric[:,2])
loan_data_numeric[:,2]
```


```python
temporary_stats[0,columns_numeric[3]]
```

#### Loaned Amount, Interest Rate, Total Payment, Installment


```python
header_numeric
```


```python
for i in [1,3,4,5]:
    loan_data_numeric[:,i] = np.where(loan_data_numeric[:,i] == temporary_fill,
                                      temporary_stats[2, columns_numeric[i]],
                                      loan_data_numeric[:,i])
```


```python
loan_data_numeric
```

### Currency Change

#### The Exchange Rate


```python
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter = ',', autostrip = True, skip_header = 1, usecols = 3)
EUR_USD
```


```python
loan_data_strings[:,0]
```


```python
exchange_rate = loan_data_strings[:,0]

for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i,
                             EUR_USD[i-1],
                             exchange_rate)    

exchange_rate = np.where(exchange_rate == 0,
                         np.mean(EUR_USD),
                         exchange_rate)

exchange_rate
```


```python
exchange_rate.shape
```


```python
loan_data_numeric.shape
```


```python
exchange_rate = np.reshape(exchange_rate, (10000,1))
```


```python
loan_data_numeric = np.hstack((loan_data_numeric, exchange_rate))
```


```python
header_numeric = np.concatenate((header_numeric, np.array(['exchange_rate'])))
header_numeric
```

#### From USD to EUR


```python
header_numeric
```


```python
columns_dollar = np.array([1,2,4,5])
```


```python
loan_data_numeric[:,6]
```


```python
for i in columns_dollar:
    loan_data_numeric = np.hstack((loan_data_numeric, np.reshape(loan_data_numeric[:,i] / loan_data_numeric[:,6], (10000,1))))
```


```python
loan_data_numeric.shape
```


```python
loan_data_numeric
```

#### Expanding the header


```python
header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])
```


```python
header_additional
```


```python
header_numeric = np.concatenate((header_numeric, header_additional))
```


```python
header_numeric
```


```python
header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])
```


```python
header_numeric
```


```python
columns_index_order = [0,1,7,2,8,3,4,9,5,10,6]
```


```python
header_numeric = header_numeric[columns_index_order]
```


```python
loan_data_numeric
```


```python
loan_data_numeric = loan_data_numeric[:,columns_index_order]
```

### Interest Rate


```python
header_numeric
```


```python
loan_data_numeric[:,5]
```


```python
loan_data_numeric[:,5] = loan_data_numeric[:,5]/100
```


```python
loan_data_numeric[:,5]
```

### Checkpoint 2: Numeric


```python
checkpoint_numeric = checkpoint("Checkpoint-Numeric", header_numeric, loan_data_numeric)
```


```python
checkpoint_numeric['header'], checkpoint_numeric['data']
```

## Creating the "Complete" Dataset: String and numeric datasets are combined to form the final dataset (Data is sorted based on the first column (ID)).


```python
checkpoint_strings['data'].shape
```


```python
checkpoint_numeric['data'].shape
```


```python
loan_data = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))
```


```python
loan_data
```


```python
np.isnan(loan_data).sum()
```


```python
header_full = np.concatenate((checkpoint_numeric['header'], checkpoint_strings['header']))
```

## Sorting the New Dataset


```python
loan_data = loan_data[np.argsort(loan_data[:,0])]
```


```python
loan_data
```


```python
np.argsort(loan_data[:,0])
```

## Storing the New Dataset: (The preprocessed dataset is saved in a new CSV file named "loan-data-preprocessed.csv.")


```python
loan_data = np.vstack((header_full, loan_data))
```


```python
np.savetxt("loan-data-preprocessed.csv", 
           loan_data, 
           fmt = '%s',
           delimiter = ',')
```
