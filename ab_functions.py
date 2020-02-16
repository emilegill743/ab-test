import numpy as np

def get_perm_reps(data1, data2, func, samples):

   '''Resample permutations of two datasets : data1, data2
      and calculate permutation replicates : perm_reps
      (values of test statistic for resampled datasets)
      from a given function : func'''
   
   # Combine datasets
   data = data1.append(data2)

   # Initialise empty array
   perm_reps = np.empty(samples)

   for i in range(samples):
      # Resample data
      perm_data = np.random.permutation(data)
      perm1 = perm_data[:len(data1)]
      perm2 = perm_data[len(data1):]

      # Calculate test statistic from given function
      perm_reps[i] = func(perm1, perm2)

   return(perm_reps)

def conv_diff(data1, data2):

   '''Return conversion rate differential
      between two test variants'''

   conv1 = np.sum(data1) / len(data1)
   conv2 = np.sum(data2) / len(data2)

   return (conv2 - conv1)

def permutation_test(data1, data2, test_stat_func, perm_samples=10000):
   
   '''Returns p-value for permutation test on given datasets : data1, data2
      given a function to calculate the test statistic : test_stat_func,
      number of permutation samples to be used : perm_samples
   '''
   # Calculate test statistic for observed datasets
   test_stat = test_stat_func(data1, data2)

   # Generate permutation replicates
   perm_reps = get_perm_reps(data1, data2, test_stat_func, perm_samples)

   # Calculate p-value
   p = np.sum(perm_reps >= test_stat) / len(perm_reps)

   return(p)



