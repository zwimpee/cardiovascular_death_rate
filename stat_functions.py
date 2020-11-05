"""
* Author: Zachary Wimpee
* ----------------------*
* stat_funcs
*-----------*
* This file contains set of functions 
* used to perform inferential statistical 
* analysis for Cardiovascular Death Rate Capstone
"""

# importing modules
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------#
# Function: ecdf
#----------------------------------------#

def ecdf(data):
    """
    #---------------------------------------#
    # Basic function to get x and y values
    # for empirical cumulative distribution 
    # function.
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # data - array-like structure for which
    #        to compute ecdf
    #--------Returns------------------------#
    # x, y - points for plotting
    #        the ecdf
    #---------------------------------------#
    """
    
    # get length of input: n
    n = len(data)
    
    # sort data for x-values: x
    x = np.sort(data)
    
    # get ecdf value for each x: y
    y = np.arange(1, n+1) / n
    
    # return the sorted array
    # and corresponding ecdf value: x, y
    return x, y
#----------------------------------------#
#----------------------------------------#


#----------------------------------------#
# Function: ecdf_plot
#----------------------------------------#

def ecdf_plot(df_name,var_col,var_desc,
              ptile_array=np.array([2.5, 25, 50, 75, 97.5]),
              ptile_color='red'):
    """
    #---------------------------------------#
    # Function generates an ECDF plot for a 
    # specified column of input pandas DataFrame.
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # df_name - pandas DataFrame object
    #            
    # var_col - column name of variable in df_name 
    #           to plot ECDF
    # var_desc - description to use for axis label
    #
    # ptile_array - array-like set of percentile values
    #               to plot on top of distribution
    # ptile_color - color  to use for percentile markers
    #--------Returns------------------------#
    # None
    #---------------------------------------#
    """
    
    # store array of column values: data
    data = df_name[var_col]
    
    # calculate percentiles: ptiles
    ptiles = np.percentile(data, ptile_array)
    
    # get points to plot ECDF: x_ecdf, y_ecdf
    x_ecdf, y_ecdf = ecdf(data)
    
    # initialize Seaborn's default figure settings
    sns.set()
    
    # generate ECDF
    plt.plot(x_ecdf,y_ecdf,marker='.',linestyle='none');
    
    # set the plot title and axis labels
    plt.xlabel(var_desc);
    plt.ylabel('ECDF');
    plt.title('ECDF of '+var_desc);
    
    # plot the percentiles as solid diamonds
    plt.plot(ptiles,ptile_array/100,marker='D',
             color=ptile_color,linestyle='none');
    
    # create a legend for the plot
    plt.legend((var_desc,'percentiles'), loc='upper left');

    # display the plot
    plt.show()
#----------------------------------------#
#----------------------------------------#  
    
    
#----------------------------------------#
# Function: feat_ecdf_multi
#----------------------------------------#
    
def feat_ecdf_multi(df_incr,df_decr,feature,feature_name,loc='lower right'):
    """
    #---------------------------------------#
    # Function generates ECDF plots for data partition
    # of CDR sign change for specified feature
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # df_incr - pandas DataFrame object for CDR increase partition member
    # df_decr - pandas DataFrame object for CDR decrease partition member
    #
    # feature - column name of variable in DataFrame
    # feature_name - used for plot axis label 
    #--------Returns------------------------#
    # None
    #---------------------------------------#
    """
    
    # initialize array for percentile markers
    ptile_array=np.array([2.5, 25, 50, 75, 97.5])
    
    # get feature arrays from corresponding DataFrames: feat_incr, feat_decr
    feat_incr = df_incr[feature]
    feat_decr = df_decr[feature]
    
    # compute percentiles for feature arrays
    ptile_incr = np.percentile(feat_incr,ptile_array)
    ptile_decr = np.percentile(feat_decr,ptile_array)
    
    # get sorted arrays and ecdf values
    x_incr, y_incr = ecdf(feat_incr)
    x_decr, y_decr = ecdf(feat_decr)

    # plot ecdfs
    plt.plot(x_incr, y_incr, marker='None', linestyle='dashed',color='blue');
    plt.plot(x_decr, y_decr, marker='None', linestyle='solid',color='green');
    
    # plot percentiles
    plt.plot(ptile_incr, ptile_array/100, marker='D', color='red',
         linestyle='none');
    plt.plot(ptile_decr, ptile_array/100, marker='D', color='yellow',
         linestyle='none');


    # set title, axis labels, and legend
    plt.title('ECDF of partitioned data');
    plt.xlabel(feature_name);
    plt.ylabel('ECDF');
    plt.legend(('Increasing CDR', 'Decreasing CDR','Increasing percentiles','Decreasing percentiles'), loc=loc);
    
    #display the plot
    plt.show()
#----------------------------------------#
#----------------------------------------#     

#----------------------------------------#
# Function: perm_samp
#----------------------------------------#

def perm_samp(prtn_incr, prtn_decr):
    """
    #---------------------------------------#
    # Function takes members of a partition
    # and returns permuted partition members
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # prtn_incr - all rows with corresponding increase in CDR
    # prtn_decr - all rows with corresponding decrease in CDR
    #--------Returns------------------------#
    # incr_perm,decr_perm - permuted partition members
    #---------------------------------------#
    """    
    
     # get unpartitioned set of values: data
    data = np.concatenate((prtn_incr, prtn_decr))

    # permute data: permuted_data
    data_perm = np.random.permutation(data)

    # split the permuted data into new partition: incr_perm, decr_perm
    incr_perm = data_perm[:len(prtn_incr)]
    decr_perm = data_perm[len(prtn_incr):]

    # return the permuted partition members
    return incr_perm,decr_perm
#----------------------------------------#
#----------------------------------------#  

#----------------------------------------#
# Function: ecdf_perms
#----------------------------------------#

def ecdf_perms(df_incr, df_decr, feat):
    """
    #---------------------------------------#
    # Function generates 50 permutation replicates
    # of input partition, and plots the ECDF for 
    # replicates and original partition, for a specified
    # feature
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # df_incr - pandas DataFrame object for CDR increase partition
    # df_decr - pandas DataFrame object for CDR decrease partition
    #
    # feat - column name of variable in DataFrame
    #--------Returns------------------------#
    # None
    #---------------------------------------#
    """
    
    # get feature arrays from corresponding DataFrames: feat_incr, feat_decr
    feat_incr = df_incr[feat]
    feat_decr = df_decr[feat]
    
    # initialize figure of size (15,10)
    #plt.figure(figsize=(15,10))
    
    # generate 50 permutation replicates,
    # and plot them on the initialized figure
    for _ in range(50):
        
        # get permuted partition members
        feat_incr_perm, feat_decr_perm = perm_samp(feat_incr,feat_decr)

        # compute ECDFs
        x_incr, y_incr = ecdf(feat_incr_perm)
        x_decr, y_decr = ecdf(feat_decr_perm)

        # plot ECDFs of permuted members
        plt.plot(x_incr, y_incr, marker='.', linestyle='none',
                 color='red', alpha=0.002);
        plt.plot(x_decr, y_decr, marker='.', linestyle='none',
                 color='yellow', alpha=0.002);

    # get values and plot ECDFs from original partition members
    x_incr, y_incr = ecdf(feat_incr)
    x_decr, y_decr = ecdf(feat_decr)
    plt.plot(x_incr, y_incr, marker='None', linestyle='dashed', color='blue',label='incr data');
    plt.plot(x_decr, y_decr, marker='None', linestyle='solid', color='green',label='decr data');

    # label axes, set margin, and show plot
    plt.margins(0.02)
    plt.xlabel(feat);
    plt.ylabel('ECDF');
    plt.title('Partition Member and Permutation Replicate Feature Distributions');
    #plt.legend(('incr perm', 'decr perm','incr data','decr data'), loc='lower right');
    plt.legend(loc='lower right');
    plt.show()
#----------------------------------------#
#----------------------------------------#    
    

#----------------------------------------#
# Function: bootstrap_replicate_1d
#----------------------------------------#

def bootstrap_replicate_1d(data):
    """
    #---------------------------------------#
    # Function takes 1-D input array-like object and creates a 
    # bootstrap replicate by sampling from the input with replacement 
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # data - array-like structure for which
    #        to generate bootstrap replicate
    #--------Returns------------------------#
    # bs_sample - bootstrap replicate generated from data,
    #             has same length as data
    #---------------------------------------#
    """
    
    # sample with replacement from data
    # to get bootstrap replicate of same length: bs_sample
    bs_sample = np.random.choice(data, len(data))
    
    # return the resulting bootstrap replicate
    return bs_sample
#----------------------------------------#
#----------------------------------------#


#----------------------------------------#
# Function: draw_bs_reps_1d
#----------------------------------------#

def draw_bs_reps_1d(data, reps=2):
    """
    #---------------------------------------#
    # Function takes 1-D input array-like object and creates a 
    # set of bootstrap replicates using bootstrap_replicate_1d
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # data - array-like structure for which
    #        to generate bootstrap replicates
    #
    # reps - number of replicates to generate
    #--------Returns------------------------#
    # bs_reps - set of bootstrap replicates generated
    #           from data
    #---------------------------------------#
    """
    
    # initialize dict to store replicates: bs_reps
    bs_reps = {}

    # generate replicates
    for i in range(reps):
        bs_reps[i] = bootstrap_replicate_1d(data)
    
    # return replicate dictionary
    return bs_reps
#----------------------------------------#
#----------------------------------------#

#----------------------------------------#
# Function: bootstrap_replicate_df
#----------------------------------------#

def bootstrap_replicate_df(df):
    """
    #---------------------------------------#
    # Function takes input pandas DataFrame and creates a 
    # bootstrap replicate by sampling from the input with replacement 
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # df - pandas DataFrame for which to create replicate
    #--------Returns------------------------#
    # bs_sample - bootstrap replicate generated from df,
    #             has same shape as df
    #---------------------------------------#
    """
    
    # get length of df: N
    N = len(df)
    
    # take N samples with replacement: bs_sample
    bs_sample = df.sample(N,replace=True)
    
    # return replicate DataFrame
    return bs_sample
#----------------------------------------#
#----------------------------------------#

#----------------------------------------#
# Function: draw_bs_reps_df
#----------------------------------------#

def draw_bs_reps_df(df, reps=2):
    """
    #---------------------------------------#
    # Function takes input pandas DataFrame and creates a 
    # set of bootstrap replicates using bootstrap_replicate_df
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # df - pandas DataFrame for which to create replicates
    #
    # reps - number of replicates to generate
    #--------Returns------------------------#
    # bs_reps - dictionary of replicate DataFrames generated
    #           from df
    #---------------------------------------#
    """
    
    # initialize dict to store replicates: bs_reps
    bs_reps = {}

    # generate replicates
    for i in range(reps):
        bs_reps[i] = bootstrap_replicate_df(df)

    # return replicate DataFrames
    return bs_reps
#----------------------------------------#
#----------------------------------------#


#----------------------------------------#
# Function: draw_perm_reps
#----------------------------------------#

def draw_perm_reps(prtn_incr, prtn_decr, test_stat_func, reps=1):
    """
    #---------------------------------------#
    # Function takes set of partition members and  
    # test statistic function, and returns a distribution of  
    # replicate values of the test function for a specified number
    # of permuted partition replicates
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # prtn_incr - all rows with corresponding increase in CDR
    # prtn_decr - all rows with corresponding decrease in CDR
    #
    # test_stat_func - function that returns a test statistic value for 
    #                  each permuted partition
    # reps - number of permutations to generate
    #--------Returns------------------------#
    # perm_replicates - array of test statistic
    #                   values for the set of generated
    #                   permutation replicates
    #---------------------------------------#
    """
    
    # initialize array of replicates: perm_replicates
    perm_replicates = np.empty(reps)
    
    # generate permutation replicates
    # calculate test statistic
    # store value in perm_replicates
    for i in range(reps):
        # generate permutation replicate
        incr_perm, decr_perm = perm_samp(prtn_incr, prtn_decr)

        # compute the test statistic
        perm_replicates[i] = test_stat_func(incr_perm, decr_perm)
    
    # return array of permutation replicate test statistic values
    return perm_replicates
#----------------------------------------#
#----------------------------------------#

#----------------------------------------#
# Function: diff_of_means
#----------------------------------------#

def diff_of_means(data_1, data_2):
    """
    #---------------------------------------#
    # Function takes 2  input array-like objects,
    # and returns the magnitude of the difference
    # of their means
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # data_1 - input array-like object 
    # data_2 - input array-like object
    #--------Returns------------------------#
    # abs(diff) - the absolute value of the difference
    #             in means for the inputs
    #---------------------------------------#
    """
    
    # take the difference of the means: diff
    diff = np.mean(data_1) - np.mean(data_2)

    # return the absolute value of diff
    return abs(diff)
#----------------------------------------#
#----------------------------------------#

#----------------------------------------#
# Function: feat_p_val
#----------------------------------------#

def feat_p_val(df_incr, df_decr, feat):
    """
    #---------------------------------------#
    # Function takes 2  pandas DataFrame objects,
    # and a specified column name. Corresponding arrays
    # are defined from these inputs. A test statistic function
    # diff_of_means is calculated from these arrays, and 10,000
    # permutation replicate values of diff_of_means are generated.
    # The fraction of replicate values that are greater than the value 
    # of the original partition is returned.
    #---------------------------------------#
    #------Parameters-----------------------# 
    #
    # df_incr - pandas DataFrame object for CDR increase partition member
    # df_decr - pandas DataFrame object for CDR decrease partition member
    #
    # feat - column name of variable in DataFrame
    #--------Returns------------------------#
    # p - the fraction of permutation replicate
    #     test statistic values that are greater than
    #     or equal to the test statistic value of the 
    #     input partition
    #---------------------------------------#
    """ 
    
    # get partitions: feat_incr, feat_decr
    feat_incr = df_incr[feat]
    feat_decr = df_decr[feat]
    
    # get true difference in means: data_diff_means
    data_diff_means = diff_of_means(feat_incr, feat_decr)
    
    # get set of permutation replicates
    perm_reps = draw_perm_reps(feat_incr, feat_decr,
                                 diff_of_means, reps=10000)
    
    # Compute p-value: p
    p = np.sum(perm_reps >= data_diff_means) / len(perm_reps)
    
    # return p value
    return p
#----------------------------------------#
#----------------------------------------#





