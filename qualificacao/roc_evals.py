import pandas as pd
import numpy as np

def generate_conf_df(df, df_ref):
    confMatrix = np.zeros([2,2], dtype=int)
     
    for col, col_ref in zip(df.columns, df_ref.columns):
        for row, row_ref, index in zip(df[col], df_ref[col_ref], np.arange(0,len(df.columns))):
            if(df[col].index[index] != col):
                if row == row_ref and row == 0:
                    confMatrix[1,1] += 1
                elif row == row_ref and row == 1:
                    confMatrix[0,0] += 1
                elif row != row_ref and row == 1:
                    confMatrix[0,1] += 1
                else:
    return pd.DataFrame(confMatrix,columns=["p", "n"], index = ["Y", "N"])


#roc fp*tp

def fp_rate(conf_df):
    return conf_df["n"]["Y"]/(conf_df["n"]["Y"]+ conf_df["n"]["N"])

def tp_rate(conf_df): #since it is positive what are the odds of being classsified as positive - recall
    return conf_df['p']['Y']/(conf_df['p']['Y'] + conf_df['p']['N'])

def precision(conf_df): #from those i choose as correct, how many really were
    return conf_df['p']['Y']/(conf_df['p']['Y'] + conf_df['n']['Y'])
