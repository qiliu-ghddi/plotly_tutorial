# %% [markdown]
# # An example notebook for looking at data with multiple structures per row

# %% [markdown]
# ## Imports and loading the data

# %%
import pandas as pd
import plotly.express as px
import numpy as np
import molplotly
from rdkit import Chem
from rdkit.Chem import AllChem

# %% [markdown]
# The most common use case for seeing multiple structures per row is reaction yield prediction. B.J. Shields et al. released a very nicely structured dataset in their 2021 paper [Bayesian reaction optimization as a tool for chemical synthesis](https://doi.org/10.1038/s41586-021-03213-y), which we will use as an example.

# %%
# df = pd.read_csv('https://raw.githubusercontent.com/b-shields/edbo/master/experiments/data/aryl_amination/experiment_index.csv',
#                  index_col=0)
df = pd.read_csv('experiment_index.csv',
                 index_col=0)
df

# %% [markdown]
# ## Data Modelling

# %% [markdown]
# To see the plotting in action, we first construct a simple regression model using Morgan fingerprints and a Random Forest model.

# %%
def constructInputArray(df, columns, encoder):
    """Construct a numpy array from the provided dataframe columns
       using the encoder function.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list[str]): Which columns to use for featurizing the data.
        encoder (function): Function that transforms data in the provided columns into features.

    Returns:
        np.ndarray: Featurized data
    """
    encodings = encoder(df, columns)
    input_list = []
    for col in columns:
        tmp_list = [encodings[x] for x in df[col]]
        tmp_stacked = np.concatenate(tmp_list)
        input_list.append(tmp_stacked)
    return np.concatenate(input_list, axis=1)

def morganFingerprintEncoder(df, columns):
    """Read the unique values in the provided columns of the df and return
       dictionary of features for each unique value.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list[str]): Which columns to use for featurizing the data.

    Returns:
        dict: Where keys are unique molecules and values are the corresponding Morgan Fingerpritns
    """
    df_slice = df[columns]
    unique_vals = np.unique(df_slice.values)
    out_dict = {}
    for val in unique_vals:
        mol = Chem.MolFromSmiles(val)
        out_dict[val] = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)).reshape(1,-1)
    return out_dict

# %% [markdown]
# Do a random split of the data.

# %%
df_train = df.sample(frac=.1)
df_test = df.drop(index=df_train.index)
smile_cols = ['Aryl_halide_SMILES', 'Additive_SMILES', 'Base_SMILES', 'Ligand_SMILES']
X_train = constructInputArray(df_train, smile_cols,
                        morganFingerprintEncoder)
Y_train = df_train['yield'].values
X_test = constructInputArray(df_test, smile_cols,
                        morganFingerprintEncoder)

# %%
from sklearn.ensemble import RandomForestRegressor

# %% [markdown]
# Train the model and get the predictions.

# %%
model = RandomForestRegressor()
model.fit(X_train, Y_train)

# %%
Y_pred = model.predict(X_test)
df_test['yield_pred'] = Y_pred

# %% [markdown]
# ## Data Plotting

# %% [markdown]
# Now we can use molplotly to see all the components corresponding to each point in the scatter plot! Select the SMILES columns you'd like to plot by choosing from the dropdown menu :)

# %%
fig_scatter = px.scatter(df_test,
                         x="yield",
                         y="yield_pred",
                         title='Regression with many smiles columns!',
                         labels={'yield': 'Measured yield',
                                 'yield_pred': 'Predicted yield'},
                         width=1200,
                         height=800)

app_scatter = molplotly.add_molecules(fig=fig_scatter,
                                      df=df_test,
                                    #   smiles_col='Base_SMILES'
                                      smiles_col=smile_cols,
                                      )

# change the arguments here to run the dash app on an external server and/or change the size of the app!
app_scatter.run_server(mode='inline', port=8752, height=1000)




