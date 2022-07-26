# %% [markdown]
# # Simple usage and formatting

# %% [markdown]
# This notebook contains a simple scatterplot use case for `molplotly` and introduces the formatting options for the package.

# %% [markdown]
# ## Imports and Data Loading

# %% [markdown]
# Import pandas for data manipulation, plotly for plotting, and molplot for visualising structures!

# %%
import pandas as pd
import plotly.express as px
import molplotly

# %% [markdown]
# Let's load the ESOL dataset from [ESOL: Estimating Aqueous Solubility Directly from Molecular Structure](https://doi.org/10.1021/ci034243x) - helpfully hosted by the [deepchem](https://github.com/deepchem/deepchem) team but also included as `example.csv` in the repo.

# %%
df_esol = pd.read_csv('example.csv')
# df_esol = pd.read_csv(
#     'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv')
df_esol['y_pred'] = df_esol['ESOL predicted log solubility in mols per litre']
df_esol['y_true'] = df_esol['measured log solubility in mols per litre']


# %%
# df_esol.head()
# df_esol.tail()
df_esol.sample()


# %%
df_esol.info()

# %% [markdown]
# ## Simple Scatterplot

# %% [markdown]
# Let's make a scatter plot comparing the measured vs predicted solubilities using [`plotly`](https://plotly.com/python/)

# %%
df_esol['delY'] = df_esol["y_pred"] - df_esol["y_true"]
fig_scatter = px.scatter(df_esol,
                         x="y_true",
                         y="y_pred",
                         color='delY',
                         title='ESOL Regression (default plotly)',
                         labels={'y_pred': 'Predicted Solubility',
                                 'y_true': 'Measured Solubility',
                                 'delY': 'ΔY'},
                         width=1200,
                         height=800)

# This adds a dashed line for what a perfect model _should_ predict
y = df_esol["y_true"].values
fig_scatter.add_shape(
    type="line", line=dict(dash='dash'),
    x0=y.min(), y0=y.min(),
    x1=y.max(), y1=y.max()
)

fig_scatter.show()

# %% [markdown]
# now all we have to do is `add_molecules`!

# %%
fig_scatter.update_layout(title='ESOL Regression (with add_molecules!)')

app_scatter = molplotly.add_molecules(fig=fig_scatter,
                                      df=df_esol,
                                      smiles_col='smiles',
                                      title_col='Compound ID'
                                      )

# change the arguments here to run the dash app on an external server and/or change the size of the app!
app_scatter.run_server(mode='inline', port=8710, width=1200, height=800)


# %% [markdown]
# ## Formatting

# %% [markdown]
# Cool right? Let's explore some formatting options:

# %% [markdown]
# ### Hoverbox transparency
# 
# the transparency of the hoverbox and the drawn molecule can be controlled by the `alpha` and `mol_alpha` parameters, respecively. The default values are `0.75` and `0.7` by personal preference (those are the values in the plot above), here is an example with smaller alpha values:

# %%
fig_scatter.update_layout(title='ESOL Regression with more transparent hoverboxes')

app_scatter_alpha = molplotly.add_molecules(fig=fig_scatter,
                                      df=df_esol,
                                      smiles_col='smiles',
                                      title_col='Compound ID',
                                      alpha=0.4,
                                      mol_alpha=0.3,
                                      )

# change the arguments here to run the dash app on an external server and/or change the size of the app!
app_scatter_alpha.run_server(mode='inline', port=8720,  width=1200, height=800)


# %% [markdown]
# ### Additional captions
# 
# Apart from showing the $(x,y)$ coordinates (you can turn them off using `show_coords=False`), we can add extra values to show up in the mouse tooltip by specifying `caption_cols` - the values in these columns of `df_esol` are also shown in the hover box.
# 
# We can also apply some function transformations to the captions via `caption_transform` - in this example, rounding all our numbers to 2 decimal places.
# 
# 

# %%
fig_scatter.update_layout(
    title='ESOL Regression (with add_molecules & extra captions)')

app_scatter_with_captions = molplotly.add_molecules(fig=fig_scatter,
                                                    df=df_esol,
                                                    smiles_col='smiles',
                                                    title_col='Compound ID',
                                                    caption_cols=['Molecular Weight', 'Number of Rings'],
                                                    caption_transform={'Predicted Solubility': lambda x: f"{x:.2f}",
                                                                       'Measured Solubility': lambda x: f"{x:.2f}",
                                                                       'Molecular Weight': lambda x: f"{x:.2f}"
                                                                       },
                                                    show_coords=True)

app_scatter_with_captions.run_server(mode='inline', port=8702, height=1000)


# %% [markdown]
# ### Colors & Size
# 
# What about adding colors? Here I've made an arbitrary random split of the dataset into `train` and `test`. When plotting, this leads to two separate plotly "curves" so the condition determining the color of the points needs to be passed in to the `add_molecules` function in order for the correct SMILES to be selected for visualisation - this is done via `color_col`. Notice that the `title` for the molecules in the hover box have the same color as the data point! 
# 
# For fun I also used the `size` argument in the scatter plot to change the size of the markers in proportion to the molecular weight.
# 
# (notice I've been choosing different `port` numbers in all my plots, this is so that they don't interfere with each other!)

# %%
from sklearn.model_selection import train_test_split

train_inds, test_inds = train_test_split(df_esol.index)
df_esol['dataset'] = [
    'Train' if x in train_inds else 'Test' for x in df_esol.index]

fig_train_test = px.scatter(df_esol,
                            x="y_true",
                            y="y_pred",
                            size='Molecular Weight',
                            color='dataset',
                            title='ESOL Regression (colored by random train/test split)',
                            labels={'y_pred': 'Predicted Solubility',
                                    'y_true': 'Measured Solubility'},
                            width=1200,
                            height=800)
# fig.show()
app_train_test = molplotly.add_molecules(fig=fig_train_test,
                                         df=df_esol,
                                         smiles_col='smiles',
                                         title_col='Compound ID',
                                         color_col='dataset')

app_train_test.run_server(mode='inline', port=8703, height=1000)


# %% [markdown]
# ### Markers
# 
# In addition to colors, plotly also allows data splitting with different marker shapes by passing in the `symbol` parameter - this further complicates the number of plotly curves so the column used for determining marker shape also needs to be passed into the `add_molecules` function.

# %%
fig_train_test_marker = px.scatter(df_esol,
                            x="y_true",
                            y="y_pred",
                            symbol='Minimum Degree',
                            color='dataset',
                            title='ESOL Regression (colored by random train/test split)',
                            labels={'y_pred': 'Predicted Solubility',
                                    'y_true': 'Measured Solubility'},
                            width=1200,
                            height=800)
app_marker = molplotly.add_molecules(fig=fig_train_test_marker,
                                         df=df_esol,
                                         smiles_col='smiles',
                                         title_col='Compound ID',
                                         color_col='dataset',
                                         marker_col='Minimum Degree')

app_marker.run_server(mode='inline', port=8801,  width=1200, height=800)



