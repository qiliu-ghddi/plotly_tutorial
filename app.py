import pandas as pd
import plotly.express as px

import molplotly

# load a DataFrame with smiles
df_esol = pd.read_csv('delaney-processed.csv')
df_esol['y_pred'] = df_esol['ESOL predicted log solubility in mols per litre']
df_esol['y_true'] = df_esol['measured log solubility in mols per litre']

# generate a scatter plot
fig = px.scatter(df_esol, x="y_true", y="y_pred")

# add molecules to the plotly graph - returns a Dash app
app = molplotly.add_molecules(fig=fig,
                            df=df_esol,
                            smiles_col='smiles',
                            title_col='Compound ID',
                            )

# run Dash app inline in notebook (or in an external server)
app.run_server(mode='inline', port=8700, height=1000)