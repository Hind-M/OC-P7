# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import plotly.express as px
import pandas as pd
import numpy as np

import os

import joblib

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

colors = {
    'background': '#000000',
    'text': '#7FDBFF'
}


#########################################################################
port = int(os.environ.get("PORT", 5000))
print(port)

#########################################################################



# Load data from csv
df = pd.read_csv('data/data_dashboard_orig_proba.csv')
# Load the random forest model from pkl file 
random_forest_shap = joblib.load('data/random_forest_shap.pkl')

df = df.iloc[:20000]

# Get back categorical columns values already encoded with one hot encoding
def reverse_ohe(df, col_name):
    list_col = [col for col in df.columns if col_name in col]
    df_reverse_ohe = df[list_col]
    ser_reverse_ohe = pd.DataFrame(df, columns = ['SK_ID_CURR']) 
    ser_reverse_ohe[col_name] = df_reverse_ohe.apply(lambda x: df_reverse_ohe.columns[x.argmax()].replace(col_name, "")
                                                           , axis = 1)
    return ser_reverse_ohe


# Create interpretability dataframe
yg_df = reverse_ohe(df, 'NAME_YIELD_GROUP_')
pf_df = reverse_ohe(df, 'NAME_PORTFOLIO_')
ct_df = reverse_ohe(df, 'NAME_CLIENT_TYPE_')
pt_df = reverse_ohe(df, 'NAME_PRODUCT_TYPE_')


glob_interp_df = pd.DataFrame(df, columns = ['SK_ID_CURR', 'EXT_SOURCE_3'])

glob_interp_df = glob_interp_df.round({'EXT_SOURCE_3': 2})

glob_interp_df = pd.merge(glob_interp_df, yg_df, on='SK_ID_CURR')
glob_interp_df = pd.merge(glob_interp_df, pf_df, on='SK_ID_CURR')
glob_interp_df = pd.merge(glob_interp_df, ct_df, on='SK_ID_CURR')
glob_interp_df = pd.merge(glob_interp_df, pt_df, on='SK_ID_CURR')

glob_interp_df['OBS_60_CNT_SOCIAL_CIRCLE'] = df['OBS_60_CNT_SOCIAL_CIRCLE']


# Rename columns more explicitely
glob_interp_df = glob_interp_df.rename(columns={'SK_ID_CURR': 'Client Id',
        'EXT_SOURCE_3' : 'External source score',
        'NAME_YIELD_GROUP_' : 'Grouped interest rate into small medium and high of the previous application (NAME_YIELD_GROUP)',
        'NAME_PORTFOLIO_' : 'Portfolio name of previous application',
        'NAME_CLIENT_TYPE_' : 'Client type',
        'NAME_PRODUCT_TYPE_' : 'Product type',
        'OBS_60_CNT_SOCIAL_CIRCLE' : 'Nb of observations of clients social surroundings with observable 60 DPD (days past due) default (OBS_60_CNT_SOCIAL_CIRCLE)'})
        
glob_interp_df_filtered = glob_interp_df[glob_interp_df['Client Id']==glob_interp_df['Client Id'].iloc[0]]
print(glob_interp_df_filtered.shape)


# Create options for dropdown
dd_options=[]
for val in df['SK_ID_CURR']:
    dd_options.append({'label':'{}'.format(val, val), 'value':val})

# Plot random forest features importances
feature_importance = random_forest_shap.feature_importances_
indices_fi = np.argsort(feature_importance)[::-1]
features = list(df.columns)
features = features[1:-4]
print("features")
print(len(features))
# print(features[:6])
# print(features[-6:])


# plt.xticks(range(len(features)), np.array(features)[indices], rotation=90)
# plt.xlim([-1, 5])
# labels = dict(zip([str(x) for x in range(0,5)] , [str(x) for x in list(np.array(features)[indices_fi][:5])]))
# print(labels)


labels={'index':'Features', 'value':'Features importance'}
fig_feature_imp = px.bar(feature_importance[indices_fi][:7], title="Features importance", labels=labels) #, color="r")
fig_feature_imp.layout.update(showlegend=False)
fig_feature_imp.update_xaxes(
    ticktext=[str(x) for x in list(np.array(features)[indices_fi][:7])],
    tickvals=[str(x) for x in range(0,7)],
)

# Create dataframe for client s descriptive information

# Personal information:
# CODE_GENDER (ohe)
# CNT_CHILDREN
# NAME_FAMILY_STATUS(ohe)
# NAME_HOUSING_TYPE(ohe)
cg_df = reverse_ohe(df, 'CODE_GENDER_')
nfs_df = reverse_ohe(df, 'NAME_FAMILY_STATUS_')
nht_df = reverse_ohe(df, 'NAME_HOUSING_TYPE_')

glob_pers_df = pd.DataFrame(df, columns = ['SK_ID_CURR', 'CNT_CHILDREN'])
glob_pers_df = pd.merge(glob_pers_df, cg_df, on='SK_ID_CURR')
glob_pers_df = pd.merge(glob_pers_df, nfs_df, on='SK_ID_CURR')
glob_pers_df = pd.merge(glob_pers_df, nht_df, on='SK_ID_CURR')

# Rename columns more explicitely
glob_pers_df = glob_pers_df.rename(columns={'SK_ID_CURR': 'Client Id',
        'CNT_CHILDREN' : 'Client number of children',
        'CODE_GENDER_' : 'Client gender',
        'NAME_FAMILY_STATUS_' : 'Client family status',
        'NAME_HOUSING_TYPE_' : 'Client housing situation'})

data_pers_df = glob_pers_df[glob_pers_df['Client Id']==glob_pers_df['Client Id'].iloc[0]]
        
# Social and educational status:
# NAME_EDUCATION_TYPE(ohe)
# OCCUPATION_TYPE(ohe)
# NAME_INCOME_TYPE(ohe)
# AMT_INCOME_TOTAL

net_df = reverse_ohe(df, 'NAME_EDUCATION_TYPE_')
ot_df = reverse_ohe(df, 'OCCUPATION_TYPE_')
nit_df = reverse_ohe(df, 'NAME_INCOME_TYPE_')

glob_soc_df = pd.DataFrame(df, columns = ['SK_ID_CURR', 'AMT_INCOME_TOTAL'])
glob_soc_df = pd.merge(glob_soc_df, net_df, on='SK_ID_CURR')
glob_soc_df = pd.merge(glob_soc_df, ot_df, on='SK_ID_CURR')
glob_soc_df = pd.merge(glob_soc_df, nit_df, on='SK_ID_CURR')

# Rename columns more explicitely
glob_soc_df = glob_soc_df.rename(columns={'SK_ID_CURR': 'Client Id',
        'AMT_INCOME_TOTAL' : 'Client income',
        'NAME_EDUCATION_TYPE_' : 'Highest education level achieved of the client',
        'OCCUPATION_TYPE_' : 'Client occupation',
        'NAME_INCOME_TYPE_' : 'Client income type'})

# Geographical information:
# REGION_POPULATION_RELATIVE
# REGION_RATING_CLIENT

glob_geo_df = pd.DataFrame(df, columns = ['SK_ID_CURR', 'REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT'])

# Rename columns more explicitely
glob_geo_df = glob_geo_df.rename(columns={'SK_ID_CURR': 'Client Id',
        'REGION_POPULATION_RELATIVE' : 'Normalized population of region where client lives (higher number means the client lives in more populated region)',
        'REGION_RATING_CLIENT' : 'Our rating of the region where client lives (1,2,3)'})


# Create figures for comparison between clients

# CODE_GENDER
# CNT_CHILDREN
# NAME_FAMILY_STATUS
# NAME_HOUSING_TYPE
# NAME_EDUCATION_TYPE
# OCCUPATION_TYPE
# NAME_INCOME_TYPE
# AMT_INCOME_TOTAL
# REGION_POPULATION_RELATIVE
# REGION_RATING_CLIENT
# labels={'index':'Features', 'value':'Features importance'}
# fig_feature_imp = px.bar(feature_importance[indices_fi][:7], title="Features importance", labels=labels) #, color="r")
# fig_feature_imp.layout.update(showlegend=False)
# fig_feature_imp.update_xaxes(
    # ticktext=[str(x) for x in list(np.array(features)[indices_fi][:7])],
    # tickvals=[str(x) for x in range(0,7)],
# )
labels_comp = glob_pers_df['Client gender'].value_counts().index
values_comp = glob_pers_df['Client gender'].value_counts().values
names_comp = glob_pers_df['Client gender'].unique()
fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Gender distribution of all clients')

fig_desc_comp_sim = fig_desc_comp

# Display the final layout
app.layout = html.Div(children=[
   
    html.H1(
        children='Loan attribution scoring',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
       
    # html.Label('Client Id', style={'fontWeight': 'bold'}),
    # dcc.Dropdown(id='dd_select',
        # options=dd_options,
        # value=df['SK_ID_CURR'].iloc[0],
        # style={
            # 'height': '30px', 
            # 'width': '150px',
        # }
    # ),
    
    # html.Div(children=[dcc.Markdown('''
    # The score of the client number **354482**
    # has a *loan reimbursement* score of:
    # ''', id='parag_interp'),
    # html.Div(round(df['pred_proba_0'].iloc[0], 2), style={'color': 'green'}, id='score_interp')],
    # style={'marginTop': 10,
    # 'width': '200px',}),
        
    # style={'font-style': 'italic'}
    html.Div(children=[
    
        html.Div(children=[
            html.Label('Client Id', style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='dd_select',
                options=dd_options,
                value=df['SK_ID_CURR'].iloc[0],
                style={
                    'height': '30px', 
                    'width': '150px',
                }),
            html.Div(children=[dcc.Markdown('''
                The score of the client number **354482**
                has a *loan reimbursement* score of:
                ''', id='parag_interp'),
                html.Div(round(df['pred_proba_0'].iloc[0], 2), style={'color': 'green'}, id='score_interp')],
                style={
                    'marginTop': 10,
                    'width': '200px'
                })
            ], style={'display': 'inline-block', 'width' : '350px', 'height' : '300px'} # 'marginLeft':50
        ),
        html.Div(children=[
            dcc.Graph(
                id='fig_feat_imp',
                figure=fig_feature_imp
            )], style={'display': 'inline-block', 'width' : '1150px', 'height' : '300px'}
        )
    ]),
    
    html.Div(children=[
        html.H5(children='Interpretability of client score'),
        dash_table.DataTable(
            id='table_interp',
            columns=[{"name": i, "id": i} for i in glob_interp_df_filtered.columns],
            data=glob_interp_df_filtered.to_dict("rows"),
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'whiteSpace': 'normal', 'height': 'auto'
            },
            style_cell={'textAlign': 'left'}
    )]),
    
    html.Div(children=[
        html.H5(children='Descriptive information of client', style={'marginTop': 20}),
        dcc.Dropdown(id='dd_desc',
                options=[
                {'label': 'Personal information', 'value': 'PI'},
                {'label': 'Social and educational status', 'value': 'SES'},
                {'label': 'Geographical information', 'value': 'GI'}],
                value='PI',
                style={
                    'height': '30px', 
                    'width': '300px',
                    'marginTop': 20,
                    'marginBottom': 20
                }
        ),
        dash_table.DataTable(
            id='table_desc',
            columns=[{"name": i, "id": i} for i in data_pers_df.columns],
            data=data_pers_df.to_dict("rows"),
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'whiteSpace': 'normal', 'height': 'auto'
            },
            style_cell={'textAlign': 'left'}
    )]),

    html.Div(children=[
        html.H5(children='Descriptive information comparison', style={'marginTop': 20}),
        dcc.Dropdown(id='dd_desc_comp',
                options=[
                {'label': 'Gender', 'value': 'G'},
                {'label': 'Number of children', 'value': 'NoC'},
                {'label': 'Family status', 'value': 'FS'},
                {'label': 'Housing type', 'value': 'HT'},
                {'label': 'Education', 'value': 'ET'},
                {'label': 'Occupation', 'value': 'OT'},
                {'label': 'Income type', 'value': 'IT'},
                {'label': 'Income amount', 'value': 'IA'},
                {'label': 'Region normalized population', 'value': 'RNP'},
                {'label': 'Region rating', 'value': 'RR'}],
                value='G',
                style={
                    'height': '30px', 
                    'width': '300px',
                    'marginTop': 20,
                    'marginBottom': 20
                }
    )]),
    
    html.Div(children=[
        dcc.Graph(
            id='fig_desc',
            figure=fig_desc_comp
        )], style={'display': 'inline-block', 'width' : '750px', 'height' : '300px'}
    ),
    html.Div(children=[
        dcc.Graph(
            id='fig_desc_sim',
            figure=fig_desc_comp_sim
        )], style={'display': 'inline-block', 'width' : '750px', 'height' : '300px'}
    ),
])


# Define callback for interpretability section
@app.callback(
    [dash.dependencies.Output('parag_interp', 'children'),
    dash.dependencies.Output('score_interp', 'children'),
    dash.dependencies.Output('score_interp', 'style'),
    dash.dependencies.Output('table_interp', 'data')],
    [dash.dependencies.Input('dd_select', 'value')])
def update_interp_output(value):
    score = 0
    style = {}
    text_interp = ''
    client_pred = df[df['SK_ID_CURR']==value]['predictions'].iloc[0]
    client_proba_0 = df[df['SK_ID_CURR']==value]['pred_proba_0'].iloc[0]
    client_proba_1 = df[df['SK_ID_CURR']==value]['pred_proba_1'].iloc[0]
    data = glob_interp_df[glob_interp_df['Client Id']==value].to_dict("rows")

    
    if(client_pred==0):
        text_interp = 'The score of the client number **{}** has a *loan reimbursement* score of:'.format(value)
        score = round(client_proba_0, 2)
        style={'color': 'green'}
    else:
        text_interp = 'The score of the client number **{}** has a *loan reimbursement FAILURE* score of:'.format(value)
        score = round(client_proba_1, 2)
        style={'color': 'red'}
    
    return text_interp, score, style, data
    
    
# Define callback for descriptive info section
@app.callback(
    [dash.dependencies.Output('table_desc', 'data'),
    dash.dependencies.Output('table_desc', 'columns'),],
    [dash.dependencies.Input('dd_select', 'value'),
    dash.dependencies.Input('dd_desc', 'value')])
def update_desc_info_output(client_id_value, desc_info_value):
    
    data_df = glob_pers_df[glob_pers_df['Client Id']==client_id_value]
    data = data_df.to_dict("rows")
    columns = [{"name": i, "id": i} for i in data_df.columns]
    
    if (desc_info_value=='PI'):
        data_df = glob_pers_df[glob_pers_df['Client Id']==client_id_value]
        data = data_df.to_dict("rows")
        columns = [{"name": i, "id": i} for i in data_df.columns]
    
    elif (desc_info_value=='SES'):
        data_df = glob_soc_df[glob_soc_df['Client Id']==client_id_value]
        data = data_df.to_dict("rows")
        columns = [{"name": i, "id": i} for i in data_df.columns]
    
    elif (desc_info_value=='GI'):
        data_df = glob_geo_df[glob_geo_df['Client Id']==client_id_value]
        data = data_df.to_dict("rows")
        columns = [{"name": i, "id": i} for i in data_df.columns]
    return data, columns

# Define callback for comparative plots section
@app.callback(
    [dash.dependencies.Output('fig_desc', 'figure'),
    dash.dependencies.Output('fig_desc_sim', 'figure'),],
    [dash.dependencies.Input('dd_select', 'value'),
    dash.dependencies.Input('dd_desc_comp', 'value')])
def update_desc_plot_comp(client_id_value, desc_value):
    labels_comp = glob_pers_df['Client gender'].value_counts().index
    values_comp = glob_pers_df['Client gender'].value_counts().values
    names_comp = glob_pers_df['Client gender'].unique()
    
    fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Gender distribution of all clients')
    fig_desc_comp_sim = fig_desc_comp
    
    # Get similar scoring group of clients
    client_pred_proba_0 = df[df['SK_ID_CURR']==client_id_value]['pred_proba_0'].iloc[0]
    client_pred_proba_0_min = max(0, client_pred_proba_0-0.1)
    client_pred_proba_0_max = min(1, client_pred_proba_0+0.1)
    
    df_filtered = df[(df['pred_proba_0']>=client_pred_proba_0_min)&(df['pred_proba_0']<=client_pred_proba_0_max)]
    
    #
    if (desc_value =='G'):
        # Compare to all clients
        labels_comp = glob_pers_df['Client gender'].value_counts().index
        values_comp = glob_pers_df['Client gender'].value_counts().values
        names_comp = glob_pers_df['Client gender'].unique()
        
        fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Gender distribution of all clients')
        
        # Compare to similar clients
        df_reduced = glob_pers_df[glob_pers_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        labels_sim = df_reduced['Client gender'].value_counts().index
        values_sim = df_reduced['Client gender'].value_counts().values
        names_sim = df_reduced['Client gender'].unique()
        
        fig_desc_comp_sim = px.pie(values=values_sim, labels=labels_sim, names=names_sim, title = 'Gender distribution of similar scoring clients')

    elif (desc_value =='NoC'):
        # Compare to all clients
        labels_comp = glob_pers_df['Client number of children'].value_counts().index
        values_comp = glob_pers_df['Client number of children'].value_counts().values
        names_comp = glob_pers_df['Client number of children'].unique()
        
        fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Number of children distribution of all clients')
        
        # Compare to similar clients
        df_reduced = glob_pers_df[glob_pers_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        labels_sim = df_reduced['Client number of children'].value_counts().index
        values_sim = df_reduced['Client number of children'].value_counts().values
        names_sim = df_reduced['Client number of children'].unique()
        
        fig_desc_comp_sim = px.pie(values=values_sim, labels=labels_sim, names=names_sim, title = 'Number of children distribution of similar scoring clients')
        
    elif (desc_value =='FS'):
        # Compare to all clients
        labels_comp = glob_pers_df['Client family status'].value_counts().index
        values_comp = glob_pers_df['Client family status'].value_counts().values
        names_comp = glob_pers_df['Client family status'].unique()
        
        fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Family status distribution of all clients')
    
        # Compare to similar clients
        df_reduced = glob_pers_df[glob_pers_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        labels_sim = df_reduced['Client family status'].value_counts().index
        values_sim = df_reduced['Client family status'].value_counts().values
        names_sim = df_reduced['Client family status'].unique()
        
        fig_desc_comp_sim = px.pie(values=values_sim, labels=labels_sim, names=names_sim, title = 'Family status distribution of similar scoring clients')
        
    elif (desc_value =='HT'):
        # Compare to all clients
        labels_comp = glob_pers_df['Client housing situation'].value_counts().index
        values_comp = glob_pers_df['Client housing situation'].value_counts().values
        names_comp = glob_pers_df['Client housing situation'].unique()
        
        fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Housing type distribution of all clients')
        
        # Compare to similar clients
        df_reduced = glob_pers_df[glob_pers_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        labels_sim = df_reduced['Client housing situation'].value_counts().index
        values_sim = df_reduced['Client housing situation'].value_counts().values
        names_sim = df_reduced['Client housing situation'].unique()
        
        fig_desc_comp_sim = px.pie(values=values_sim, labels=labels_sim, names=names_sim, title = 'Housing type distribution of similar scoring clients')
    
    elif (desc_value =='ET'):
        # Compare to all clients
        labels_comp = glob_soc_df['Highest education level achieved of the client'].value_counts().index
        values_comp = glob_soc_df['Highest education level achieved of the client'].value_counts().values
        names_comp = glob_soc_df['Highest education level achieved of the client'].unique()
        
        fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Education level distribution of all clients')
    
        # Compare to similar clients
        df_reduced = glob_soc_df[glob_soc_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        labels_sim = df_reduced['Highest education level achieved of the client'].value_counts().index
        values_sim = df_reduced['Highest education level achieved of the client'].value_counts().values
        names_sim = df_reduced['Highest education level achieved of the client'].unique()
        
        fig_desc_comp_sim = px.pie(values=values_sim, labels=labels_sim, names=names_sim, title = 'Education level distribution of similar scoring clients')
        
    elif (desc_value =='OT'):
        # Compare to all clients
        labels_comp = glob_soc_df['Client occupation'].value_counts().index
        values_comp = glob_soc_df['Client occupation'].value_counts().values
        names_comp = glob_soc_df['Client occupation'].unique()
        
        fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Occupation distribution of all clients')
        
        # Compare to similar clients
        df_reduced = glob_soc_df[glob_soc_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        labels_sim = df_reduced['Client occupation'].value_counts().index
        values_sim = df_reduced['Client occupation'].value_counts().values
        names_sim = df_reduced['Client occupation'].unique()
        
        fig_desc_comp_sim = px.pie(values=values_sim, labels=labels_sim, names=names_sim, title = 'Occupation distribution of similar scoring clients')
        
    elif (desc_value =='IT'):
        # Compare to all clients
        labels_comp = glob_soc_df['Client income type'].value_counts().index
        values_comp = glob_soc_df['Client income type'].value_counts().values
        names_comp = glob_soc_df['Client income type'].unique()
        
        fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Income type distribution of all clients')
        
        # Compare to similar clients
        df_reduced = glob_soc_df[glob_soc_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        labels_sim = df_reduced['Client income type'].value_counts().index
        values_sim = df_reduced['Client income type'].value_counts().values
        names_sim = df_reduced['Client income type'].unique()
        
        fig_desc_comp_sim = px.pie(values=values_sim, labels=labels_sim, names=names_sim, title = 'Income type distribution of similar scoring clients')
        
    elif (desc_value =='IA'):
        # Compare to all clients
        fig_desc_comp = px.histogram(glob_soc_df, nbins=300, x='Client income', labels={'Client income': 'Income amount'}, title = 'Income amount distribution of all clients')
        fig_desc_comp.update_xaxes(range=[2e4, 1e6])
        
        # Compare to similar clients
        df_reduced = glob_soc_df[glob_soc_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        fig_desc_comp_sim = px.histogram(df_reduced, nbins=300, x='Client income', labels={'Client income': 'Income amount'}, title = 'Income amount distribution of similar scoring clients')
        fig_desc_comp_sim.update_xaxes(range=[2e4, 1e6])
        
    elif (desc_value =='RNP'):
        # Compare to all clients
        fig_desc_comp = px.histogram(glob_geo_df, nbins=30, x='Normalized population of region where client lives (higher number means the client lives in more populated region)', title = 'Region rating distribution of all clients', labels={'Normalized population of region where client lives (higher number means the client lives in more populated region)' : 'Normalized population of regions'})
        
        # Compare to similar clients
        df_reduced = glob_geo_df[glob_geo_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        fig_desc_comp_sim = px.histogram(df_reduced, nbins=30, x='Normalized population of region where client lives (higher number means the client lives in more populated region)', title = 'Region rating distribution of similar scoring clients', labels={'Normalized population of region where client lives (higher number means the client lives in more populated region)' : 'Normalized population of regions'})
    
    elif (desc_value =='RR'):
        # Compare to all clients
        labels_comp = glob_geo_df['Our rating of the region where client lives (1,2,3)'].value_counts().index
        values_comp = glob_geo_df['Our rating of the region where client lives (1,2,3)'].value_counts().values
        names_comp = glob_geo_df['Our rating of the region where client lives (1,2,3)'].unique()
        
        fig_desc_comp = px.pie(values=values_comp, labels=labels_comp, names=names_comp, title = 'Region rating distribution of all clients')
        
        # Compare to similar clients
        df_reduced = glob_geo_df[glob_geo_df['Client Id'].isin(df_filtered['SK_ID_CURR'].tolist())]
        labels_sim = df_reduced['Our rating of the region where client lives (1,2,3)'].value_counts().index
        values_sim = df_reduced['Our rating of the region where client lives (1,2,3)'].value_counts().values
        names_sim = df_reduced['Our rating of the region where client lives (1,2,3)'].unique()
        
        fig_desc_comp_sim = px.pie(values=values_sim, labels=labels_sim, names=names_sim, title = 'Region rating distribution of similar scoring clients')
        
    return [fig_desc_comp, fig_desc_comp_sim]
    

if __name__ == '__main__':
    # port = int(os.environ.get("PORT", 5000))
    # print(port)
    # app.run_server(debug=False, host='0.0.0.0', port=port)
    app.run_server(debug=False)