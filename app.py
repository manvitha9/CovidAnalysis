
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import plotly
import plotly.express as px
import plotly.graph_objects as go
#plt.rcParams['figure.figsize']=17,8
import plotly.offline as pyo
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.tools as tls
from fbprophet import Prophet
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import os
import flask

# server = flask.Flask(__name__)
# app = dash.Dash(__name__, server = server)
# server.secret_key = os.environ.get('secret_key', 'secret')

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True

# init_notebook_mode(connected=True)
# cd '/content/drive/My Drive/Project_v1'
# Data from The New York Times, based on reports from state and local health agencies.
# Tracking Page - https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html
#df = pd.read_csv('us-states.csv')
df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv',error_bad_lines= False)
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
df['code'] = df['state'].map(us_state_abbrev)
# checking if there are any unmapped values for code
drop = df.notna()
all(drop)
df_new = df.set_index(['date','state'])
state_wise = df_new.loc['2020-04-17'].drop('fips',axis = 1)
state_wise.style.background_gradient(cmap='Reds')


#df=pd.read_csv(r"C\Users\rahul\OneDrive\Desktop\SPRING\Python\Final Project\covid-19-data-master\us.csv")
us_cases_deaths=pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv',error_bad_lines= False)
#plotly Express - Confirmed Cases in USA.
fig_total=px.bar(us_cases_deaths,x="date",y="cases",color='cases',title='Confirmed cases in United States',log_y=True)
#plotly Express - Confirmed Deaths in USA.
fig_total_deaths=px.bar(us_cases_deaths,x="date",y="deaths",color='deaths',title='Confirmed Deaths in United States')

# Difference between two rows in the USA.
#df['us_cases_deaths'] = df['us_cases_deaths'] - df['us_cases_deaths'].shift(-1)
us_cases_deaths["diff"] = us_cases_deaths["cases"].diff(-1)
us_cases_deaths["diff"] = us_cases_deaths["diff"].abs()
fig_daily=px.bar(us_cases_deaths,x="date",y="diff",color='diff',title='Increase in number of cases w.r.t Date')

# new map in plotly

### colorscale:
scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.3, '#ff4d4d'],
       [0.5, '#ff1a1a'],[0.7, '#cc0000'],[1.0, '#4d0000']] # reds

### create empty list for data object:
data_slider = []

for date in df.date.unique():
  #select the date
  df_selected = df[df['date']== date]
  # convert the data to string type
  for col in df_selected.columns:
    df_selected[col] = df_selected[col].astype(str)

  df_selected['text'] = df_selected['state'] + '<br>'+ ' Cases:' + df_selected['cases']+'<br>'+ ' Deaths:' + df_selected['deaths']

  ### create the dictionary with the data for the current date
  data_one_day = dict(
                        type='choropleth',
                        locations = df_selected['code'],
                        z=df_selected['cases'].astype(float),
                        locationmode='USA-states',
                        colorscale = scl,
                        text = df_selected['text'],
                        )

  data_slider.append(data_one_day)  # I add the dictionary to the list of dictionaries for the slider


##  create the steps for the slider

steps = []
startdate = "01/21/2020"
for i in range(len(data_slider)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label='Date {}'.format(pd.to_datetime(startdate,format="%m/%d/%Y")+ pd.DateOffset(days=i))) # label to be displayed for each step (year)
    step['args'][1][i] = True
    steps.append(step)

##  I create the 'sliders' object from the 'steps'

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]
# I set up the layout (including slider option)

layout = dict(geo=dict(scope='usa',projection={'type': 'albers usa'}),sliders=sliders)

# I create the figure object:
fig_chloropleth = dict(data=data_slider, layout=layout)
# prediction with prophet

confirmed_total_usa_cases =  pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv",error_bad_lines=False)
# confirmed_total_usa_cases.drop(confirmed_total_usa_cases.tail(10).index,inplace=True)

del confirmed_total_usa_cases['deaths']
confirmed_total_usa_cases= confirmed_total_usa_cases.rename(columns={'date': 'ds', 'cases': 'y'})
m = Prophet(interval_width = 0.95)
m.fit(confirmed_total_usa_cases)
future = m.make_future_dataframe(periods =  7)
# predicting the future with data. and upper and lower limits of y value.
forecast = m.predict(future)
forecast_bounds =forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(7)
confirmed_forecast_plot = m.plot(forecast)
forecast_plot = tls.mpl_to_plotly(confirmed_forecast_plot)
confirmed_forecast_plot1 =  m.plot_components(forecast)


#setting the app layout
app.layout = html.Div(children=[
                      html.H1(children = 'COVID19 Visualization and Forecasting ',style={'textAlign': 'center','color': '#7FDBFF','padding': 15}),

                      html.H2(children = 'An overview of cases in the US and prediction for the next 7 days',style={'textAlign': 'center','padding': 10}),

                      dcc.Tabs(id="tabs", value='tab-1',children=[
                        dcc.Tab(label='Total cases', value='tab-1'),
                        dcc.Tab(label='Daily new cases', value='tab-2'),
                        dcc.Tab(label='Reported Deaths', value='tab-3')]),
                      html.Div(id = 'tabs-content',style={"margin-left": "15px","margin-top": "10px"}),

                      dcc.Dropdown(id = 'cases', options = [{'label':'State_wise','value': 'state'},{'label':'World_wise','value': 'world'}],value = 'state',
                                   style={'width':'150px',"margin-left": "5px"}),
                      html.Div(id = 'cases_chloropleth',style={"margin-left": "15px","margin-top": "10px"}),

                      dcc.Tabs(id="prediction", value='graph',children=[
                        dcc.Tab(label='Total cases', value='graph'),
                        dcc.Tab(label='Daily new cases', value='table')],style={"padding" : "10px"}),

                      html.Div(id = 'prediction-content',style={"padding" : "10px"}),

                      ])

#callback to control the tab content
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([dcc.Graph(figure = fig_total )])
    elif tab == 'tab-2':
        return html.Div([dcc.Graph(figure = fig_daily)])
    elif tab == 'tab-3':
        return html.Div([dcc.Graph(figure = fig_total_deaths)])
@app.callback(Output('cases_chloropleth','children'),
              [Input('cases','value')])
def update_output(value):
    if value == 'state':
        return html.Div([dcc.Graph(figure = fig_chloropleth)])
    elif value == 'world':
        return html.Div([dcc.Graph(figure = fig_chloropleth)])
#callback for prediction
@app.callback(Output('prediction-content', 'children'),
              [Input('prediction', 'value')])
def render_content(tab):
    if tab == 'graph':
        return html.Div([dcc.Graph(figure =forecast_plot)])
    elif tab == 'table':
        return dash_table.DataTable( id='Prediction_table', columns=[{"name": i, "id": i} for i in forecast_bounds], data=forecast_bounds.to_dict('records'))

if __name__ == '__main__':
    app.run_server(debug=True)
