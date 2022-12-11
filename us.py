import pandas as pd
import ssl
from urllib.request import urlopen
import plotly.express as px
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

def us_data(url):
    url_deaths = url
    df_deaths = pd.read_html(url_deaths)[0]
    df_deaths = df_deaths[['USA State','Total Deaths','Population']]
    return df_deaths

def vaccinations_data(file):
    df_vaccinations = pd.read_excel(file)
    df_vaccinations.rename(columns={"Place":"USA State"},inplace=True)
    return df_vaccinations

if __name__ == "__main__":
    us_data_url = "https://www.worldometers.info/coronavirus/country/us/#graph-deaths-daily"
    us_code_url = "./Data/us-state-ansi-fips.csv"
    us_vaccinations_url = "./Data/vaccination.xlsx"

    df_deaths = us_data(us_data_url)

    df_code = pd.read_csv(us_code_url)
    df_code.rename(columns={"stname":"USA State"},inplace=True)
    df_code.rename(columns={" stusps":"code"},inplace=True)

    df_deaths['Population'] = df_deaths['Population'].astype("float64")
    df_deaths['death_ratio'] = df_deaths[['Total Deaths','Population']].apply(lambda x:x['Total Deaths']/x['Population'],axis=1)
    df = pd.merge(df_deaths,df_code,on="USA State")

    df_vaccinations = vaccinations_data(us_vaccinations_url)
    df = pd.merge(df_vaccinations,df,on="USA State")

    df['code'] = df['code'].str.strip()

    fig = px.choropleth(locations=df['code'], locationmode="USA-states", color=df['death_ratio'], scope="usa",title="death_ratio")
    fig.show()

    fig = px.choropleth(locations=df['code'], locationmode="USA-states", color=df['Percent fully vaccinated'], scope="usa",title="vaccination_ratio")
    fig.show()

    plt.bar(df[:10]["code"],df[:10]["Percent fully vaccinated"])
    plt.gcf().autofmt_xdate()
    plt.title("10 states' vaccination rate")
    plt.show()

    plt.bar(df[:10]["code"],df[:10]["death_ratio"])
    plt.gcf().autofmt_xdate()
    plt.title("10 states' death_ratio")
    plt.show()