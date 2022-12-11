import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Get all countries' data of new cases and new deaths
def country_data(filename):
    ow = pd.read_csv(filename)
    ow = ow[["location","date","new_cases","new_deaths"]]
    ow = ow.sort_values(["location","date"])
    ow = ow.rename(columns={"location":"country"})
    return ow

# Get 2022 each country' population
def population_data(filename):
    population = pd.read_csv(filename)
    population = population.rename(columns={'2022_last_updated': 'population'})
    population = population[["country","population"]]
    return population

# Get all countries' vaccinations data
def vaccinations_data(filename):
    vaccinations = pd.read_csv(filename)
    # vaccinations = vaccinations[["date","country","total_vaccinations","people_vaccinated","people_fully_vaccinated"]]
    vaccinations = vaccinations[["date","country","people_vaccinated"]]
    return vaccinations

def f_1(row):
    row.population = row.population.replace(",","")
    return row

# Calculate the vaccination ratio
def f_2(row):
    row.ratio = row.people_vaccinated / row.population * 100 
    return row

# Select specific country to analyze
def dfCountry(countryName,df):
    mydf = df
    mydf = mydf[mydf["country"]==countryName]
    return mydf

# Show the figure
def show_plt(x,y,xName,yName):
    plt.scatter(x,y,color='olive')
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.show()

# Calculate the r2 scores
def model(ndArray1,ndArray2):
    lrModel = LinearRegression()
    x = np.array(ndArray1).reshape(-1, 1)
    y = np.array(ndArray2).reshape(-1, 1)
    lrModel.fit(x,y)
    
    # R2 Score
    r2 = lrModel.score(x,y)
    return r2,lrModel

# Calculate the regression line and draw it
def regression_line(x,y,lrModel,xlabel,ylabel):
    alpha = lrModel.coef_[0][0]
    beta = lrModel.intercept_[0]
    print('The fitting equation is: Y = %.6fX + %.6f' % (alpha, beta))
    plt.scatter(x, y, color='olive')
    plt.plot(x, alpha*x + beta, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Calculate the MAE,MSE,RMSE,R2
def evaluation(x,y,lrModel):
    test_x = np.array(x).reshape(-1, 1)
    test_y = np.array(y).reshape(-1, 1)
    predict_y = lrModel.predict(test_x)
    MAE = mean_absolute_error(test_y,predict_y)
    MSE = mean_squared_error(test_y,predict_y)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_y,predict_y)
    return MAE,MSE,RMSE,R2

if __name__ == "__main__":
    filenameCountryData = "./Data/owid-covid-data.csv"
    filenamePopulation = "./Data/2022_population.csv"
    filenameVaccinations = "./Data/country_vaccinations.csv"

    ow = country_data(filenameCountryData)
    population = population_data(filenamePopulation)
    vaccinations = vaccinations_data(filenameVaccinations)

    ow = ow[ow["country"].isin(set(population.country))]
    vaccinations = vaccinations[vaccinations["country"].isin(set(population.country))]

    df = pd.merge(ow,population)
    df = pd.merge(df,vaccinations,on=["country","date"])

    df = df.apply(f_1,axis=1)
    df.population = df.population.astype("float64")
    df.people_vaccinated = df.people_vaccinated.astype("float64")
    df = df.assign(ratio=[0]*len(df))
    df = df.apply(f_2, axis=1)

    # Belgium Italy France
    mydf = dfCountry("Italy",df)
    mydf.dropna(axis=0,subset = ["people_vaccinated"],inplace=True)

    show_plt(mydf.ratio,mydf.new_deaths,"Vaccination rate (%) ","New deaths")
    
    # R2 Score
    r2 = model(mydf['ratio'],mydf['new_deaths'])[0]
    print("R2-score:%.2f" % r2)

    plt.figure()
    y = np.array(mydf['new_cases']).reshape(-1, 1)
    list = []
    for i in mydf["date"]:
        list.append(i)
    date = [datetime.strptime(d, '%Y-%m-%d').date() for d in list]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(date[::25])
    plt.plot(date, y)
    plt.xlabel("date")
    plt.ylabel("New cases")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

    mydf = mydf[75:300]
    show_plt(mydf.ratio,mydf.new_deaths,"Vaccination rate (%) ","New deaths")

    # R2 Score
    r2 = model(mydf['ratio'],mydf['new_deaths'])[0]
    print("R2-score:%.2f" % r2)

    lrModel = model(mydf['ratio'],mydf['new_deaths'])[1]

    regression_line(mydf.ratio,mydf.new_deaths,lrModel,"Vaccination rate (%) ","New deaths")

    eva = evaluation(mydf['ratio'],mydf['new_deaths'],lrModel)
    MAE,MSE,RMSE,R2 = eva[0],eva[1],eva[2],eva[3]
    print("Mean absolute error(MAE) is: %.2f" % MAE)
    print("Mean Squared Error(MSE) is: %.2f" % MSE)
    print("Root Mean Squared Error(RMSE) is: %.2f" % RMSE)
    print("R2 score is: %.2f" % R2)