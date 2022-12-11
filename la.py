import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score

la_population = 10097820

def vaccination_data(file):
    la_vaccinations = pd.read_csv(file)
    return la_vaccinations

def la_data(file):
    us_cases_deaths = pd.read_csv(file)
    return us_cases_deaths

def f_1(row):
    row.population = row.population
    return row

def f_2(row):
    row.ratio = row.cumulative_at_least_one_dose / row.population * 100 
    return row

def show_plt(x,y,xName,yName):
    plt.scatter(x,y,color='olive')
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.show()

def model(ndArray1,ndArray2):
    lrModel = LinearRegression()
    x = np.array(ndArray1).reshape(-1, 1)
    y = np.array(ndArray2).reshape(-1, 1)
    lrModel.fit(x,y)
    
    # R2 Score
    r2 = lrModel.score(x,y)
    return r2,lrModel

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
    fileVaccination = "./Data/LAvaccination.csv"
    fileCaseDeaths = "./Data/us_cases_deaths.csv"

    la = vaccination_data(fileVaccination)
    la = la.set_index("county").loc["Los Angeles"]
    la = la.sort_values(by="administered_date")
    la = la[["administered_date","partially_vaccinated","cumulative_fully_vaccinated","cumulative_at_least_one_dose"]]
    la.rename(columns={"administered_date":"date"},inplace=True)

    us_cases_deaths = la_data(fileCaseDeaths)
    us_cases_deaths = us_cases_deaths.set_index("county").loc["Los Angeles"]
    us_cases_deaths = us_cases_deaths.sort_values(by="date")
    us_cases_deaths = us_cases_deaths[50:]

    df = pd.merge(la,us_cases_deaths,how="right",on="date")
    new_deaths = df["deaths"].diff()
    new_cases = df["cases"].diff()

    df.insert(loc=2,column="population",value=la_population)
    df.insert(loc=len(df.columns),column="new_deaths",value=new_deaths)
    df.insert(loc=len(df.columns),column="new_cases",value=new_cases)
    df = df.drop(index=0)
    df = df[df['new_deaths']<=800]
    df.dropna(axis=0,subset = ["cumulative_at_least_one_dose"],inplace=True)

    df.population = df.population.astype("float64")
    df['people_vaccinated'] = df['cumulative_at_least_one_dose'].astype("float64")

    df = df.apply(f_1,axis=1)
    df = df.assign(ratio=[0]*len(df))
    df = df.apply(f_2, axis=1)

    mydf = df
    mydf['ratio'] = mydf[['cumulative_at_least_one_dose','population']].apply(lambda x:x['cumulative_at_least_one_dose']/x['population'],axis=1)
    mydf = mydf.set_index("date")
    mydf = mydf.loc["2021-01-01":"2021-10-01"]
    
    x = np.array(mydf['ratio']).reshape(-1, 1)
    y = np.array(mydf['new_deaths']).reshape(-1, 1)

    show_plt(mydf.ratio,mydf.new_deaths,"Vaccination rate","New deaths")

    r2 = model(mydf['ratio'],mydf['new_deaths'])[0]
    print("R2-score:%.2f" % r2)

    lrModel = model(mydf['ratio'],mydf['new_deaths'])[1]

    regression_line(mydf.ratio,mydf.new_deaths,lrModel,"Vaccination rate","New deaths")

    eva = evaluation(mydf['ratio'],mydf['new_deaths'],lrModel)
    MAE,MSE,RMSE,R2 = eva[0],eva[1],eva[2],eva[3]
    print("Mean absolute error(MAE) is: %.2f" % MAE)
    print("Mean Squared Error(MSE) is: %.2f" % MSE)
    print("Root Mean Squared Error(RMSE) is: %.2f" % RMSE)
    print("R2 score is: %.2f" % R2)