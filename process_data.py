import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
row_count=0
with open("data.csv", 'r') as f:
    for line in f:
        row_count += 1
row_count = row_count - 1
first_row = True
X = np.zeros((row_count, 6))
with open("data.csv", 'r') as train_file:
    csv_reader = csv.reader(train_file, delimiter=',')
    i = 0
    for row in csv_reader:
        if first_row:
            first_row = False
            continue
        row = [w.replace('None', 'nan') for w in row]
        X[i] = row[2:]
        i=i+1

PM25 = X[:, 0]
PM10 = X[:, 1]
SO2 = X[:, 2]
NO2 = X[:, 3]
O3 = X[:, 4]
CO = X[:, 5]

def twentyFourAvg(X):
    Y = np.zeros((int(np.size(X, 0)/24), 1))
    for i in range(0, np.size(X, 0), 24):
        Y[int(i/24)] = np.nanmean(X[i:i+24])
    return Y

def eightAvg(X):
    Y = np.zeros((int(np.size(X, 0)/8), 1))
    for i in range(0, np.size(X, 0), 8):
        Y[int(i/8)] = np.nanmean(X[i:i+8])
    return Y

def calcAQI(X, vals):
    aqi = [50, 100, 200, 300, 400]
    aqi_vals = np.zeros((np.size(X, 0)))

    for i in range(np.size(X, 0)):
        prev_val = 0
        prev_aqi = 0
        for j in range(np.size(vals, 0)):
            if X[i] <= vals[j]:
                aqi_vals[i] = prev_aqi + (X[i] - prev_val) / (vals[j] - prev_val) * (aqi[j]-prev_aqi)
                break
            prev_aqi = aqi[j]-1
            prev_val = vals[j]
    return aqi_vals
#calculate 24-hour average for PM25, PM10, SO2, NO2
PM25_avg = twentyFourAvg(PM25)
PM10_avg = twentyFourAvg(PM10)
SO2_avg = twentyFourAvg(SO2)
NO2_avg = twentyFourAvg(NO2)

#calculate 8-hour average for O3, CO
O3_avg = eightAvg(O3)
CO_avg = eightAvg(CO)

#calculate the AQI sub-indices
AQI_PM25 = calcAQI(PM25_avg, [30, 60, 90, 120, 250])
AQI_PM10 = calcAQI(PM10_avg, [50, 100, 250, 350, 430])
AQI_SO2 = calcAQI(SO2_avg, [40, 80, 380, 800, 1600])
AQI_NO2 = calcAQI(NO2_avg, [40, 80, 180, 280, 400])
AQI_O3_temp = calcAQI(O3_avg, [50, 100, 168, 208, 748])
AQI_CO_temp = calcAQI(CO_avg, [1, 2, 10, 17, 34])

AQI_O3 = np.zeros((int(np.size(AQI_O3_temp, 0)/3)))
AQI_CO = np.zeros((int(np.size(AQI_CO_temp, 0)/3)))

#find the max out of the 8 hour averages for O3 and CO
for i in range(0, np.size(AQI_O3_temp, 0), 3):
    AQI_O3[int(i/3)] = np.max(AQI_O3_temp[i:i+2])
    AQI_CO[int(i/3)] = np.max(AQI_CO_temp[i:i+2])

#calculate the AQI for each day
AQI = np.zeros((np.size(AQI_PM10, 0)))
AQI_comp = np.zeros((np.size(AQI_PM10, 0)))

for i in range(np.size(AQI_PM10, 0)):
    AQI[i] = np.max([AQI_PM25[i], AQI_PM10[i], AQI_SO2[i], AQI_NO2[i], AQI_O3[i], AQI_CO[i]])
    AQI_comp[i] = np.argmax([AQI_PM25[i], AQI_PM10[i], AQI_SO2[i], AQI_NO2[i], AQI_O3[i], AQI_CO[i]])


months = [31, 28, 31, 30, 31, 30, 31,  31, 30, 31, 30, 31]
monthlyAQI = np.zeros(12)
monthlyAQI_cause = np.zeros(12)

monthlyPM25 = np.zeros(12)
monthlyPM10 = np.zeros(12)
monthlySO2 = np.zeros(12)
monthlyNO2 = np.zeros(12)
monthlyO3 = np.zeros(12)
monthlyCO = np.zeros(12)

cumul_days = 0
for i in range(np.size(months, 0)):
    X = AQI[cumul_days:cumul_days+months[i]]
    Y = AQI_comp[cumul_days:cumul_days+months[i]]
    a = np.nonzero(X)
    monthlyAQI[i] = np.mean(X[a])
    monthlyAQI_cause[i] = stats.mode(Y)[0]

    X = AQI_PM25[cumul_days:cumul_days+months[i]]
    a = np.nonzero(X)
    monthlyPM25[i] = np.mean(X[a])

    X = AQI_PM10[cumul_days:cumul_days+months[i]]
    a = np.nonzero(X)
    monthlyPM10[i] = np.mean(X[a])

    X = AQI_SO2[cumul_days:cumul_days+months[i]]
    a = np.nonzero(X)
    monthlySO2[i] = np.mean(X[a])

    X = AQI_NO2[cumul_days:cumul_days+months[i]]
    a = np.nonzero(X)
    monthlyNO2[i] = np.mean(X[a])

    X = AQI_O3[cumul_days:cumul_days+months[i]]
    a = np.nonzero(X)
    monthlyO3[i] = np.mean(X[a])

    X = AQI_CO[cumul_days:cumul_days+months[i]]
    a = np.nonzero(X)
    monthlyCO[i] = np.mean(X[a])

    cumul_days = cumul_days + months[i]

print(monthlyAQI)
print(np.mean(monthlyAQI))
print(monthlyAQI_cause)

pm10mask = np.isfinite(monthlyPM10)
pm25mask = np.isfinite(monthlyPM25)
so2mask = np.isfinite(monthlySO2)
no2mask = np.isfinite(monthlyNO2)
o3mask = np.isfinite(monthlyO3)
comask = np.isfinite(monthlyCO)

monthsX = np.arange(12) + 1

print(monthlyPM10)
print(monthlyPM25)
print(monthlySO2)
print(monthlyNO2)
print(monthlyO3)
print(monthlyCO)

plt.plot(monthsX, monthlyAQI)
plt.xlabel("Months")
plt.ylabel("AQI")
plt.show()

plt.plot(monthsX[pm10mask], monthlyPM10[pm10mask])
plt.xlabel("Months")
plt.ylabel("PM10 index")
plt.show()

plt.plot(monthsX[pm25mask], monthlyPM25[pm25mask])
plt.xlabel("Months")
plt.ylabel("PM2.5 index")
plt.show()

plt.plot(monthsX[so2mask], monthlySO2[so2mask])
plt.xlabel("Months")
plt.ylabel("SO2 index")
plt.show()

plt.plot(monthsX[no2mask], monthlyNO2[no2mask])
plt.xlabel("Months")
plt.ylabel("NO2 index")
plt.show()

plt.plot(monthsX[o3mask], monthlyO3[o3mask])
plt.xlabel("Months")
plt.ylabel("O3 index")
plt.show()

plt.plot(monthsX[comask], monthlyCO[comask])
plt.xlabel("Months")
plt.ylabel("CO index")
plt.show()

#calculate the histogram of AQI severity
plt.hist(AQI, bins = [0,50,100,200,300,400, 500], stacked=True)
plt.title("Histogram of AQI vs days")
plt.show()
hist = np.histogram(AQI, [0, 50, 100, 200, 300, 400, 500])
print(hist)
