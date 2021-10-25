# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
"""
藉由分析住宿者是入住前就可以蒐集的資料，預測最後的這筆訂單是否會取消?
"""


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime


# %%
df = pd.read_csv("hotel_bookings.csv")
df.head()


# %%
df.dropna()


# %%
df.shape


# %%
df.columns


# %%
df.dtypes

# %% [markdown]
# # Combine the duplicate or not using columns

# %%
"""
the reason of dropping the column:
------------------------------------
deposit_type: it isn't about money.
required_car_parking_spaces: The almost record value is 0.
agent: this columns means agent ID, I think that hasn't too much use.
company: this columns means company ID, I think that hasn't too much use.
previous_cancellations: we can konw the customer canceled or not from "is_canceled" column.
previous_bookings_not_canceled: we can konw the customer canceled or not from "is_canceled" column.
is_repeated_guest: it doesn't about money directly.
days_in_waiting_list: it doesn't about money directly.
arrival_date_week_number: we can know this values by other columns.
distribution_channel: it doesn't about money directly.
"""
df = df.drop(columns=["deposit_type", "required_car_parking_spaces", "agent", "company", 'previous_cancellations',
       'previous_bookings_not_canceled', "is_repeated_guest", "arrival_date_week_number"])

# %% [markdown]
# # Combine the value about arrival date(year, month, date)

# %%
#MONTH DICTIONARY
MONTH_DICTIONARY = {"January": "01", "February":"02", "March":"03", "April":"04", "May":"05", "June":"06",
                    "July":"07", "August":"08", "September":"09", "October":"10", "November":"11", "December":"12"}

#change the value in arrival_data_month
df['arrival_date_month'] = df['arrival_date_month'].apply(lambda x: MONTH_DICTIONARY[x])

#create the value in arrival_data
hour_min_sec = " 12:00:00"
df['arrival_date'] = df["arrival_date_year"].astype("str") + "-" +df["arrival_date_month"] + "-" +                         df["arrival_date_day_of_month"].astype("str") + hour_min_sec

#create timestamp by transform arrival date to timestamp (pd.timestamp)
#datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
df["timestamp"] = df.apply(lambda record: datetime.strptime(record["arrival_date"], "%Y-%m-%d %H:%M:%S").timestamp(), axis=1)

#drop the duplicate 
#df.drop(columns=["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"])


# %%
df.columns


# %%
df.head()

# %% [markdown]
# # Chose the Inforamtion that hotel can recept before the customer check in.

# %%
#this columns could be know before customers check in
#we can change []
info_before_check_in_columns = ["hotel", "is_canceled", "lead_time","adults",
                                "children", "babies", "reserved_room_type", "meal", "customer_type", "adr"]

info_after_check_in_columns = ["assigned_room_type", 'reservation_status', 'reservation_status_date']


# %%
df_ibci = df[info_before_check_in_columns]
df_ibci.head()

# %% [markdown]
# # Encode the column which values type is not integer or float

# %%
from sklearn.preprocessing import LabelEncoder 


# %%
hotel_le = LabelEncoder()
meal_le = LabelEncoder()
room_le = LabelEncoder()
customerType_le = LabelEncoder()
df_ibci['meal'] = meal_le.fit_transform(df_ibci["meal"])
df_ibci["customer_type"] = customerType_le.fit_transform(df_ibci["customer_type"])
df_ibci["hotel"] = hotel_le.fit_transform(df_ibci["hotel"])
df_ibci["reserved_room_type"] = room_le.fit_transform(df_ibci["reserved_room_type"])


# %%
df_ibci.head()

# %% [markdown]
# # Plot the the corelation coefficient heatmap 

# %%
sns.heatmap(df_ibci.corr(), cmap="YlGn")

# %% [markdown]
# # Is canceled

# %%
#print(hotel_le.classes_)
plt.bar("Isn't canceled", df_ibci[df_ibci["is_canceled"]==0]["is_canceled"].shape[0])
plt.bar("Is canceled", df_ibci[df_ibci["is_canceled"]==1]["is_canceled"].shape[0])
plt.ylabel("Quantity of data")

# %% [markdown]
# 

# %%
city_hotel_condition = df_ibci["hotel"] == 0
resort_hotel_condition = df_ibci['hotel'] == 1
isnt_canceled_condition = df_ibci["is_canceled"] == 0
is_canceled_condition = df_ibci["is_canceled"] == 1

# %% [markdown]
# # Chat Line of  Lead time 

# %%
is_canceled_mean_leadtime = []
isnt_canceled_mean_leadtime = []
months = np.sort(df["arrival_date_month"].unique()) 
for month in months:
    month_condition = df["arrival_date_month"]==month
    isnt_canceled_mean = df[month_condition & isnt_canceled_condition]["lead_time"].mean()
    is_canceled_mean = df[month_condition & is_canceled_condition]["lead_time"].mean()
    isnt_canceled_mean_leadtime.append(isnt_canceled_mean)
    is_canceled_mean_leadtime.append(is_canceled_mean) 
plt.plot(months, isnt_canceled_mean_leadtime, label='Isnt canceled')
plt.plot(months, is_canceled_mean_leadtime, label='Canceled')
plt.legend(loc='best')
plt.xlabel("month")
plt.ylabel("")


# %%
#let it become the new feature.
df_ibci["mean_leadtime"] = df.apply(lambda row: is_canceled_mean_leadtime[int(row["arrival_date_month"])-1] if row["is_canceled"] == 1                                 else isnt_canceled_mean_leadtime[int(row["arrival_date_month"])-1], axis=1)

# %% [markdown]
# # Chat line of mean of adr each month

# %%
is_canceled_mean_adr = []
isnt_canceled_mean_adr = []
months = np.sort(df["arrival_date_month"].unique()) 
for month in months:
    month_condition = df["arrival_date_month"]==month
    isnt_canceled_mean = df[month_condition & isnt_canceled_condition]["adr"].mean()
    is_canceled_mean = df[month_condition & is_canceled_condition]["adr"].mean()
    isnt_canceled_mean_adr.append(isnt_canceled_mean)
    is_canceled_mean_adr.append(is_canceled_mean) 
plt.plot(months, isnt_canceled_mean_adr, label='Isnt canceled')
plt.plot(months, is_canceled_mean_adr, label='Canceled')
plt.legend(loc='best')
plt.xlabel("month")
plt.ylabel("")


# %%
#let it become the new feature.
df_ibci["mean_adr"] = df.apply(lambda row: is_canceled_mean_adr[int(row["arrival_date_month"])-1] if row["is_canceled"] == 1                                 else isnt_canceled_mean_adr[int(row["arrival_date_month"])-1], axis=1)

# %% [markdown]
# # Number of staying day

# %%
# df["number_of_staying_day"].unique()


# %%
# isnt_canceled_condition = df_ibci["is_canceled"] == 0
# is_canceled_condition = df_ibci["is_canceled"] == 1

# xtick = np.sort(df["number_of_staying_day"].unique())
# is_canceled_staying_day = []
# isnt_canceled_staying_day = []
# for value in xtick:
#     staying_day_condition = df["number_of_staying_day"] == value
#     isnt_canceled_quantity = df[isnt_canceled_condition & staying_day_condition].shape[0]
#     is_canceled_quantity =  df[is_canceled_condition & staying_day_condition].shape[0]
#     isnt_canceled_staying_day.append(isnt_canceled_quantity)
#     is_canceled_staying_day.append(is_canceled_quantity)
# fig = plt.figure()
# plt.plot(xtick, isnt_canceled_staying_day, label='Isnt canceled')
# plt.plot(xtick, is_canceled_staying_day, label='Canceled')
# plt.legend(loc='best')


# %%


# %% [markdown]
# # Rate of hotel cenceled and quantity of all data

# %%
quantity_of_adults_isnt_canceled =[]
quantity_of_adults_is_cenceled =[]
for quantity in range((df_ibci["adults"].max())):
    quantity_condition = df_ibci['adults'] == quantity
    quantity_of_adults_isnt_canceled.append(df_ibci[quantity_condition & isnt_canceled_condition].shape[0])
    quantity_of_adults_is_cenceled.append(df_ibci[quantity_condition&is_canceled_condition].shape[0])

plt.plot(range(df_ibci["adults"].max()), quantity_of_adults_isnt_canceled, label="Inst canceled")
plt.plot([quantity for quantity in range((df_ibci["adults"].max()))], 
        quantity_of_adults_is_cenceled, label='Canceled')
plt.xlim([0,5])
plt.legend(loc='best')
plt.xlabel("Number of adults")
plt.ylabel("Quantity of record")


# %%
#Children
quantity_of_children_isnt_canceled =[]
quantity_of_children_is_canceled =[]
for quantity in range(int(df_ibci["children"].max())):
    quantity_condition = df_ibci['children'] == quantity
    quantity_of_children_isnt_canceled.append(df_ibci[quantity_condition & isnt_canceled_condition].shape[0])
    quantity_of_children_is_canceled.append(df_ibci[quantity_condition&is_canceled_condition].shape[0])

plt.plot(range(int(df_ibci["children"].max())), quantity_of_children_isnt_canceled, label="Inst canceled")
plt.plot(range(int(df_ibci["children"].max())), 
        quantity_of_children_is_canceled,label='Canceled')
plt.xlim([0,5])
plt.legend(loc='best')
plt.xlabel("Number of children")
plt.ylabel("Quantity of record")


# %%
#Baby
quantity_of_babies_isnt_canceled =[]
quantity_of_babies_is_canceled =[]
for quantity in range(int(df_ibci["babies"].max())):
    quantity_condition = df_ibci['babies'] == quantity
    quantity_of_babies_isnt_canceled.append(df_ibci[quantity_condition & isnt_canceled_condition].shape[0])
    quantity_of_babies_is_canceled.append(df_ibci[quantity_condition&is_canceled_condition].shape[0])

plt.plot(range(int(df_ibci["babies"].max())), quantity_of_babies_isnt_canceled, 0.2,label="Isnt canceled")
plt.plot(range(int(df_ibci["babies"].max())), 
        quantity_of_babies_is_canceled, 0.2,label='Canceled')
plt.xlim([0,5])
plt.legend(loc='best')
plt.xlabel("Number of Babies")
plt.ylabel("Quantity of record")

# %% [markdown]
# # What kind of meal does the customer eat?
# 

# %%
meal_le.classes_


# %%
quantity_of_meal_isnt_canceled =[]
quantity_of_meal_is_canceled =[]
for index in range(5):
    quantity_condition = df_ibci['meal'] == index
    quantity_of_meal_isnt_canceled.append(df_ibci[quantity_condition & isnt_canceled_condition].shape[0])
    quantity_of_meal_is_canceled.append(df_ibci[quantity_condition & is_canceled_condition].shape[0])

plt.plot(range(len(df_ibci["meal"].unique())), quantity_of_meal_isnt_canceled, label="Isnt Canceled")
plt.plot(range(len(df_ibci["meal"].unique())), 
         quantity_of_meal_is_canceled,label='Canceled')
plt.xlim([0,5])
plt.legend(loc='best')
plt.xlabel("Meal Category")
plt.ylabel("Quantity of record")
plt.xticks(np.arange(5), ['BB', 'FB', 'HB', 'SC', 'Undefined'])

# %% [markdown]
# # What kind of customer type ?

# %%
customerType_le.classes_


# %%
quantity_of_customer_isnt_canceled=[]
quantity_of_customer_is_canceled =[]
for index in range(4):
    quantity_condition = df_ibci['customer_type'] == index
    quantity_of_customer_isnt_canceled.append(df_ibci[quantity_condition & isnt_canceled_condition].shape[0])
    quantity_of_customer_is_canceled.append(df_ibci[quantity_condition & is_canceled_condition].shape[0])

plt.plot(range(len(df_ibci["customer_type"].unique())), quantity_of_customer_isnt_canceled,label="Isnt Canceled")
plt.plot(range(len(df_ibci["customer_type"].unique())), 
         quantity_of_customer_is_canceled,label='Canceled')
plt.xlim([0,4])
plt.legend(loc='best')
plt.xlabel("Customer type")
plt.ylabel("Quantity of record")
plt.xticks(np.arange(4), ['Contract', 'Group', 'Transient', 'Transient-Party'])

# %% [markdown]
# # Which month has the most Canceled Demanding?

# %%
MONTH_DICTIONARY.values()


# %%
quantity_of_isnt_canceled_each_month = []
quantity_of_is_canceled_each_month = []
for month in MONTH_DICTIONARY.values():
    in_withch_month = df["arrival_date_month"] == month
    quantity_of_isnt_canceled_each_month.append(df[ isnt_canceled_condition & in_withch_month].shape[0])
    quantity_of_is_canceled_each_month.append(df[ is_canceled_condition & in_withch_month].shape[0])
plt.plot( [month for month in MONTH_DICTIONARY.values()], quantity_of_isnt_canceled_each_month, label='Isnt canceled')
plt.plot( [month for month in MONTH_DICTIONARY.values()], quantity_of_is_canceled_each_month, label='Canceled')
plt.xlabel("Month")
plt.ylabel("Quantity of booking")
plt.title("Quantity in different hotel in each month")
plt.legend(loc='best')
"""
Conclusion:
"""

# %% [markdown]
# # What time do people cancel the booking

# %%
is_canceled = df["is_canceled"] == 1
is_not_canceled = df['is_canceled'] == 0
quantity_of_canceled_in_each_year = []
quantity_of_isnt_canceled_in_each_year = []

for year in [2015, 2016, 2017]:
    year_condition = df["arrival_date_year"] == year
    quantity_of_canceled_in_each_year.append(df[is_canceled & year_condition].shape[0])
    quantity_of_isnt_canceled_in_each_year.append(df[is_not_canceled & year_condition].shape[0])
plt.plot(["2015", '2016', '2017'], quantity_of_isnt_canceled_in_each_year, label="Isnt Canceled")    
plt.plot(["2015", '2016', '2017'], quantity_of_canceled_in_each_year, label="Canceled")
plt.xlabel("Month")
plt.ylabel("Quantity")
plt.legend(loc='best')

# %% [markdown]
# # Create Simple model

# %%
df_ibci.columns


# %%
df.head()


# %%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
#canceled      44224
#not canceled  75166
#["hotel", "number_of_staying_day", "adults", "children", "babies", "reserved_room_type", "meal", "customer_type"]
X = df.loc[:,['lead_time', "adr"]]
y = df["is_canceled"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

std = StandardScaler()
X_std = std.fit_transform(X)
X_train_std = std.fit_transform(X_train)
X_test_std = std.fit_transform(X_test)


# %%
ptn = Perceptron()
ptn.fit(X_train_std, y_train)
print("Accuracy of training data in perceptron: %.3f", ptn.score(X_train_std, y_train))
print("Accuracy of training data in perceptron: %.3f", ptn.score(X_test_std, y_test))


# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_std, y)
label = knn.predict(X_std)
print("Accuracy of KNN: %.3f"%(knn.score(X_std, y)))

plt.figure(figsize=(12,9))
plt.scatter([row[0]for row in X_std[label==0]], [row[0]for row in X_std[label==0]], label="Isnt Canceled",  alpha=0.4)
plt.scatter([row[0]for row in X_std[label==1]], [row[1]for row in X_std[label==1]],label="Canceled", alpha=0.4)
plt.scatter([row[0]for row in X_std[label!=y]], [row[1]for row in X_std[label!=y]],label="Error",  alpha=0.4)
plt.xlabel("leadtime")
plt.ylabel("adr")
plt.xlim([-2,6])
plt.ylim([-2,10])
plt.legend(loc="best")


# %%
from minisom import MiniSom


# %%
som  = MiniSom(2, 2, 2,sigma=0.3, learning_rate=0.1)
max_iter = 50000
q_error = []
x_iter = []
for i in range(max_iter):
    percent = 100 * (i+1) / max_iter
    randInt = np.random.randint(len(X_train_std))
    som.update(X_train_std[randInt], som.winner(X_train_std[randInt]), i, max_iter)
    if (i+1) % 100 == 0:
        q_error.append(som.quantization_error(X_train_std))
        x_iter.append(i)
        
plt.plot(x_iter, q_error)
plt.ylabel('quantization error')
plt.xlabel('iteration index')


# %%
plt.figure(figsize=(8, 7))
frequencies = som.activation_response(X_train_std)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.show()


# %%
cluster=[(0, 0), (0, 1), (1, 0), (1, 1)]
clusterA=[]
clusterM = [[], [], [], []]
for index, row in enumerate(X_std):
    clusterA.append(cluster.index(som.winner(row)))
    clusterM[cluster.index(som.winner(row))].append(index)


# %%
plt.scatter([row[0] for row in X_std[pd.Series(clusterA)==0]],
            [row[1] for row in X_std[pd.Series(clusterA)==0]], label='cluster0', alpha=0.6)
plt.scatter([row[0] for row in X_std[pd.Series(clusterA)==1]],
            [row[1] for row in X_std[pd.Series(clusterA)==1]], label='cluster1', alpha=0.6)
plt.scatter([row[0] for row in X_std[pd.Series(clusterA)==2]], 
            [row[1] for row in X_std[pd.Series(clusterA)==2]], label='cluster2', alpha=0.6)
plt.scatter([row[0] for row in X_std[pd.Series(clusterA)==3]],
            [row[1] for row in X_std[pd.Series(clusterA)==3]], label='cluster3', alpha=0.6)
plt.xlabel("leadtime")
plt.ylabel("adr")
plt.xlim([-2,6])
plt.ylim([-2,10])
plt.legend(loc="best")


# %%
model = []
for index, x_index in enumerate(clusterM):
    ptn = Perceptron()
    x = X_std[x_index]
    label = y[x_index]
    x_train, x_test, label_train, label_test = train_test_split(x, label, test_size=0.3) 
    ptn.fit(x_train, label_train)
    print("Model", cluster[index])
    print("   Accuracy of training data: %.3f"% ptn.score(x_train, label_train))
    print("   Accuracy of training data: %.3f"% ptn.score(x_test, label_test))
    model.append(ptn)


# %%
random_index = np.random.randint(len(X_std), size=10000)
error = 0
for index, randIndex in enumerate(random_index):
    clusterType = cluster.index(som.winner(X_std[randIndex]))
    clusterAnswer = model[clusterType].predict(X_std[[randIndex]])
    if y[randIndex] != clusterAnswer[0]:
        error += 1
print("Accuracy of SOM+KNN:", 1 - error/len(random_index))


# %%
X_std[clusterA==0]


# %%
pd.Series(clusterA)==1


# %%



