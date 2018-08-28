import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds


dataFrame = pd.read_csv('./test_files/user_record_clean_test_db.csv', header=0, 
                         names=['personID', 'contentID', 'eventStrength', 'timestamp'],
                         engine='python')
print(dataFrame.shape)
print(dataFrame.head(10))

transform = {'VIEW':1, 'LIKE':2, 'COMMENT CREATED':3, 'BOOKMARK':4, 'FOLLOW':5}
print(transform['VIEW'])

ts = dataFrame.copy()
ts = ts.sort_values(by=['timestamp'],ascending=True)
#ts = ts.sort_values(by=[], inplace=True)
valid_ts = 1477059239
testing_ts = 1482248589

R_df = dataFrame.pivot(index='personID', columns='contentID', values='eventStrength').fillna(0)

R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis=1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
R_demeaned.shape

Train_df = dataFrame.copy() 
Valid_df = dataFrame.copy()

counter = -1
trainToDiscard = []
validToDiscard = []
for i in Train_df[:]['timestamp']:
    counter += 1
    if i > testing_ts:
        validToDiscard.append(counter)
    if i > valid_ts:
            trainToDiscard.append(counter)

for i in trainToDiscard:
    Train_df.set_value(i, 'eventStrength', 0)

for i in validToDiscard:
    Valid_df.set_value(i, 'eventStrength', 0)
#Train_df.head()
#Train_df
print(len(trainToDiscard), len(validToDiscard))
Train_df = Train_df.pivot(index='personID', columns='contentID', values='eventStrength').fillna(0)
Valid_df = Valid_df.pivot(index='personID', columns='contentID', values='eventStrength').fillna(0)
Train = Train_df.as_matrix()
Valid = Valid_df.as_matrix()

train_mean = np.mean(Train, axis=1)
valid_mean = np.mean(Valid, axis=1)

# NORMALIZE
Train_demeaned = Train - train_mean.reshape(-1, 1)
Valid_demeaned = Valid - train_mean.reshape(-1, 1)
def rmse(targetMatrix, originalMatrix):
    originalNonzero = originalMatrix[np.where(originalMatrix != 0.0)]
    target = targetMatrix[np.where(originalMatrix != 0.0)]
    #print(originalNonzero.shape)
    return np.sqrt(np.mean((originalNonzero - target)**2))

U, Sigma, Vt = svds(Train_demeaned, k=50)
user_predicted_ratings = U @ np.diag(Sigma) @ Vt + train_mean.reshape(-1, 1)

rmse(user_predicted_ratings, Train)
train_RMSEs = []
valid_RMSEs = []
for latent in range(15,41):
    U, Sigma, Vt = svds(Train_demeaned, k=latent)
    train_predicted_ratings = U @ np.diag(Sigma) @ Vt + train_mean.reshape(-1, 1)
    train_rmse = rmse(train_predicted_ratings, Train)
    train_RMSEs.append(train_rmse)
    
    U, Sigma, Vt = svds(Valid_demeaned, k=latent)
    valid_predicted_ratings = U @ np.diag(Sigma) @ Vt + train_mean.reshape(-1, 1)
    valid_rmse = rmse(valid_predicted_ratings, Train)
    valid_RMSEs.append(valid_rmse)
    print(latent, train_rmse, valid_rmse)

l = len(train_RMSEs)
x = np.linspace(1, l, l)
y1 = np.array(train_RMSEs)
y2 = np.array(valid_RMSEs)
plt.figure()
plt.plot(np.arange(15,41),y1)
plt.show()
plt.figure()
plt.plot(np.arange(15,41),y2)
plt.show()