from tensorforce.agents import vpg_agent
from CalculateEER import SKGetEER as GetEER
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

data = pickle.load(open("../Data/CMU2X_dat.pickle", "rb"))
X = np.array(data['data'])
Y = np.array(data['labels'])

n = X.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
cnnData = pickle.load(open("./cnnPredictions.pickle", "rb"))
xgboostData = pickle.load(open("./xgboostPredictions.pickle", "rb"))

cnnT = cnnData["test"]
xgT = xgboostData["test"]

cnnV = cnnData["valid"]
xgV = xgboostData["valid"]

agent = vpg_agent.VPGAgent(
    states=dict(type='float', shape=(2,)),
    actions=dict(type='float', shape=(2,)),
    network=[
        dict(type='dense', size=10),
        dict(type='dense', size=10)
    ]
)
bestEER = 10000


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def getAccuracy(cnn, xg, output, w1, w2):
    combined = (w1 * cnn + w2 * xg) / (w1 + w2)
    acc = accuracy(combined, output)
    return acc


def getScore(cnn, xg, output, w1, w2):
    global bestEER
    combined = (w1*cnn + w2*xg)/(w1+w2)
    eer = GetEER(combined, output)
    if eer < bestEER:
        #print(eer, w1, w2)
        bestEER = eer
    return eer


def fitness(w):
    return 1 - getScore(cnnV, xgV, y_valid, abs(w[0]), abs(w[1]))


print("Test Acc:", getAccuracy(cnnT, xgT, y_test, -0.03711697, -2.0327823 ))
print("Valid Acc:", getAccuracy(cnnV, xgV, y_valid, -0.03711697, -2.0327823 ))
raise KeyError


state = [1, 1]
eers = []
weights = []
for i in range(1000):
    action = agent.act(state)
    #print(action)
    reward = 100*fitness(action)
    w = action
    testEER = getScore(cnnT, xgT, y_test, abs(w[0]), abs(w[1]))
    # Add experience, agent automatically updates model according to batch size
    agent.observe(reward=reward, terminal=False)
    eers.append(testEER)
    weights.append(w)
    ind = np.argmin(eers)
    print(i, action, testEER, min(eers), weights[ind])
    state = action
ind = np.argmin(eers)
print(min(eers), weights[ind])