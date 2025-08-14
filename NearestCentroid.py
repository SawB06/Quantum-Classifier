# --------------------
#  NearestCentroid.py
# --------------------
# Implements a QML nearest centroid classifier

# Dataset: MNIST digits (currently subset: 0, 1)
# Dimensionality: Reduced to 8D using PCA.
# Quantum Backend: Several options available (IonQ simulator, IonQ Aria1, default qubit, etc.)


# ============
#  1. Imports
# ============
import pennylane as qml
import numpy as np
from sklearn.datasets import fetch_openml 
from sklearn.decomposition import PCA 


# =====================
#  2. Static Variables
# =====================
CLASSES = [1,0]         
TESTPERCLASS = 10


# =====================
#  3. Data Preparation
# =====================
mnist = fetch_openml('mnist_784', version=1) 
data = mnist.data.astype(np.float32)
data /= 255.0
labels = mnist.target.astype(int)
pca = PCA(n_components=8)
datasmall = pca.fit_transform(data)
datasmall -= datasmall.min(axis=0, keepdims=True)
mask = np.isin(labels, CLASSES)
filtered_data = datasmall[mask]
filtered_labels = labels[mask]
class_indices = {c: np.where(filtered_labels == c)[0] for c in CLASSES}


# ================================
#  4. Prepare Test and Train Sets
# ================================
testIndex_list = []
train_indices = {}
for c in CLASSES:
    index = class_indices[c]
    take = min(TESTPERCLASS, len(index))
    test_c = np.random.choice(index, take, replace=False)
    testIndex_list.append(test_c)
    train_indices[c] = np.setdiff1d(index, test_c)
testIndex = np.concatenate(testIndex_list)
testSubjects = filtered_data[testIndex]                     
centroids = {c: filtered_data[train_indices[c]].mean(axis=0) for c in CLASSES}
centroid_norms = {c: np.linalg.norm(centroids[c]) for c in CLASSES}


# ===================
#  5. Angle Encoding 
# ===================
def getThetas(x):  
    x = np.array(x)
    x_norm = np.linalg.norm(x)
    x = x / x_norm
    r7 = np.sqrt(x[7]**2 + x[6]**2)
    r6 = np.sqrt(x[5]**2 + x[4]**2)
    r5 = np.sqrt(x[3]**2 + x[2]**2)
    r4 = np.sqrt(x[1]**2 + x[0]**2)
    r3 = np.sqrt(r7**2 + r6**2)
    r2 = np.sqrt(r5**2 + r4**2)
    r1 = np.sqrt(r3**2 + r2**2)
    thetas = np.zeros(7)
    thetas[0] = 0 if r1 == 0 else np.arccos(r3 / r1)
    thetas[1] = 0 if r3 == 0 else np.arccos(r7 / r3)
    thetas[2] = 0 if r2 == 0 else np.arccos(r5 / r2)
    for i in range(3, 7):
        a = 2 * (i - 3)
        b = a + 1
        r = np.sqrt(x[a]**2 + x[b]**2)
        thetas[i] = 0 if r == 0 else np.arccos(x[a] / r) if x[b] >= 0 else 2 * np.pi - np.arccos(x[a] / r)
    return thetas, x_norm


# ========================
#  6. Custom Unitary Gate
# ========================
def rbs(theta):
    return np.array([
        [1, 0,           0,          0],
        [0, np.cos(theta), np.sin(theta), 0],
        [0, -np.sin(theta), np.cos(theta), 0],
        [0, 0,           0,          1]
    ])


# =========================
#  7. Quantum Device Setup
# =========================
#dev = qml.device("ionq.simulator",wires=8,shots=500,api_key="YOUR_API_KEY") 
#dev = qml.device("ionq.qpu",wires=8, shots=150,api_key="YOUR_API_KEY")
#dev = qml.device("default.qubit", wires=8)
dev = qml.device("lightning.qubit", wires=8)


# ================================
#  8. Quantum Distance Estimation
# ================================
@qml.qnode(dev)
def estimateDistance(x,y):
    thetasX,n = getThetas(x)
    thetasY,n = getThetas(y)
    qml.PauliX(wires=0)
    qml.QubitUnitary(rbs(thetasX[6]), wires=[0, 4])  
    qml.QubitUnitary(rbs(thetasX[5]), wires=[0, 2])  
    qml.QubitUnitary(rbs(thetasX[4]), wires=[4, 6])  
    qml.QubitUnitary(rbs(thetasX[3]), wires=[0, 1])  
    qml.QubitUnitary(rbs(thetasX[2]), wires=[2, 3])  
    qml.QubitUnitary(rbs(thetasX[1]), wires=[4, 5])  
    qml.QubitUnitary(rbs(thetasX[0]), wires=[6, 7]) 
    qml.QubitUnitary(rbs(thetasY[0]).T, wires=[6, 7]) 
    qml.QubitUnitary(rbs(thetasY[1]).T, wires=[4, 5]) 
    qml.QubitUnitary(rbs(thetasY[2]).T, wires=[2, 3]) 
    qml.QubitUnitary(rbs(thetasY[3]).T, wires=[0, 1])  
    qml.QubitUnitary(rbs(thetasY[4]).T, wires=[4, 6])  
    qml.QubitUnitary(rbs(thetasY[5]).T, wires=[0, 2])  
    qml.QubitUnitary(rbs(thetasY[6]).T, wires=[0, 4])  
    return qml.probs(wires=0)


# ================================
#  9. Nearest Centroid Classifier
# ================================
def classify(sample):
    normX = np.linalg.norm(sample) 
    dists = {}
    for c in CLASSES:
        p = estimateDistance(sample, centroids[c])  
        p1 = float(np.clip(p[1], 0.0, 1.0))         
        cosXY = np.sqrt(p1)                           
        normY = centroid_norms[c]
        dist = np.sqrt(normX**2 + normY**2 - 2.0*normX*normY*cosXY)
        dists[c] = dist
    return min(dists, key=dists.get)



# ========================
#  10. Classifier Results
# ========================
y_true, y_pred = [], []
correct = 0
for index in testIndex:
    correctAnswer = filtered_labels.iloc[index]
    sample = filtered_data[index]
    pred = classify(sample)
    print(f"True: {correctAnswer}, Pred: {pred}")
    y_true.append(correctAnswer)
    y_pred.append(pred)
from sklearn.metrics import classification_report, accuracy_score

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.3%}\n")
print(classification_report(y_true, y_pred, digits=3))