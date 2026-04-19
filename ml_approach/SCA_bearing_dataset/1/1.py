from scipy.io import loadmat

# Load file
data = loadmat("train.mat")

# Saare keys (variables) dekho
print(data.keys())

# Example: kisi variable ko access karo
signal = data['faultType']

print(type(signal))
print(signal.shape)
