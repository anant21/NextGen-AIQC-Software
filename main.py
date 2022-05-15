from os import system, environ
system("clear")

def stopMessage():
    print("\n[*!*] Keyboard Interruption: Forcely STOPPED !!")
    print("\n\n   ########################################")
    print("   ****************************************")      
    print('\n              TOOL STOPPED!!')
    print("\n   ****************************************")    
    print("   ########################################\n")    
    exit(0)

print("""
   <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

    ######  ##       #######  ######    ########  #####  #####  ## 
    ##  ##  ##  ###  ##   ##  ##           ##     ## ##  ## ##  ##
    ######  ##       ## ####  ##           ##     ## ##  ## ##  ##
    ##  ##  ##       ####### #######      ##     #####  #####  ######
                          ##
   <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    """)
print("\n[*] Importing modules ...")

try:
    print("\n   [+] Face Recognition Modules")
    import cv2
    import face_recognition

    print("   [+] Numpy Modules")
    import numpy as np
    from numpy.random import randint

    print("   [+] Torch Modules")
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function
    from torchvision import datasets, transforms

    print("   [+] Time, Geopy, Warnings, PIL, Crypto Modules")
    import time
    import warnings
    import matplotlib.pyplot as plt
    from datetime import datetime
    from PIL import ImageGrab
    try:
        from geopy.geocoders import Nominatim
    except:
        pass
    from Crypto import Random
    from Crypto.Cipher import AES

    print("   [+] Qiskit Libraries")
    import qiskit
    from qiskit import transpile, assemble
    from qiskit.visualization import *
    from qiskit import QuantumCircuit, Aer, transpile, assemble


    print("\n[=>] Import successful.\n")

except KeyboardInterrupt:
    stopMessage()

#Suppressing unnecessary warnings
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

warnings.filterwarnings('ignore')
suppress_qt_warnings()

#Function to get Timestamp
def gettimestamp():
    now = datetime.now()
    dtString = now.strftime("%c")
    return dtString


#Quantum Circuits
class QuantumCircuitML:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        
        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])

# Creating Quantum-Classical Class with pytorch
class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuitML(1, backend, shots)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)

# Creating Hybrid Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)


print("\n[*] Starting the tool ...")
time.sleep(0.5)

# Camera Location
try:
    print("\n[*] Fetching device location ...")
    loc = Nominatim(user_agent="GetLoc")
    geoLoc = loc.geocode("28.7197058,77.0660904")
    print("   [+]", geoLoc)
    print("\n[=>] Location fetched.\n")
except KeyboardInterrupt:
    stopMessage()
    

# Implementation testing
print("\n[*] Model-Accuracy testing using MNIST dataset ...")
print("\n   [+] Implementation testing")
simulator = qiskit.Aer.get_backend('aer_simulator')
circuit = QuantumCircuitML(1, simulator, 100)
print('      [~] Expected value for rotation pi: {}'.format(circuit.run([np.pi])[0]))
print('      [!] Quantum Circuit Diagram: use Jupyter Notebook')
circuit._circuit.draw()


# Data Loading and Preprocessing - (MNIST Dataset for Testing)
## Concentrating on the first 100 samples
print("   [+] Data Loading and Preprocessing")
n_samples = 100
X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

## Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)


# Testing data
print("   [+] Testing data")
n_samples = 50
X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)



# Training the network
try:
    print("   [+] Training the network")
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.NLLLoss()

    epochs = 20
    loss_list = []

    model.train()
    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculating loss
            loss = loss_func(output, target)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()
            
            total_loss.append(loss.item())
        loss_list.append(sum(total_loss)/len(total_loss))
        print('      [~] Training [{:.0f}%]          Loss: {:.4f}'.format(
            100. * (epoch + 1) / epochs, loss_list[-1]))
except KeyboardInterrupt:
    stopMessage()

# Training graph plot
plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Neg Log Likelihood Loss')


# Testing the network
print('   [+] Testing the network')
model.eval()
with torch.no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
        
    print('\n   [%] Performance on test data:\n\t\tLoss: {:.4f}\n\t\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )

print('\n[=>] Model-Accuracy calculation successful.\n')


# Training Images - (Actual Data for usage)
print('\n[*] Making the tool ready for usage ...')
print("\n   [+] Training new images")
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Image Encoding
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Encryption Functions
def pad(s):
    return s + b"\0" * (AES.block_size - len(s) % AES.block_size)

def encrypt(message, key, key_size=256):
    message = pad(message)
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(message)

def encrypt_file(file_name, key):
    with open(file_name, 'rb') as fo:
        plaintext = fo.read()
    enc = encrypt(plaintext, key)
    with open(file_name + ".enc", 'wb') as fo:
        fo.write(enc)

def decrypt(ciphertext, key):
    iv = ciphertext[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext[AES.block_size:])
    return plaintext.rstrip(b"\0")

def decrypt_file(file_name, key):
    with open(file_name, 'rb') as fo:
        ciphertext = fo.read()
    dec = decrypt(ciphertext, key)
    with open("bob_"+file_name[:-4], 'wb') as fo:
        fo.write(dec)


# Quantum Encryption Functions
def encode_message(bits, bases):
    message = []
    for i in range(n):
        qc = QuantumCircuit(1,1)
        if bases[i] == 0: # Prepare qubit in Z-basis
            if bits[i] == 0:
                pass 
            else:
                qc.x(0)
        else: # Prepare qubit in X-basis
            if bits[i] == 0:
                qc.h(0)
            else:
                qc.x(0)
                qc.h(0)
        qc.barrier()
        message.append(qc)
    return message

def measure_message(message, bases):
    backend = Aer.get_backend('aer_simulator')
    measurements = []
    for q in range(n):
        if bases[q] == 0: # measuring in Z-basis
            message[q].measure(0,0)
        if bases[q] == 1: # measuring in X-basis
            message[q].h(0)
            message[q].measure(0,0)
        aer_sim = Aer.get_backend('aer_simulator')
        #qobj = assemble(message[q], shots=1, memory=True)
        
        result = aer_sim.run(message[q],shots=1, memory=True).result()
        measured_bit = result.get_memory(message[q])
        measurements.append(int(measured_bit[0]))
    return measurements

def remove_garbage(a_bases, b_bases, bits):
    good_bits = []
    for q in range(n):
        if a_bases[q] == b_bases[q]:
            # If both used the same basis, add
            # this to the list of 'good' bits
            good_bits.append(bits[q])
    return good_bits

def sample_bits(bits, selection):
    sample = []
    for i in selection:
        # use np.mod to make sure the
        # bit we sample is always in 
        # the list range
        i = np.mod(i, len(bits))
        # pop(i) removes the element of the
        # list at index 'i'
        sample.append(bits.pop(i))
    return sample

def qAlice_key(n):
    
    alice_bits = randint(2, size=n)
    message = encode_message(alice_bits, alice_bases)
    alice_key = remove_garbage(alice_bases, bob_bases, alice_bits)
    return (alice_key, message)

def qBob_key(n):
    bob_results = measure_message(message, bob_bases)
    bob_key = remove_garbage(alice_bases, bob_bases, bob_results)
    return bob_key


print("   [+] Encoding images")
encodeListKnown = findEncodings(images)
print('   [+] Encoding Completed')
print("\n[=>] Device is ready to use.\n")

try:
    print("\n[*] Video Capturing ...")
    cap = cv2.VideoCapture(0)
except:
    print('\n[x] Can\'t open camera!')
    exit(0)

try:
    while True:
        success, img = cap.read()
        
        print("\n   ----------------------------")
        print("\n[*] Identifying the suspect ...")
            
        try:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        except:
            print("\n[x] Video Capture failed: Connect the camera.\n")
            exit(0)

        time.sleep(0.5)
        print("\n   [+] Matching in database")

        suppress_qt_warnings()

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print("   [+] Face Distance:", faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print("\n[=>] Match found!\n")

                print("\n[*] @Suspect Details:")
                timestamp = gettimestamp()
                location = geoLoc
                details = name[:3]+", "+name[4:-1]+", "+str(location.address)+", "+str(timestamp)
                
                print("\n   [~] Name:",name[4:-1])
                print("   [~] ID:",name[:3])
                print("   [~] Location:",location.address)
                print("   [~] Coordinates:",location[1])
                print("   [~] Time:",timestamp)

                ## Encrypt data
                print("\n\n[*] Data Quantum Encryption ...")

                file1 = open("to_enc.txt", "a")
                file1.write(details)
                file1.write("\n")
                file1.close()

                print("\n   [+] Quantum Key Distribution")
                np.random.seed(seed=0)
                n = 256

                alice_bases = randint(2, size=n)
                bob_bases = randint(2, size=n)

                alice, message = qAlice_key(n)
                bob = qBob_key(n)
                
                sample_size = 35
                bit_selection = randint(n, size=sample_size)

                alice_samp = sample_bits(alice,bit_selection)
                bob_samp = sample_bits(bob,bit_selection)

                if (alice_samp == bob_samp):

                    print("   [+] Key distribution successful")

                    kk = ""
                    for i in alice:
                        kk += str(i)

                    key = hex(int(kk,2))
                    key = key.encode('UTF-8')
                    key = pad(key)

                    encrypt_file('to_enc.txt', key)
                    print("\n[=>] Data Encryption using Quantum Cryptography successful!\n")

                    print("\n[*] Transferring encrypted data to Bob ...")
                    print("\n   [+] Data reached")
                    print("   [+] Data decryption started")

                    decrypt_file('to_enc.txt.enc', key)
                    print("   [+] Data decryption successful")
                    print("\n[=>] Data transfer successful.")

                    print("\n\n[*] Reading Logs File:\n    ----------")
                    file2 = open("bob_to_enc.txt", "r")
                    print(file2.read())
                    file2.close()

                
                else:
                    print("\n[!x!]> Interception detected.")


                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name[4:-1], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    stopMessage()

finally:    
    print("\n\n[#] THANKS FOR USING NEXT-GEN Quantum Crime Surveillance Tool!\n") 
