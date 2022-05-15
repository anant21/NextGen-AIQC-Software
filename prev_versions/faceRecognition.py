import os

os.system("clear")

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

import cv2
import numpy as np
import face_recognition
import time
from numpy.random import randint
from datetime import datetime
from PIL import ImageGrab
from geopy.geocoders import Nominatim
from os import environ
from Crypto import Random
from Crypto.Cipher import AES
from qiskit import QuantumCircuit, Aer, transpile, assemble

print("\n[=>] Import successful.\n")

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


print("\n[*] Starting up the tool ...")
time.sleep(0.5)

print("\n[*] Fetching device location ...")

# Camera Location
loc = Nominatim(user_agent="GetLoc")
geoLoc = loc.geocode("28.7197058,77.0660904")

print("\n[=>] Location fetched.\n")

def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("\n[*] Encoding images ...")
encodeListKnown = findEncodings(images)
print('\n[=>] Encoding Completed.\n')
print("\n[#] Device is ready!\n")


def gettimestamp():
    now = datetime.now()
    dtString = now.strftime("%c")

    return dtString
    

cap = cv2.VideoCapture(0)

# Encryption functions

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


# Quantum Encryptions
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


try:
    while True:
        success, img = cap.read()
        #img = captureScreen()
        
        print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("\n[*] Identifying the suspect ...")
            
        try:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        except:
            print("\n[?] Video Capture failed: Connect the camera.\n")
            exit(0)

        time.sleep(0.5)
        print("[*] Matching in database ...")

        suppress_qt_warnings()

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print("[*] Face Distance:", faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print("\n[#] Match found!\n")

                print("Details"+"\n=======")
                timestamp = gettimestamp()
                location = geoLoc

                details = name[:3]+", "+name[4:-1]+", "+str(location.address)+", "+str(timestamp)
                
                print("Name:",name[4:-1])
                print("ID:",name[:3])
                print("Location:",location.address)
                print("Coordinates:",location[1])
                print("Time:",timestamp)

                ## Encrypt data
                print("\n[*] Encrypting data ...")
                time.sleep(0.7)

                file1 = open("to_enc.txt", "a")
                file1.write(details)
                file1.write("\n")
                file1.close()

                np.random.seed(seed=0)
                n = 256

                alice_bases = randint(2, size=n)
                bob_bases = randint(2, size=n)

                alice, message = qAlice_key(n)
                bob = qBob_key(n)

                # print("Alice:",len(alice))
                
                sample_size = 35
                bit_selection = randint(n, size=sample_size)

                alice_samp = sample_bits(alice,bit_selection)
                bob_samp = sample_bits(bob,bit_selection)

                # print("Alice:",len(alice))


                if (alice_samp == bob_samp):

                    print("\n[=>] Key distribution done successfully.")
                    time.sleep(0.5)        

                    kk = ""
                    for i in alice:
                        kk += str(i)

                    key = hex(int(kk,2))
                    key = key.encode('UTF-8')
                    key = pad(key)

                    encrypt_file('to_enc.txt', key)
                    print("\n[=>] Data quantum cryptographically encrypted!")

                    print("\n[*] Transferring data to Bob ...")
                    print("\n[*] Decrypting data ...")

                    decrypt_file('to_enc.txt.enc', key)

                    time.sleep(0.7)
                    print("\n[=>] Data decrypted successfully!")

                    print("\n\nLog Data:\n========")
                    file2 = open("bob_to_enc.txt", "r")
                    print(file2.read())
                    file2.close()

                
                else:
                    print("[!!]> Interception detected.")


                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name[4:-1], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
except KeyboardInterrupt:
    print("\n\n########################################")
    print("****************************************")      
    print('\n             TOOL STOPPED!!')
    print("\n****************************************")    
    print("########################################\n")    
