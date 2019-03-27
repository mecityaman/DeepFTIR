import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import pandas as pd

chemicals = pd.read_pickle('chemicals.pkl')
xl = 400; xu = 4000; nx = 901;
x = np.linspace(xl,xu,nx) # Define the domain

n_factor = 0.5# Noise magnitude
def GenNoise(S): # Noise generating function
      n = np.random.randn(len(S))
      N = S+n_factor*n
      return N

# Create the Dataset
nts = 1000  # Number of train signals
ntests = 1 # Number of test signals
npulse = 1;  # Number of pulses signals
base = 1;    # Base value of signals
nc = 5    # Number of Chemicals
train_signals = np.zeros([nc*nts,nx],dtype=np.float32)
train_signals_noised = np.zeros([nc*nts,nx],dtype=np.float32)
test_signals = np.zeros([nc*ntests,nx],dtype=np.float32)
test_signals_noised = np.zeros([nc*ntests,nx],dtype=np.float32)


Signal_1 = np.zeros([nc,901],dtype=np.float32)
for i in range(nc):
      Signal_1[i] = chemicals[i]
      plt.plot(x,Signal_1[i])

plt.xlabel('Sample(n)')
plt.ylabel('Reading(V)')
plt.show()

for i in range(nts):
      for j in range(nc):
            train_signals[j*nts+i] = Signal_1[j]
#            train_signals_noised[j*nts+i] = GenNoise(train_signals[j*nts+i])

for i in range(len(train_signals)):
      train_signals_noised[i] = train_signals[i]

for i in range(ntests):
      for j in range(nc):
          test_signals[j*ntests+i] = Signal_1[j]
          test_signals_noised[j*ntests+i] = GenNoise(test_signals[j*ntests+i])
          plt.plot(x,test_signals_noised[j])
st = np.std(train_signals_noised)
sm = np.mean(train_signals)

"""
AutoEncoder
"""
tf.reset_default_graph()

num_inputs=nx
num_hid1=int(nx/2)
num_hid2=int(nx/4)
num_hid2_n = int(nx/8)
num_hid3_n = int(nx/4)
num_hid3=num_hid1
num_output=num_inputs
lr=0.0005  # Learning rate
ep = 2e-6 # Epsilon, rate of decrease of lr
actf=tf.nn.relu # rectified linear unit, deactivated for output layer

X=tf.placeholder(tf.float32,shape=[None,num_inputs],name = 'x')
Y=tf.placeholder(tf.float32,shape=[None,num_inputs],name = 'y')

initializer=tf.variance_scaling_initializer()

# wX=b
w1=tf.Variable(initializer([num_inputs,num_hid1]),dtype=tf.float32)
w2=tf.Variable(initializer([num_hid1,num_hid2]),dtype=tf.float32)
w3=tf.Variable(initializer([num_hid2,num_hid2_n]),dtype=tf.float32)
w4=tf.Variable(initializer([num_hid2_n,num_hid3_n]),dtype=tf.float32)
w5=tf.Variable(initializer([num_hid3_n,num_hid3]),dtype=tf.float32)
w6=tf.Variable(initializer([num_hid3,num_output]),dtype=tf.float32)

b1=tf.Variable(tf.zeros(num_hid1))
b2=tf.Variable(tf.zeros(num_hid2))
b2_n=tf.Variable(tf.zeros(num_hid2_n))
b3_n=tf.Variable(tf.zeros(num_hid3_n))
b3=tf.Variable(tf.zeros(num_hid3))
b4=tf.Variable(tf.zeros(num_output))

hid_layer1=actf(tf.matmul(X,w1)+b1)
hid_layer2=actf(tf.matmul(hid_layer1,w2)+b2)
hid_layer2_n = actf(tf.matmul(hid_layer2,w3)+b2_n)
hid_layer3_n=  actf(tf.matmul(hid_layer2_n,w4)+b3_n)
hid_layer3=actf(tf.matmul(hid_layer3_n,w5)+b3)
output_layer=tf.matmul(hid_layer3,w6)+b4

loss=tf.reduce_mean(tf.square(output_layer-X)) # UNSupervised because of X
#loss=tf.reduce_mean(tf.square(output_layer-Y)) # Supervised because of Y

optimizer=tf.train.AdamOptimizer(lr,epsilon = ep)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()
print('Signal to Noise ratio is ',sm/st)
num_epoch=1000
#batch_size=1000
num_test_images=5
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(num_epoch):
        X_batch = train_signals_noised
        y_batch = train_signals
        sess.run(train,feed_dict={X:X_batch,Y:y_batch})
        train_loss=loss.eval(feed_dict={X:X_batch,Y:y_batch})
        print("epoch {} loss {}".format(epoch,train_loss))

    results=output_layer.eval(feed_dict={X:test_signals_noised[:num_test_images] ,Y:test_signals[:num_test_images]})
#Plotter
f, ax = plt.subplots(num_test_images,3, sharey=True)
for i in range(num_test_images):
#          ax[i][0].plot(x,test_signals[i],x,test_signals_noised[i],alpha = 0.7)
          ax[i][0].plot(x,test_signals_noised[i],alpha = 0.7)
          ax[i][0].plot(x,test_signals[i],alpha = 0.7)
          ax[i][1].plot(x,results[i])
          ax[i][2].plot(x,test_signals[i],label = 'S')
          ax[i][2].plot(x,results[i],label = 'F')

          ax[i][0].set_xlabel('Sample')
          ax[i][1].set_xlabel('Sample')
          ax[i][2].set_xlabel('Sample')

          ax[i][1].set_ylabel('F')
          ax[i][2].set_ylabel('S & F')
          ax[i][0].set_ylabel('S & N')
          ax[i][2].legend()

          ax[i][0].grid()
          ax[i][1].grid()
          ax[i][2].grid()

ax[0][0].set_title('Original Signal & Noise')
ax[0][1].set_title('Filtered Signal')
ax[0][2].set_title('Original Signal & Filtered Signal')
print('Signal to Noise ratio is ',sm/st)