import numpy as np
import matplotlib.pyplot as plt
import CTRNN

size = 10
duration = 100
stepsize = 0.01

time = np.arange(0.0,duration,stepsize)
outputs = np.zeros((len(time),size))
states = np.zeros((len(time),size))

nn = CTRNN.CTRNN(size)
nn.load("ctrnnActiveThenNot.npz")
nn.modifyTimeConstants(1.5)

step = 0
for t in time:
    nn.step(stepsize)
    states[step] = nn.states
    outputs[step] = nn.outputs
    step += 1

activity = np.sum(np.abs(np.diff(outputs,axis=0)))/(duration*size*stepsize)
print("Overall activity: ",activity)

plt.plot(time,outputs)
plt.xlabel("Time")
plt.ylabel("Outputs")
plt.title("Neural output activity")
#plt.show()

plt.plot(time,states)
plt.xlabel("Time")
plt.ylabel("States")
plt.title("Neural state activity")
#plt.show()

nn.save("Experiment1a")
#nn.showTimeConstants()