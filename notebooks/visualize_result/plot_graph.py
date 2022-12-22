import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib import rc
from pathlib import Path
plt.style.use('classic')
plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=8)
# plt.rc('axes', labelsize=8)

print(os.getcwd())
f_name = "./results/log_yolov7_after.csv"
# f_name = "./results/yolov5w09.csv"
f_name = Path(f_name)
frame = pd.read_csv(f_name,)
print(frame.columns)
fitness_series = frame.composite_fitness.values
map05_series   = frame.mAP05
map95_series   = frame.mAP95
N = 40
max_fitness_value = np.max(fitness_series[N:]) 
arg_max           = np.argmax(fitness_series[N:]) + N
fig = plt.figure(dpi = 200, figsize = (8,6))
length = len(fitness_series)
plt.plot(fitness_series, marker = '.', color = 'black', label = 'Взвешенный критерий')
plt.plot(map95_series, label = 'mAP0.95', marker = 'o')
plt.plot(map05_series, label = 'mAP0.5',  marker = '.')
plt.plot(arg_max, max_fitness_value, marker = '*', color = 'red', markersize = 12)
plt.legend(loc = 4,fontsize = 12)
plt.axvline(arg_max, alpha = 0.5)

ax = fig.gca()
ax.set_ylabel('Значение', fontsize = 14 )
ax.set_xlabel('Эпоха', fontsize = 14)
ax.set_ylim(0.3,1)
ax.set_xlim(0, length)
ax.set_xticks(np.arange(0, length, 10))
plt.grid()
plt.margins(x=0, y= 0)
fig.savefig('./results/' + f_name.stem + '.jpg', transparent=True)

plt.show()