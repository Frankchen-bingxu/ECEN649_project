import numpy as np
import matplotlib.pyplot as plt
x1=[1,3,5,10]
y1=[0.36,0.38,0.76,0.82]
x2=[1,3,5,10]
y2=[0.13,0.07,0.01,0.005]
x3=[1,3,5,10]
y3=[0.68,0.62,0.82,0.9]
x=np.arange(20,350)
l1=plt.plot(x1,y1,'r--',label='Total accuracy')
l2=plt.plot(x2,y2,'g--',label='False Positive')
l3=plt.plot(x3,y3,'b--',label='False Negative')
plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
plt.title('The change in the system')
plt.xlabel('round')
plt.ylabel('percent')
plt.legend()
plt.show()
