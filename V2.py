import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lenet_Copy1 as r
    # r.New() - новая пара
    # r.Mistake() - ошибки в паре
    # r.Graf() - ошибки в паре и график сложения по горизонтали
    # r.Hor(картинка)
    # r.Horizontal()
    # r.Plot
    
def Start(x,l):
#     return (-math.erf((x-l/4)/5)+1)/2
    if x > l/3: return 0
    else: return -(x-l/3)/(l/3)

def End(x,l):
#     return (math.erf((x-l/2)/5)+1)/2
    if x < l*2/3: return 0
    else: return (x-l*2/3)/(l*2/3)

def Gaus(x, μ, σ = 1):
    return (math.exp(-(x-μ)**2/(2*σ**2)))

def High(x,max_x):
    if x < (max_x/2): return 0
    else: return (x-max_x/2)/(max_x/2)
    
def Low(x,max_x):
    if x > (max_x/2): return 0
    else: return -(x-max_x/2)/(max_x/2)
    
def Summ(x,y):
    return max(x,y)

def Pow(x,y):
    return min(x,y)

def Zero(a):
    for i in range(len(a)):
        if a[i] < 1:
            x1=i
        else:
            break
    for i in range(len(a)-1, 0, -1):
        if a[i] < 1:
            x2=i
        else:
            break
    a=a[x1+1:x2]
    return a 
    
def Agrmax2(a):
    b = a.copy()
    b = sorted(b ,reverse = True)
    for i in range(len(a)):
        if a[i] == b[1]: 
            if i == 7:
#                 if b[1] > 0.1: 
#                     print(a)
#                     print(int(b[1]*100)/100)
                return i

class A:
    def __init__(self, y, y_false, x_false, n):
        self.y = y[n]
        self.y_false = y_false[n]
        self.x = x_false[n]
        self.hor = [[],[],[],[],[],[]]
        
        self.hor[0] = Zero(r.Hor(self.x)) #значения графика
        self.l = len(self.hor[0])
        
        self.hor[1] = np.zeros(self.l)  # В начале
        self.hor[2] = np.zeros(self.l)  # В конце
        self.hor[3] = np.zeros(self.l)  # Большое
        self.hor[4] = np.zeros(self.l)  # Маленькое
        self.hor[5] = np.zeros(self.l)  # 2/3
        
        for i in range(self.l):
#             self.hor[1][i] = Start(i, self.l)
            self.hor[1][i] = Gaus(i, 0, 2)
#             self.hor[2][i] = End(i, self.l)
            self.hor[2][i] = Gaus(i, self.l, 2)
            self.hor[3][i] = High(self.hor[0][i], max(self.hor[0]))
            self.hor[4][i] = Low(self.hor[0][i], max(self.hor[0]))
            self.hor[5][i] = Gaus(i, self.l*2/3, 1.5)
                   
    def P1(self):
        a=[]
        for i in range(self.l):
            a.append(Pow(self.hor[1][i], self.hor[3][i]))
            
        return a,max(a)
    
    def P2(self):
        a =[]
        for i in range(self.l):
            a.append(Pow(self.hor[5][i], self.hor[3][i]))
            
        return a,max(a)
    
    def P3(self):
        a =[]
        for i in range(self.l):
            a.append(Pow(self.hor[2][i], self.hor[4][i]))
            
        return a,max(a)
    
    def N1(self):
        a =[]
        for i in range(self.l):
            a.append(Pow(self.hor[5][i], self.hor[4][i]))
        return a,max(a)
    
    def N2(self):
        a =[]
        for i in range(self.l):
            a.append(Pow(self.hor[2][i], self.hor[3][i]))
        return a,max(a)
    
    def N(self):
        return min([self.N1()[1], self.N2()[1]])
    
    def P(self):
        return min([ self.P2()[1], self.P3()[1]])#self.P1()[1]
    
    def Plot(self, name = 'fig', n=18, m=10):
        plt.figure(figsize=(n,m))
        
        plt.subplot(3,3,1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.x, cmap=plt.cm.binary)
        
        plt.subplot(3,2,2)
        plt.plot(np.arange(self.l), self.hor[0], linewidth = 3,)
        plt.title('Сумма пикселей по горизонтали', fontsize = 20)
        plt.ylim([0,max(self.hor[0])+1])
        plt.tick_params(labelsize = 10)
        plt.xlabel('Номер строки', fontsize = 15)       
        plt.ylabel('Сумма пикселей', fontsize = 15 )
        
        fig.subplots_adjust(wspace=0, hspace=0.)
        plt.savefig(name, bbox_inches='tight')
        plt.show()
        
    def Plot2(self):
        plt.figure(figsize=(36,20))
        
        plt.subplot(3,3,1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.x, cmap=plt.cm.binary)
        plt.xlabel("Cеть-" + str(self.y_false) + "\n Цифра-" + str(self.y) )
        
        
        plt.subplot(3,2,2)
        plt.plot(np.arange(self.l), self.hor[0], linewidth = 5,)
        plt.title('Сумма пикселей по горизонтали', fontsize = 30)
        plt.ylim([0,max(self.hor[0])+1])
        plt.tick_params(labelsize = 30)
                 
        plt.subplot(3,2,3)
#         plt.plot(np.arange(self.l), self.P1()[0], label = 'P1')
        plt.plot(np.arange(self.l), self.P2()[0], label = '<<В середине много>>', linewidth = 5)
        plt.plot(np.arange(self.l), self.P3()[0], label = '<<В конце мало>>', linewidth = 5)
        plt.title('Возможность <<7>>', loc='right',fontsize = 30)
        plt.legend(fontsize = 25)
        plt.tick_params(labelsize = 30)
        
        plt.subplot(3,2,4)
        plt.plot(np.arange(self.l), self.N1()[0], label = '<<В середине мало>>', linewidth = 5)
        plt.plot(np.arange(self.l), self.N2()[0], label = '<<В конце много>>', linewidth = 5)
        plt.title('Возможность <<2>>', loc='right', fontsize = 30)
        plt.legend(fontsize = 25)
        plt.tick_params(labelsize = 30)
        
        plt.savefig('1',bbox_inches='tight')
        plt.show()

    def Data(self, name):
        f =open(str(name) + '.txt', 'w')
        f.write('x y\n')
        for i in np.arange(self.l):
            f.write(str(i) + ' ' + str(float(self.hor[0][i])) +'\n')
        f.close()
#Тест
count_chang = 0
predict = r.predict.copy()
pred = r.pred.copy()
count = []
for i in range(len(predict)):
    if predict[i] == 2 and Agrmax2(pred[i]) == 7 and pred[i][7] > 0.1:
        f = A(predict, predict, r.x_test, i)
        if f.P() > 2*f.N(): 
            predict[i] = 7
            count_chang += 1
            count.append(i)
            
mask = predict == r.y_test

y = r.y_test[~mask].copy()
x_false = r.x_test[~mask].copy()
y_false = predict[~mask].copy()
print(count_chang)

