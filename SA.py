import random
import numpy as np
import matplotlib.pyplot as plt
import schematax as sx
import struct



with open('easy20.txt', 'rU') as kfile:
    lines = kfile.readlines()
    n = int(lines[0]) # Number of items
    c = int(lines[n+1]) # Knapsack capacity
    items = {int(line.split()[0]) : tuple(map(int, line.split()))[1:] for line in lines[1:n+1]} 




def bitsToFloat(b):
#    s = struct.pack('>l', b)
    return struct.unpack('d', struct.pack('Q', int(b, 0)))[0]
    #return struct.unpack('f', b)[0]

def k_fit(b):
    if len(b) != n:
        print "error"
        return -1
    w = 0
    v = 0
    for i in range(n):
        if b[i] =='1':
            v += items[i+1][0] 
            w += items[i+1][1]
    #print v
    if w >c:
        v = (1./((w-c)))*v
  #  print v
    return v  
#print bitsToFloat('0b01000000001010010001011')


def booths(indv):
    
    x = int('0b' + indv[:int(len(indv)/2.0)],0)
    y = int('0b' + indv[int(len(indv)/2.0):],0)
  #  y_bits = struct.pack('>l', int('0b' + indv[int(len(indv)/2.0):],0))

  
#    x = struct.unpack('>f',x_bits)[0]
 #   y = struct.unpack('>f',y_bits)[0]
   
   # print x,y
#    print x,y
   # return -(x**2 + y**2)
   # z = (x+2*y-7)**2 + (2*x + y -5)**2
    z = x+y
    return -z


#print booths('10101010101010101010101010101010')


def all_ones(indv):
    return indv.count('1')

class SA(object):

    def __init__(self,func = k_fit,n=20,p=20,mu=0.00,s=0,e=False):
        """
        n: number of indvs
        p: number of bits for each indv
        mu: mutation rate
        s: selction type: 0 = roullete whel
        e: eleitism.
        """ 

        self.func = func
        self.n = n
        self.p = p
        self.mu = mu
        self.s = s
        self.e = e
        self.schemata =[]

        self.pop = [] #current pop go here
        self.best = '' #stores best overall indivdual
        self.bests = [] #stores best individual from each gen
        self.best_f = -float('inf') #the fittness of the best overal indivdual
        self.bests_f = []
        self.av_f = [] #stores the average fitness of each population
        self.max_f = 0 # stores the best max fitness of the best schema  

    def init_pop(self):
        self.pop = [''.join(str(random.choice(['1','0'])) for _ in xrange(self.p)) for _ in xrange(self.n) ]
       
    def mutate(self,indv):
        return ''.join(str(int(not(int(x)))) if random.random() <= self.mu else x for x in indv)

    def eval_pop(self):
        self.fs = {}

        bestp = ''
        bestpf = -float('inf')
        for indv in self.pop:
            f = self.func(indv)
            self.fs[indv] = f

            if f > self.best_f:
                self.best = indv
                self.best_f = f

            if  f > bestpf:
                bestp = indv
                bestpf = f

        self.bests.append(bestp)
        self.bests_f.append(bestpf)
        self.av_f.append(np.mean(self.fs.values()))


    #def select sche

    def select(self):
        #roulete wheel selection
        av_o = np.mean([s.get_anti_order() for s in self.schemata if not(s.is_empty_schema())])
        print av_o
        schemata = [s for s in self.schemata if (not s.is_empty_schema()) and s.get_anti_order() >= np.floor(av_o) ]
        max_f = max([s.fit for s in schemata])*1.
        min_f = min([s.fit for s in schemata])


      #  print max_f,min_f
        for s in schemata:
            if max_f == min_f:
                s.fit = 0.1
            else:
                s.fit = (s.fit - min_f)/(max_f-min_f)
            #print s.fit
        self.max_f = sum([s.fit for s in schemata])
        
        pick = random.uniform(0,self.max_f)
        current = 0

        for indv in schemata:
            current += indv.fit
            if current >= pick:
                return indv
        return indv     
   # def get_good_schemata(self):
   #     meanf = np.mean([s for s in self.schemata])
   #     meano = np.mean([s.get_order() for s in selfschemata])

    #    self.good = [s for s in self.schemata if s.get_order() >= meano and all_ones(s) >= meanfit]
    
            


    def make_next_gen(self):
        self.eval_pop()
        self.schemata = sx.complete(self.pop,func = self.func)
        new = []
        if self.e:
            new.append(self.bests[-1])
        while len(new) <= self.n:
            mum = self.select()
            dad = self.select()
            
           # o1 = ''
           # for i in xrange(self.p):
           #     if mum[i] == '*':
           #         o1 += random.choice(['1','0'])
          #      else:
           #         o1 += mum[i]
           # new += [self.mutate(o1)]
            son = ''
            for i in xrange(self.p):
                if mum[i] == dad[i] and mum[i] != '*':
                    son += mum[i]
                elif mum[i] != dad[i] and dad[i] == '*':
                    son += mum[i]
                elif mum[i] != dad[i] and mum[i] == '*':
                    son += dad[i]
                else:
                    son += random.choice(['1','0'])


            new +=  [self.mutate(son)]
        self.pop = new

    def run(self,steps=100):
       # self.init_pop()
        for i in range(steps):
            print "gen: ",i
            self.make_next_gen()
            print self.pop
          #  print self.bests
            print self.best


if __name__ == "__main__":
    from GA import GA
    ss = []
    gs = []
    for i in range(10):
        pop = [''.join(str(random.choice(['1','0'])) for _ in xrange(20)) for _ in xrange(20) ]
        s = SA(e=True,func=k_fit,n=20,p=20,mu=0.0) 
        s.pop = pop
        s.run(100)
        g = GA(func = k_fit,n=20, p =20,e=True, mu=0.0)
        g.pop = pop
        g.run(100)
        ss.append(s.bests_f)
        gs.append(g.bests_f)
        #plt.plot(s.av_f)
        #plt.plot(g.av_f)
    #    plt.plot(s.bests_f)
    #    plt.plot(g.bests_f)
    #    plt.show()
   # pass

    ss = reduce(np.add,ss)
    gs = reduce(np.add,gs)
    plt.plot(map(lambda x: x/10., ss))
    plt.plot(map(lambda x: x/10., gs))
    plt.show()
#    plt.plot(g.bests_f)
#    plt.show()
