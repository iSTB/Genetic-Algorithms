import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import schematax as sx

def all_ones(indv):
    return indv.count('1')

with open('easy20.txt', 'rU') as kfile:
    lines = kfile.readlines()
    n = int(lines[0]) # Number of items
    c = int(lines[n+1]) # Knapsack capacity
    items = {int(line.split()[0]) : tuple(map(int, line.split()))[1:] for line in lines[1:n+1]} 

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
    if w >c:
        v = 0
      #  v = (1./((w-c)))*v
    return v
     
class GA(object):

    def __init__(self,func = all_ones,n=15,p=32,mu=0.005,s=0,e=False):
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


        self.pop = [] #current pop go here
        self.best = '' #stores best overall indivdual
        self.bests = [] #stores best individual from each gen
        self.best_f = -float('inf') #the fittness of the best overal indivdual
        self.bests_f = []
        self.av_f = [] #stores the average fitness of each population
        self.av_os = []
        self.av_ds = []
        self.av_fs = []
        self.schemas = [] # stores the schematic completion of each gen
        self.bb = [] #stores the building blocks of each gen
    def init_pop(self):
        self.pop = [''.join(str(random.choice(['1','0'])) for _ in xrange(self.p)) for _ in xrange(self.n) ]
       

    
    def mutate(self,indv):
        return ''.join(str(int(not(int(x)))) if random.random() <= self.mu else x for x in indv)


    def crossover(self,ma,da):

        #new = ''
        #for i in range(self.p):
        #    if ma[i] == da[i]:
        #        new += ma[i]
        #    else:
        #        new+= random.choice(['1','0']) 
        pivot = random.randint(0,self.p)

        #return [new]

    
        son = ma[:pivot] + da[pivot:]
        daught = da[:pivot] + ma[pivot:]

        return [son,daught]


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
        self.bests_f.append(max(self.fs.values()))
        self.bests.append(bestp)
        self.av_f.append(np.mean(self.fs.values()))



    def roulette_wheel(self):
        max = sum(self.fs.values())

        pick = random.uniform(0,max)

        current = 0

        for indv in self.fs.keys():
            current += self.fs[indv]
            if current > pick:
                return indv
        return indv

    def select(self):
        if self.s == 0:
            return self.roulette_wheel()


    def make_next_gen(self):
        self.eval_pop()

        new = []
        if self.e:
            new.append(self.bests[-1])
        while len(new) <= self.n:
            mum = self.select()
            dad = self.select()

            new +=  [self.mutate(x) for x in self.crossover(mum,dad)]
                        


        self.pop = new



    def get_bb(self):

        schemata = [s for s in self.schemas[-1] if not(s.is_empty_schema()) and s.get_anti_order() > 0]
        av_o = np.mean([s.get_order() for s in schemata])
        av_def = np.mean([s.get_defining_length() for s in schemata])
        av_fit = np.mean([s.fit for s in schemata])
        print av_o
        self.bb.append([s for s in schemata if s.fit >= av_fit and s.get_order() <= av_o and s.get_defining_length() <= av_def])
        self.av_fs.append(np.mean([s.fit for s in self.bb[-1]]))
        self.av_os.append(np.mean([s.get_order() for s in self.bb[-1]]))
        self.av_ds.append(np.mean([s.get_defining_length() for s in self.bb[-1]]))
    def run(self,steps=200):
        self.init_pop()
        self.perc = []
        self.schemas.append(sx.complete(self.pop, func =self.func))
        self.get_bb()

        for i in range(steps):
            print "gen: ",i
            self.make_next_gen()
            self.get_bb()
            #print self.bb
            self.schemas.append(sx.complete(self.pop, func=self.func))
            old_s,new_s = self.schemas[-2],self.schemas[-1]
            
            old_b, new_b = self.bb[-2], self.bb[-1] 

            bs = blends(old_b)
            c = 0
            for s in new_b:
                if s in bs:
                    c +=1
            if len(new_b) == 0:
                self.perc.append(0)
            else:
                self.perc.append((c*1.)/len(new_b))    

            #print self.bests
            #print self.best
def blends(schemata):
    bs = []
    c = 0
    for s1 in schemata:
        for s2 in schemata:
          #  print bs
            c +=1
            blend = ''
           # print s1,s2
            if s1.is_empty_schema() or s2.is_empty_schema():
                continue
            for i in xrange(len(s1)):
                if s1[i] == s2[i]: 
                    blend += s1[i]

                elif s1[i] == '*' and s2[i] != '*':
                        blend += s2[i]
                elif s2[i] == '*' and s1[i] != '*':
                        blend += s1[i]
            if len(blend) == len(s1):# and blend not in schemata:
                bs.append(blend)
    return list(set(bs))
if __name__ == "__main__":
    x = ['110101', '100100','011000', '110100', '011001', '111111']
    ss = sx.complete(x)

    print ss
    
    bls = blends(ss)
    sx.draw(bls + [sx.schema()], 'blends')
    sx.draw(ss, 'complete')
    
    
    
    
    
    
   # plt.plot(g.av_f)
   # plt.show()
   # fs = {}
    #for s in g.bb[-1]:
    #    o = s.get_order()
    #    f = s.fit
    #    print o,f
    #    if o not in fs.keys():
    #        fs[o] = (f,1.)
    #    else:
    #        fs[o] = (fs[o][0] + f, fs[o][1] +1.)

   # X = []
   # Y = []
   # for order in sorted(fs.keys()):
   #       X.append(order)
   #       Y.append(fs[order][0]/fs[order][1]) 
    
   # plt.plot(X,Y)
   # plt.show()
   # plt.plot(g.av_fs)
    #os = []
    #dls = []
    ps = []
    for x in range(5):
        g = GA(e=False) 
        g.run(100)
        ps.append(g.perc)
        #os.append(g.av_os)
        #dls.append(g.av_ds)

    #av_o = [np.average(col) for col in zip(*os)]
    #av_d = [np.average(col) for col in zip(*dls)]
    #o_er = [np.std(col) for col in zip(*os)]
    #d_er = [np.std(col) for col in zip(*dls)]
    av_p = [np.average(col) for col in zip(*ps)]
    p_er = [np.std(col) for col in zip(*ps)]
    plt.errorbar(range(len(av_p)),av_p,color='black',yerr=p_er )
    #plt.errorbar(range(len(av_o)),av_o,color='black',yerr=o_er )
    #plt.errorbar(range(len(av_d)), av_d,color='black',yerr=d_er,ls='--')
    plt.xlabel('Generation')
    plt.ylabel('Proportion of Building Blocks combined')
    matplotlib.rcParams.update({'font.size': 13})
    plt.show()
             


    #print g.schemas

