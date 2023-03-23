import numpy as np

D1 = np.random.randint(1,7)
D2 = np.random.randint(1,7)

print("Dice 1 and Dice 2 have results {} and {} respectively".format(D1,D2))

Trials = int(1e6)
D1 = np.random.randint(1,7,size = Trials)
D2 = np.random.randint(1,7,size = Trials)

count = 0
for i in range(Trials):
    if (D1[i]==6 and D2[i]==6):
        count += 1
print("Probability of getting 6 on both Dice is {} \n The theoretical probability is {}".format(count/Trials,1/36))



        
