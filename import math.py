import math 
x = 1 
term = 0 
result = 0 
n = 7
while term <= n: 
    result += (x**term) / math.factorial(term) 
    term += 1 