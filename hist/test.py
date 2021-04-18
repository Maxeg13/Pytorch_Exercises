class A:
    def __init__(self):
        self.x=10
        
        
def func(A):
    A.x=11
    
a=A()

func(a)
print(a.x)