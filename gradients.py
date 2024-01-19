import jax

def f1(x, y):
    q = x**2
    z = q**3 + 5*x*y
    return z

def f2(x, y): 
    z=1
    for i in range(int(y)): 
        z *= (x+float(i))
    return z

def f3(x, y):
    return x**y

df1 = jax.grad(f1, argnums=0)
df2 = jax.grad(f2, argnums=1)
df3 = jax.grad(f3, argnums=[0,1])

print(df1(1., 2.))
print(df2(1., 2.))
print(df3(1., 2.))