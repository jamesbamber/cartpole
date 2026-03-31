def euler(x, y, f, dx, a):
    return x + dx, y + dx*f(x, y, a)

def rk4(x, y, f, dx, a): 
    k1 = f(x, y, a)
    k2 = f(x + dx/2, y + dx/2 * k1, a)
    k3 = f(x + dx/2, y + dx/2 * k2, a)
    k4 = f(x + dx, y + dx * k3, a)
    return x + dx, y + dx/6 * (k1 + 2 * k2 + 2 * k3 + k4)
