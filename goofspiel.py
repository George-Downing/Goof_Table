import math
import numpy
import torch
import matplotlib
import matplotlib.pyplot

# strategy_hyperplane_uniform: rand-N-times is wrong!
# generating y[0], y[1], ... y[D-1], sum(y)==1 is different from rand-N-times and divide by its sum
# because rand-N-times samples near centroid are denser than others
# take an example, 2D: y[0] + y[1] == 1
# if rand-N-times is used, then probability density would be f(x):={0<=x<0.5: 2*x, 0.5<=x<=1: 4-2*x, else: 0}
# although points {y | sum(y)<=1} are still in hyperpyramid
# which is because points {y | sum(y)>1} (not from hyperpyramid) unlike points in hyperpyramid, it is not uniform even after divide by sum
# 2022-0326
# 
def strategy_hyperplane_uniform(N, D):
    y = []
    for i in range(N):
        y.append([])
        rem = 1
        for d in range(D-1):
            y[i].append(rem*(1-(1-numpy.random.rand())**(1/(D-d-1))))
            rem -= y[i][d]
        y[i].append(rem)
    y = torch.tensor(y)
    return y

# strategy_hyperplane_gaussian: all axes are created equal
# y = x**2 = numpy.random.randn(N, D)**2
# all dimensions are equal, but its distribution is not uniform, the pole regions get too much density, i.e. volume shrink too much
# in expectation, generating local_area_gain copies can make surfacial density back to constant (that of x's)
# although density backs to uniform, it didn't yet back to desired constant
# fortunately, since population before-and-after "x-to-y mapping" + "local density correction" is proportional to the global area
# canceling the surface area change can bring population back to N in expectation
# each raw sample at y generate "local_area_gain / global_area_gain" samples in expectation
# in real implementations, local_area_gain is typically small, especially all symmetric orthotants of x are considered
# this results global density correction is prior to local density correction, or at least simutaneously, to improve numerical characteristics
# 2022-0330
# 
def strategy_hyperplane_gaussian(N, D):
    y = numpy.random.randn(N, D)**2
    y = y/y.sum(1, keepdims=True)
    for n in range(N):
        for d in range(D):
            if numpy.isnan(y[n, d]):
                y[n, d] = 1/D
    sphere_area = D * (math.tau/2)**(D/2) / math.gamma(1+D/2)
    square_area = D * math.sqrt(D) / math.gamma(1+D)
    GLOBAL_AREA_GAIN = square_area / sphere_area
    
    local_area_gain = math.sqrt(D) / 2 * numpy.sqrt(y.prod(1, keepdims=False))
    y_out = []
    while 2*len(y_out) < N:
        copies = numpy.random.poisson(local_area_gain / GLOBAL_AREA_GAIN)
        # copies upperbound grows asymptotically 1/sqrt(2) * (tau/e)^(D/2), about 1.52^D
        for i in range(N):
            y_out += [y[i, :]] * copies[i]
    
    M = len(y_out)
    if M > N:
        y_out = y_out[0:N]
    if M < N:
        y_out = y_out + y_out[0:N-M]
    y_out = torch.tensor(y_out)
    return y_out

# Both conditional-distribution and gaussian uses non-linear processes.
# As dimension goes large, numerical characteristic will be ill-conditioned.
# It would be favorable if process uses linear or polynomial processes that native to linear algebra.
# And the factor "1/D!" in the volume of orthodox hyperpyramid (x+y+z<=1, 0<=x,y,z<=1) seems to imply some symmetry.
#
# While there do have some symmetry, the repeated units are not orthodox hyperpyramid.
# It is ordered hyperpyramid (e.g. in 3D space, 0<=x[D-1]<=x[D-2]<=...<=x[0]<=1) to be the repeated units.
# It is stem from a simple fact: EVERY D! points in [0, 1]^D share ONE correspondent point in the ordered hyperpyramid.
# The points in ordered hyperpyramid can be transformed to orthodox hyperpyramid by linear shearing transformation,
# namely, in practice, ascending order is preferred, 0<=x[0]<=x[1]<=...<=x[D-1]<=1, perform x[d+1] -= x[d] for 1<=d<=D-1.
# 
# The linear method exploits the nauture insights of the symmetry lies deeply in permutation.
# By using only linear transformation and operations native to linear spaces,
# this method can keep generating numerical well-performing samples as dimensions rising further.
# 2022-04-29
# 
def strategy_hyperplane_linear(N, D):
    Y = torch.rand([N, D-1])
    Y = Y.sort(dim=1, descending=False, stable=True)[0]
    Y = torch.cat([Y, torch.ones([N, 1])], axis=1)
    for d in range(D-1, 0, -1):
        Y[:, d] -= Y[:, d-1]
    return Y

def strategy_hyperplane(N, D):
    return strategy_hyperplane_linear(N, D)

def generate_3D_demo():
    D = 3
    N = 10000
    y = strategy_hyperplane(N, D)

    fig = matplotlib.pyplot.figure(0, [16, 6])

    ax = fig.add_axes([0.05, 0.1, 0.4, 0.8], projection="3d")
    ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], s=1)
    ax.view_init(elev=25, azim=30)

    M = 100
    y_val = torch.arange(1+M)/M
    y_den = torch.zeros(1+M, D)

    for d in range(D):
        for i in range(N):
            y_den[round(float(y[i, d] / (1/M))), d] += (1/N)/(1/M)
        y_den[0, d] *= 2
        y_den[-1, d] *= 2
    
    stat = fig.add_axes([0.55, 0.1, 0.4, 0.8])
    stat.plot(y_val, y_den)
    stat.grid()

    fig.show()

def BR_stat():
    D = 3
    N = 2

    # U_a = torch.tensor([[0, -2.0, -0.667], [2, 0, -2], [0.65, 2, 0]])
    # U_b = torch.tensor([[0, -0.8, -5.6], [1.654, 0, -0.375], [5.6, 0.8, 0]])
    U_a = torch.tensor([[0, -5.4/7, 6.6/7], [1.2, 0, -0.6], [-6.6, 5.4/7, 0]])
    U_b = torch.tensor([[0, 1.97, -0.01], [-1.2, 0, 2.6], [0.01, -1.97, 0]])
    print("U_a: ", U_a)
    print("U_b: ", U_b)

    PI_a = strategy_hyperplane(N, D)
    PI_a[0, :] = torch.tensor([0.495, 0.005, 0.500])
    PI_a[1, :] = torch.tensor([0.500, 0.005, 0.495])
    I_b = torch.zeros(N, dtype=int)
    PAYOFF_b = torch.zeros(N)
    PAYOFF_a = torch.zeros(N)
    for i_a in range(N):
        i_b_BR = 0
        u_b_BR = -torch.inf
        for i_b in range(D):
            pi_b = torch.zeros([D, 1])
            pi_b[i_b] = 1
            u_b = PI_a[i_a:i_a+1, :] @ U_b @ pi_b
            if u_b_BR < u_b:
                i_b_BR = i_b
                u_b_BR = u_b
            pi_b_BR = torch.zeros([D, 1])
            pi_b_BR[i_b_BR] = 1
        u_a = PI_a[i_a:i_a+1, :] @ U_a @ pi_b_BR
        I_b[i_a] = i_b_BR
        PAYOFF_b[i_a] = u_b_BR
        PAYOFF_a[i_a] = u_a
    print(PAYOFF_a)
    print(PAYOFF_b)
    fig = matplotlib.pyplot.figure(10, [8, 6])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
    ax.scatter3D(PI_a[:, 0], PI_a[:, 1], I_b+1, s=1, c=I_b)
    ax.set_title("Zeroth Card: what will B do according pi_a")
    ax.set_xlabel("Player A throw 1")
    ax.set_ylabel("Player A throw 2")
    ax.set_zlabel("Player B SHOULD do: magenta: 1, green: 2, yellow: 3")
    matplotlib.pyplot.show()

    fig = matplotlib.pyplot.figure(11, [8, 6])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
    ax.scatter3D(PI_a[:, 0], PI_a[:, 1], PAYOFF_b, s=1, c=I_b)
    ax.set_title("Zeroth Card: payoff of B, is max, (therefore continuous)")
    ax.set_xlabel("Player A throw 1")
    ax.set_ylabel("Player A throw 2")
    ax.set_zlabel("HOW MUCH Player B get")
    matplotlib.pyplot.show()
    
    fig = matplotlib.pyplot.figure(12, [8, 6])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
    ax.scatter3D(PI_a[:, 0], PI_a[:, 1], PAYOFF_a, s=1, c=I_b)
    ax.set_title("Zeroth Card: payoff of A, (UNFORTUNATELY, NOT continuous)")
    ax.set_xlabel("Player A throw 1")
    ax.set_ylabel("Player A throw 2")
    ax.set_zlabel("HOW MUCH Player A get")
    matplotlib.pyplot.show()
    return

def BR_demo():
    D = 3
    N = 10000

    U_a = torch.tensor([[0, -5.4/7, 6.6/7], [1.2, 0, -0.6], [-6.6, 5.4/7, 0]])
    U_b = torch.tensor([[0, 1.97, -0.01], [-1.2, 0, 2.6], [0.01, -1.97, 0]])
    print("U_a, U_b:")
    print(U_a)
    print(U_b)

    y_a = strategy_hyperplane(N, D)
    y_b = strategy_hyperplane(N, D)

    # player A find BR:
    i_a_BR = 0
    pi_a_BR = torch.tensor([1, 0, 0])
    u_a_BR = -torch.inf

    y_a[6500, 0] = 0.5
    y_a[6500, 1] = 0.0
    y_a[6500, 2] = 0.5
    for i_a in range(N):
        PRINT = False if i_a % 500 else True
        if PRINT == True:
            print("#---=---=---=---" * 4)
        
        pi_a = y_a[i_a, :]
        STOP = True if i_a == 6500 else False
            
        # player B find BR:
        i_b_BR = 0
        pi_b_BR = torch.tensor([0, 0, 0]).T
        u_b_BR = -torch.inf
        for i_b in range(D):
            pi_b = torch.tensor([0, 0, 0]).T
            pi_b[i_b] = 1
            u_b = pi_a.T @ U_b @ pi_b
            if u_b_BR <= u_b:
                i_b_BR = i_b
                pi_b_BR = pi_b
                u_b_BR = u_b
        # player B find BR.
        u_a = pi_a.T@U_a@pi_b_BR
        if STOP:
            print("STOP:")
            for _ in range(30):
                inp = input(">>> ")
                if len(inp):
                    try:
                        print(eval(inp))
                    except:
                        print("error (skipped)!")
                else:
                    break
        
        if u_a_BR <= u_a:
            i_a_BR = i_a
            pi_a_BR = pi_a
            u_a_BR = u_a
        
        if PRINT:
            print("i_a:", i_a)
            print("pi_a:", pi_a_BR)
            print("pi_b: ", pi_b_BR)
            print("u_b:", u_b_BR)
            print("u_a: ", u_a_BR)

    print("")
    print("Conclusion:")
    print("pi_a: ", pi_a_BR)
    print("u_a: ", u_a_BR)
    # player A find BR.


# 1. idea behind Best-Response:
# Let A choose strategy sigma_a,
# which throw probability pi_a at every (but actually only one in this example) information sets.
# B then find sigma_b = BR(sigma_a)
# then A find if "sigma_a = BR(BR(sigma_a))" has a solution,
# it is done by sigma_a[k+1] = BR(BR(sigma_a[k])), and so forth
# UNFORTUNATELY, BR didn't work
# 
# 2. why Best-Response FAILS to converge:
# when sigma_a near some special points, on which u_b(pure_sigma_b[i]) == u_b(pure_sigma_b[j])
# then sigma_b = BR(sigma_a), it *JUMPS* from an extreme (pole of convex, or, pure strategy) to another one.
# sigma_b jumps back and forth between pure_sigma_b[i] and pure_sigma_b[j],
# and this GAP is not infinitesimal, but FINITE.
# this won't affect much for u_b, since u_b = max(u_b(pure_sigma_b[i]), u_b(pure_sigma_b[j]), u_b(pure_sigma_b[k]), ...)
# HOWEVER, this *JUMP* does affects u_a by a huge amount, since no such equations for u_a,
# (well, unless in some zero-sum settings, but it seems didn't helps much either...)
# anyway, while u_b(sigma_a) is continuous, u_a(BR(sigma_a)) typically always JUMPS,
# which is DISASTEROUS for convergence and stability.
# 
# 3. so, possible solutions:
#   1. to make iteration continuous, *JUMP* can be broken into consecutive segments,
#      which is small_step_BR and to be check whether it will works;
#   2. scale step length by how much its will benefits the player, i.e. policy gradients, regret-based iterators, etc.;
#   3. keep in mind that replacing assigning operation with accumulation operation. (state variables don't change abruptly)
# 
# 2022-04-18
# 

def small_step_BR():
    D = 3
    U_a = torch.tensor([[0, -5.4/7, 6.6/7], [1.2, 0, -0.6], [-6.6, 5.4/7, 0]])
    U_b = torch.tensor([[0, 1.97, -0.01], [-1.2, 0, 2.6], [0.01, -1.97, 0]])
    print("U_a:")
    print(U_a)
    print("U_b:")
    print(U_b)

    pi_a = strategy_hyperplane(1, D).T
    pi_b = strategy_hyperplane(1, D).T
    PI_a = []
    PI_b = []
    u_a_rec = []
    u_b_rec = []
    pi_a_next = pi_a.clone().detach()
    pi_b_next = pi_b.clone().detach()
    for e in range(1000):
        PRINT = True if e%100==0 else False
        if PRINT:
            print("e: ", e)
        
        PI_a.append(pi_a.tolist())
        PI_b.append(pi_b.tolist())
        if PRINT:
            print("pi_a: ", pi_a)
            print("pi_b: ", pi_b)

        u_a = pi_a.T @ U_a @ pi_b
        u_b = pi_a.T @ U_b @ pi_b
        u_a_rec.append(float(u_a))
        u_b_rec.append(float(u_b))
        if PRINT:
            print("u_a: ", u_a)
            print("u_b: ", u_b)
        
        speed_a = 0.05
        u_a_pole = []
        for i_a in range(D):
            u_a_pole.append(U_a[i_a:i_a+1, :] @ pi_b)
        u_a_pole = torch.tensor(u_a_pole)
        i_a_BR = torch.argmax(u_a_pole)
        u_a_BR = u_a_pole[i_a_BR]
        pi_a_BR = torch.nn.functional.one_hot(i_a_BR, D).unsqueeze(1)
        diff_a = pi_a_BR - pi_a
        pi_a_next += diff_a * (u_a_BR - u_a) * speed_a
        
        speed_b = 0.08
        u_b_pole = []
        for i_b in range(D):
            u_b_pole.append(pi_a.T @ U_b[:, i_b:i_b+1])
        u_b_pole = torch.tensor(u_b_pole)
        i_b_BR = torch.argmax(u_b_pole)
        u_b_BR = u_b_pole[i_b_BR]
        pi_b_BR = torch.nn.functional.one_hot(i_b_BR, D).unsqueeze(1)
        diff_b =pi_b_BR - pi_b
        pi_b_next += diff_b * (u_b_BR - u_b) * speed_b

        pi_a = pi_a_next
        pi_b = pi_b_next
    
    print("Conclusion:")
    print("pi_a: ", pi_a)
    print("pi_b: ", pi_b)

    fig = matplotlib.pyplot.figure("fig", [16, 6], 96)

    PI_a = torch.tensor(PI_a).swapaxes(0, 2)[0, :, :].tolist()
    PI_b = torch.tensor(PI_b).swapaxes(0, 2)[0, :, :].tolist()
    ax = fig.add_axes([0.05, 0.1, 0.4, 0.8], projection="3d")
    ax.plot3D(PI_a[0], PI_a[1], PI_a[2], lw=1, c='r')
    ax.plot3D(PI_b[0], PI_b[1], PI_b[2], lw=1, c='b')
    ax.scatter3D(PI_a[0][-1], PI_a[1][-1], PI_a[2][-1], s=20, c='r')
    ax.scatter3D(PI_b[0][-1], PI_b[1][-1], PI_b[2][-1], s=20, c='b')
    y = strategy_hyperplane(1000, D)
    ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], s=1, c='g')

    trend = fig.add_axes([0.55, 0.1, 0.4, 0.8])
    trend.plot(range(1000), u_a_rec, 'r')
    trend.plot(range(1000), u_b_rec, 'b')
    trend.grid()
    fig.show()
    
# small_step_BR -- conclusion and limitions:
# OK, way better.
# however, the locus features saw-tooth pattern.
# the mechanism is because the driving force is pointing at the pole.
# when near the equilibrium, the utility/payoff of the poles are approching.
# the direction of driving force then alters its direction quite frequently.
# 



def CFR_plus():
    M = 3
    N = 2
    # U_a = torch.tensor([[0, -5.4/7, 6.6/7], [1.2, 0, -0.6], [-6.6, 5.4/7, 0]])
    # U_b = torch.tensor([[0, 1.97, -0.01], [-1.2, 0, 2.6], [0.01, -1.97, 0]])
    # U_a = torch.tensor([[0, -2.0, -0.667], [2, 0, -2], [0.65, 2, 0]])
    # U_b = torch.tensor([[0, -0.8, -5.6], [1.654, 0, -0.375], [5.6, 0.8, 0]])
    U_a = torch.tensor([[3.0, 3], [2, 5], [0, 6]])
    U_b = torch.tensor([[3.0, 2], [2, 6], [3, 1]])
    
    print("U_a:")
    print(U_a)
    print("U_b:")
    print(U_b)

    T = 2000
    # pi_a = strategy_hyperplane(1, M).T
    # pi_b = strategy_hyperplane(1, N).T
    # pi_a = torch.tensor([[0.02, 0.28, 0.70]]).T
    # pi_b = torch.tensor([[0.30, 0.70]]).T
    pi_a = torch.tensor([[0.800, 0.199, 0.001]]).T
    pi_b = torch.tensor([[0.669, 0.331]]).T

    PI_a = torch.zeros(M, T)
    PI_b = torch.zeros(N, T)
    u_a_rec = []
    u_b_rec = []
    pi_a_next = pi_a.clone().detach()
    pi_b_next = pi_b.clone().detach()
    R_a = pi_a.clone().detach()*20
    R_b = pi_b.clone().detach()*20
    
    for t in range(T):
        PRINT = True if t%10==0 else False
        if PRINT:
            print("t: ", t)
        
        # PI_a.append(pi_a.tolist())
        PI_a[:, t] = pi_a[:, 0]
        # PI_b.append(pi_b.tolist())
        PI_b[:, t] = pi_b[:, 0]

        if PRINT:
            print("pi_a: ", pi_a)
            print("pi_b: ", pi_b)

        u_a = pi_a.T @ U_a @ pi_b
        u_b = pi_a.T @ U_b @ pi_b
        u_a_rec.append(float(u_a))
        u_b_rec.append(float(u_b))
        if PRINT:
            print("u_a: ", u_a)
            print("u_b: ", u_b)
        
        r_a_pole = U_a @ pi_b - u_a
        r_a_pole[r_a_pole < 0] = 0
        R_a *= 0.999
        R_a += r_a_pole
        if R_a.sum():
            pi_a_next = R_a/R_a.sum()
        else:
            pi_a_next = torch.ones([M, 1])/M
        
        r_b_pole = (pi_a.T @ U_b).T - u_b
        r_b_pole[r_b_pole < 0] = 0
        R_b *= 0.999
        R_b += r_b_pole
        if R_b.sum():
            pi_b_next = R_b/R_b.sum()
        else:
            pi_b_next = torch.ones([N, 1])/N
        
        pi_a = pi_a_next
        pi_b = pi_b_next
    
    print("Conclusion:")
    print("pi_a: ", pi_a)
    print("pi_b: ", pi_b)
    
    fig = matplotlib.pyplot.figure("fig", [20, 6], 96)
    
    ax = fig.add_axes([0.025, 0.1, 0.30, 0.8])
    ax.fill_between(range(T), PI_a[0, :] * 0, PI_a[0, :])
    ax.fill_between(range(T), PI_a[0, :], PI_a[0, :] + PI_a[1, :])
    ax.fill_between(range(T), PI_a[0, :] + PI_a[1, :], PI_a[0, :] + PI_a[1, :] + PI_a[2, :])

    bx = fig.add_axes([0.350, 0.1, 0.30, 0.8])
    bx.fill_between(range(T), PI_b[0, :] * 0, PI_b[0, :])
    bx.fill_between(range(T), PI_b[0, :], PI_b[0, :] + PI_b[1, :])

    trend = fig.add_axes([0.675, 0.1, 0.30, 0.8])
    trend.plot(range(T), u_a_rec, 'r')
    trend.plot(range(T), u_b_rec, 'b')
    trend.grid()
    matplotlib.pyplot.show()
    
# CFR is more mild when e gets large, and discontinuous vanishes
# since CFR has R as integral/summation parts, it can stabilize the dynamics while iterating difference equations
#
if __name__ == "__main__":
    # numpy.random.seed(133484)
    torch.manual_seed(133484)
    CFR_plus()
    # BR_stat()
