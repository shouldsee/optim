from pymisca.proba import *
from pymisca.util import *

def l1_normF(f):
    g= lambda *args:np.sum(np.abs(f(*args)),axis=-1)
    return g
def l2_normF(f):
    g= lambda *args:np.sum(np.square(f(*args)),axis=-1)**0.5
    return g
def negF(f):
    g = lambda *args:-f(*args)
    return g
def addF(f,g):
    def h(*args):
        return f(*args) + g(*args)
    return h
def make_gradF(f,eps=1E-4):
    def gradF(*x):
        f0 = f(*x)
        grad = [0]*len(x)
        for i,xi in enumerate(x):
            xcurr = list(x)[:]
            xcurr[i]= xi +eps
            df = f(*xcurr) - f0
            grad[i]= df/eps
        return grad
    return gradF

def forward(x,T,adv=None,D=None):
    if D is None:
        D = len(x)//2
    X = [x]
#     f0 = h(*x[:D]) + 0.000

    for i in range(T):
        x = adv(x,i)
        X.append(x[:])
    X = np.array(X)
    return X

def make_adv_descent(h,D=None,lr=0.1,gradF = None):
    '''
    Standard gradient descent
    '''
    if D is None:
        D = h.func_code.co_argcount
    if gradF is None:
        gradF = make_gradF(h)
    def adv_descent(IN,i):
        x = IN[:D]
        v = IN[D:]
#         lr = 1.
    #     d = 0.1
    #     dx = np.ones(np.shape(x))*0.1
        dx = gradF(*x)
        x = np.add(x,np.multiply(-1*lr,dx))
        
        return np.hstack([x,v])
    return adv_descent
def make_adv_unif(h,alpha=None,eta=0.0,
                  dt=0.1,D=None,gradF=None):
    '''
    Prepare a descent functional from an objective function
    Samples random gradients from independent uniform distributions, assume h(X)=h(x1,x2,...,xd),
    grad(h) = \nabla_X {h} is the gradient
    then the stochastic gradient G is constructed so that E(G)=grad(h) and Var(G_i) = 1/3*(grad(h)_i - \alpha)^2,
    where \alpha is the prior belief and can be set to E_i(|grad(h)_i|) (l1 norm of the gradient).
    
    '''
    if D is None:
        D = h.func_code.co_argcount
    if gradF is None:
        gradF = make_gradF(h)
    def adv_descent(IN,i):
        x = IN[:D]
        v = IN[D:]
        v = np.array(v)
        if 1:
#         if i%2:            
            gd = gradF(*x)
#            vct = -gamma_vector(MEAN=gd,disper=disper,size=1,rv=rv).ravel()
            gd = gd/l2_norm(gd)
            vct =  -unif_vector(MEAN=gd,
                                a = alpha,size=1).ravel()
            v = (eta * v + vct)/(1.+eta)
#         else:
            x = np.add(x,np.multiply(dt,v))    
        return np.hstack([x,v])
    return adv_descent
def make_surfer(h,x=None,D = None,lr= 1.,gradF=None,f0=None,
                fix = 1,
                vt = 0.05, vn = 0.01,mask = None):
    '''
    A not so successful attempt to identify homoclinic orbits of a function ("h")
    f0: function value over the orbit
    fix: toggles between fixed orbit (fix=1) and descending orbit (fix=0)
    vt: tangential speed
    vn: orthognal speed
    '''
    if f0 is None:
        f0 = h(*x[:D])
    if gradF is None:
        gradF = make_gradF(h)
    def adv(IN,i):
        IN = np.array(IN)
        x0 = IN[:D]
        v0 = IN[D:]
        gd0 = np.add(gradF(*x0) , gradF(*np.add(x0,v0)))/2.
#         print gd0
        f = h(*x0)
        df = f-f0
        if mask is not None:
            x = x0
            v = v0
            gd = gd0
        else:
            x = x0
            v = v0
            gd= gd0
    #     v  = IN[2:]    
    #     df = lr
        s = i%2
        if s==0:       
            dx = np.multiply(lr,v)
            x = np.add(x,dx).tolist()
        elif s==1:
            v = np.array(v)
#             f = h(*np.add(x,0))
            df = f-f0
    #         dv = np.multiply(gd,-df)
            gdgd = np.sqrt(np.dot(gd,gd))
            gdn = gd / gdgd
            
            #### Find orthogonal vector
            dot = np.dot(v,gd) / gdgd**2 *gdn
            vo = v - dot  
            lvo = np.sqrt(np.dot(vo,vo))
    
            vo = vo/lvo*vt #### Constant orthogonal/tangential speed
            if fix:
                dot = gdn * -cmp(df,0)*vn  ##### constant normal speed
            else:
                dot = gdn *-vn  ##### constant normal speed

            v = dot + vo
            v = v.tolist()
            pass
        if mask is not None:
            np.putmask(x0,mask,x)
            np.putmask(v0,mask,v)
#             v0[mask]= v
        else:
            x0=x
            v0=v
        return np.hstack([x0,v0])
#         lst = [x0,v0]
#         return sum(lst,[])
    return adv    

def make_adv_grandwalk(f,gradF=None,D = None, dt = 0.1, momentum = 0.5):
    '''
    A very bad example of random vector descend
    '''
    if D is None:
        D = f.func_code.co_argcount//2
    if gradF is None:
        gradF = make_gradF(f)
    fbeta = make_fbeta(D=min(max(D,3),6))
    def adv(IN,i):
        x = IN[:D]
        v0 = IN[D:]
        gd = gradF(*x)
        gdl= l2_norm(gd)
#         gdl = np.
        if gdl:
            gdn = np.divide(gd,gdl)     
        else:
            gdn = gd
#         v = radial_randvector(fbeta,D=D,n=1,v0=gdn,E = 1.)
#         v = radial_randvector(fbeta,D=D,n=1,v0=-gdn,E = 1./gdl)
#         v = radial_randvector(fbeta,D=D,n=1,v0=-gdn,E = 1./gdl**2)
        v = radial_randvector(fbeta,D=D,n=1,v0=-gdn,E = np.clip(1./gdl,0.05,10))
#         v = radial_randvector(fbeta,D=D,n=1,v0=-gdn,E = np.clip(1./(gdl/len(gd)),0.1,5))
#         v = np.isnan
        v = (momentum * v0 + v)/(1+momentum)
        x =  np.add(x,np.multiply(v,dt)).ravel()
        OUT = np.hstack([x,np.ravel(v)])
        return OUT 
    return adv



from pymisca.vis_util import *

def add_point(pt):
    plt.plot(pt[0],pt[1],'o',markersize=10)
def traj_3D(X,ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

#     plt.sca(ax)
    ax.plot(X[:,0],X[:,1],X[:,2],'r-')
    pt = X[0]
    ax.scatter3D(pt[0],pt[1],pt[2],marker='o',s=100,c='b')
    pt = X[-1]
    ax.scatter3D(pt[0],pt[1],pt[2],marker='o',s=100,c='orange')
    # plt.plot(X[-1,0],X[-1,1],'ob',markersize=10)
    return ax    
def traj_2D(X,ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,)

#     plt.sca(ax)
    ax.plot(X[:,0],X[:,1],'r-')
    pt = X[0]
    ax.scatter(pt[0],pt[1],marker='o',s=100,c='b')
    pt = X[-1]
    ax.scatter(pt[0],pt[1],marker='o',s=100,c='orange')
    # plt.plot(X[-1,0],X[-1,1],'ob',markersize=10)
    return ax





def make_surfer(h,x=None,D = None,lr= 1.,gradF=None,f0=None,
                fix = 1,
                vt = 0.05, vn = 0.01,mask = None):
    if f0 is None:
        f0 = h(*x[:D])
    if gradF is None:
        gradF = make_gradF(h)
    def adv(IN,i):
#         if mask is not None:
#             IN = [x if ma else 0. for x,ma in zip(IN,mask) ]
#             IN[mask] = 0
    #     d = 0.1
    #     dx = np.ones(np.shape(x))*0.1
        IN = np.array(IN)
        x0 = IN[:D]
        v0 = IN[D:]
        gd0 = np.add(gradF(*x0) , gradF(*np.add(x0,v0)))/2
#         print gd0
        f = h(*x0)
        df = f-f0
        if mask is not None:
            x = np.take(x0,mask)
            v = np.take(v0,mask)
            gd= np.take(gd0,mask)
        else:
            x = x0
            v = v0
            gd= gd0
    #     v  = IN[2:]    
    #     df = lr
        s = i%2
        if s==0:       
            dx = np.multiply(lr,v)
            x = np.add(x,dx).tolist()
        elif s==1:
            v = np.array(v)
#             f = h(*np.add(x,0))
            df = f-f0
    #         dv = np.multiply(gd,-df)
            gdgd = np.sqrt(np.dot(gd,gd))
            gdn = gd / gdgd
            
            #### Find orthogonal vector
            dot = np.dot(v,gd) / gdgd**2 *gdn
            vo = v - dot  
            lvo = np.sqrt(np.dot(vo,vo))
    
            vo = vo/lvo*vt #### Constant orthogonal/tangential speed
            if fix:
                dot = gdn * -cmp(df,0)*vn  ##### constant normal speed
            else:
                dot = gdn *-vn  ##### constant normal speed

            v = dot + vo
            v = v.tolist()
            pass
        if mask is not None:
            np.put(x0,mask,x)
            np.put(v0,mask,v)
#             v0[mask]= v
        else:
            x0=x
            v0=v
        return np.hstack([x0,v0])
#         lst = [x0,v0]
#         return sum(lst,[])
    return adv    
