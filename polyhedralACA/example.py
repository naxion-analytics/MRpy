
# INITIAL PARAMETERS (test example)
######################################
pref = np.array( [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
rows =    np.array( [0,0,0,0,0,0,1,1,1, 1 ,1 ,1 ,2,2, 2, 2, 2, 2,3,3, 3, 3, 3, 3,4,4, 4, 4, 4, 4] )
columns = np.array( [2,3,4,5,6,7,4,5,10,11,12,13,0,1,12,13,18,19,8,9,10,11,16,17,0,1,12,13,14,15] )
data =    np.array( [1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1] )

X_orig = sparse.coo_matrix( (data,(rows,columns)))
a_orig = np.array( [-68.6799, -9.4126, -57.002, 64.9769, -79.9109])

UTILITIES_UB = 100.
#######################################

#### putting them together into augmented form
X_orig = sparse.coo_matrix( isFullRank(X_orig.toarray()) )

q,p = X_orig.shape
bigX_orig = sparse.bmat([[X_orig,None],[sparse.eye(p),sparse.eye(p)]]).tocsr()
bigA = np.array( [-68.6799, -9.4126, -57.002, 64.9769, -79.9109]+ [UTILITIES_UB]*p ) ### utilities at most 100.
#################################################################

n_levels = len(pref)
bigX = Crop(bigX_orig, cols=pref); X = Crop(X_orig, cols=pref)
qr,pr = bigX.shape

constrained_opt = polyhedra(bigX, bigA)  ### find an initial point in the polyhedron

params = constrained_opt.x
params[0:pr] = np.array([u/params[-1] if params[pr+i]>0 else 0. for i,u in enumerate(params[0:pr])])
X, bigX, pref = RemoveJs( X_orig, bigX_orig, params, pref) ### remove j's corresponding to ujs = 0

center = FindCenter( bigX, bigA, params) ### step closer to center

long_axis = FindLongAxis(X, center) ### create ellipse with center, find its major axis

longest = np.zeros( n_levels )
analytic_center = np.zeros( n_levels )

longest[np.where(pref != 0)] = long_axis
analytic_center[np.where(pref !=0)] = center[:n_levels]

# <codecell>

longest, analytic_center



# INITIAL PARAMETERS (test example)
# empty polyhedron example
######################################
pref = np.array( [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
data = np.array( [[0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0 ,1,-1,-1,1],
                  [0,0,1,-1 ,-1,1,0,0,0,0,0,0,0,0,1,-1,0,0,0,0],
                  [0,0,0,0,0,0,-1,1,0,0,1,-1,0,0,0,0,-1,1,0,0],
                  [0,0,0,0,1,-1,0,0,1,-1,0,0,-1,1,0,0,0,0,0,0],
                  [0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,1,-1,1,-1],
                  [1,-1,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,0,0,0,0],
                  [0,0,-1,1,-1,1,0,0,1,-1,0,0,0,0,0,0,0,0,0,0],
                  [0,0,-1,1,-1,1,0,0,0,0,0,0,1,-1,0,0,0,0,0,0]])
#######################################

UTILITIES_UB = 100. 
X_orig = sparse.coo_matrix( data)
a_orig = np.array( [43.8819, -2.6829, 20.7079, 59.4305, -57.5984, -17.7432, 88.7329, 25.0771])

q,p = X_orig.shape
delta_X = sparse.bmat([[X_orig,np.ones((q,1))],[-X_orig,np.ones((q,1))],[-1*sparse.eye(p),None]])

X = Crop(delta_X,cols=pref)
a = np.concatenate( (a_orig,-a_orig, [-UTILITIES_UB]*p ))
q,p = X.shape

# <codecell>

#### if empty polyhedron
f = lambda x: abs(x[-1])
cons = ({ 'type': 'ineq',
          'fun': lambda x: X.dot(x) - a},
        {'type': 'ineq',
         'fun': lambda x: x[:-1]}
         )
opt = optimize.minimize(f,
                        x0=np.random.random(p),
                        method='SLSQP',
                        constraints=cons,
                        options={'disp':True})

u0, delta = opt.x[:-1], opt.x[-1]

A = sparse.bmat([[X_orig, sparse.eye(8),np.zeros((8,8))],[X_orig,np.zeros((8,8)),-sparse.eye(8)]])
B = sparse.hstack([sparse.eye(20),np.zeros((20,16))])
bigX_orig = sparse.bmat( [[A,None],[B, sparse.eye(20)]])
bigA = np.concatenate( (a_orig+delta+.5, a_orig-delta-.5, [UTILITIES_UB]*20))

# <codecell>

bigX = Crop(bigX_orig, cols=pref)
qr,pr = bigX.shape
constrained_opt = polyhedra(bigX, bigA)

params = constrained_opt.x
params[0:pr] = np.array([u/params[-1] if params[pr+i]>0 else 0. for i,u in enumerate(params[0:pr])])
X, bigX, pref = RemoveJs( X_orig, bigX_orig, params, pref) ### remove j's corresponding to ujs = 0

center = FindCenter( bigX, bigA, params) ### step closer to center

# <codecell>

center

# <codecell>

analytic_center = np.zeros( n_levels )
analytic_center[np.where(pref !=0)] = center[:n_levels]

# <codecell>