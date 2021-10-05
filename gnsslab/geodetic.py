import numpy as np
from numba import jit

ellipsoids_names = np.array([
    'CLK66','GRS67','GRS80','WGS72','WGS84','ATS77','NAD27','NAD83','INTER','KRASS','CGCS2000',
])

# 0-3: a, 1/f, b, e
ellipsoids = np.array([
    [6378206.4, 294.9786982,0,0],   # Clarke 1866
    [6378160.0, 298.247167427,0,0], # GRS 1967
    [6378137.0, 298.257222101,0,0], # GRS 1980
    [6378135.0, 298.26,0,0],        # WGS 1972
    [6378137.0, 298.257223563,0,0], # WGS 1984
    [6378135.0, 298.257,0,0],       # ATS77
    [6378206.4, 294.9786982,0,0],   # North American Datum 1927
    [6378137.0, 298.257222101,0,0], # North American Datum 1983
    [6378388.0, 297.0,0,0],         # International
    [6378245.0, 298.3,0,0],         # Krassovsky (USSR)
    [6378137.0, 298.257223563,0,0], # China CGCS 2000
])

# # product of the Earth's mass and the Gravitational Constant(WGS-84)
# mu = 3.986004418e14 # m3/s
# # Earth's angular velocity (WGS-84)
# omega = 7292115e-11 # radians/sec

# calculate other parameters
ellipsoids[:,1] = 1/ellipsoids[:,1]
ellipsoids[:,2] = ellipsoids[:,0] * (1-ellipsoids[:,1])
ellipsoids[:,3] = np.sqrt(1 - (1-ellipsoids[:,1])**2)

# ellipsoids = np.array([
#     ['CLK66',    6378206.4, 294.9786982],   # Clarke 1866
#     ['GRS67',    6378160.0, 298.247167427], # GRS 1967
#     ['GRS80',    6378137.0, 298.257222101], # GRS 1980
#     ['WGS72',    6378135.0, 298.26],        # WGS 1972
#     ['WGS84',    6378137.0, 298.257223563], # WGS 1984
#     ['ATS77',    6378135.0, 298.257],       # ATS77
#     ['NAD27',    6378206.4, 294.9786982],   # North American Datum 1927
#     ['NAD83',    6378137.0, 298.257222101], # North American Datum 1983
#     ['INTER',    6378388.0, 297.0],         # International
#     ['KRASS',    6378245.0, 298.3],         # Krassovsky (USSR)
#     ['CGCS2000', 6378137.0, 298.257223563], # China CGCS 2000
# ])

# class GeoEllipsoid():
#     def __init__(self, name):
#         idx = list(ellipsoids[:,0]).index(name)
#         self.a, self.f = float(ellipsoids[idx,1]), 1/float(ellipsoids[idx,2])
#         # product of the Earth's mass and the Gravitational Constant(WGS-84)
#         self.mu = 3.986004418e14 # m3/s
#         # Earth's angular velocity (WGS-84)
#         self.omega = 7292115e-11 # radians/sec
#         # calculate other parameters
#         self.b = self.a * (1 - self.f) # semi-minor axis
#         self.e  = np.sqrt(1 - (1 - self.f)**2) # first numerical eccentricity

@jit(nopython=True)
def xyz2llr(xyz,deg=False):
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    r = np.sqrt(x*x + y*y + z*z)
    lat = np.arcsin(z/r)
    lon = np.arctan2(y, x)
    lon = np.mod(lon, 2*np.pi)
    if deg: # rad2deg
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
    llr = np.concatenate((lat.reshape(-1,1),lon.reshape(-1,1),r.reshape(-1,1)),axis=1)
    return llr

@jit(nopython=True)
def llh2xyz(llh,ell=None,deg=False):
    # ell = GeoEllipsoid("WGS84") if ell is None else ell
    ell = ellipsoids[4] if ell is None else ell # default WGS84, jit mode
    lat, lon, h = llh[:,0],llh[:,1],llh[:,2]
    
    if deg: # deg2rad
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    N = ell[0] / np.sqrt(1.0 - ell[3] * ell[3] * np.sin(lat) * np.sin(lat))
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1-ell[3]*ell[3]) + h) * np.sin(lat)
    xyz = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)),axis=1)
    return xyz

@jit(nopython=True)
def xyz2llh(xyz,ell=None,deg=False,retR=False):
    # retR: return llhr (r is radius)
    # ell = GeoEllipsoid("WGS84") if ell is None else ell
    ell = ellipsoids[4] if ell is None else ell # default WGS84, jit mode
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    if retR:
        radius = np.sqrt(x*x + y*y + z*z)
    r = np.sqrt(x*x + y*y) # r is distance from spin axis
    lat0 = np.arctan2(z, r) # first guess of lat
    # default tolerance for |lat-lat0|
    eps = np.ones(len(lat0)) * 1e-11  # 1.e-11*6378137(m)=0.06378137 (mm)
    # maximun iteration times
    max_iter = 10  # normally 3~4 iterations should be enough
    e2 = ell[3] * ell[3]
    # iterate
    for i in range(max_iter):
        # compute radius of curvature in prime vertical direction
        N = ell[0] / np.sqrt(1.0 - e2 * np.sin(lat0) * np.sin(lat0))
        lat = np.arctan2(z + N * e2 * np.sin(lat0), r) # compute lat
        if (np.abs(lat-lat0) < eps).all(): # test for convergence
            break        
        lat0 = lat # update lat0

    # if not converged, give an error message
    # pass for jit mode
    # if i == max_iter and np.abs(lat-lat0) >= eps:
        # raise ValueError('XYZ2LLH can not converge after %d iterations.' % i)

    # direct calculation of longitude and ellipsoidal height
    lon = np.arctan2(y, x)
    lon = np.mod(lon, 2*np.pi) # convert lon to (0~2*pi)
    h   = r / np.cos(lat) - N
    
    if deg: # rad2deg
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
    if retR:
        llh = np.concatenate((lat.reshape(-1,1),lon.reshape(-1,1),h.reshape(-1,1),
                                radius.reshape(-1,1)),axis=1)
    else:
        llh = np.concatenate((lat.reshape(-1,1),lon.reshape(-1,1),h.reshape(-1,1)),axis=1)
    return llh

@jit(nopython=True)
def xyz2neu(xyz, org, ell=None, xyzOC=False):
    xyz = xyz.copy().reshape(-1,3)
    # org: logal origin shape (1,3), ECEF
    # xyzOC: xyz is local centered, else is ECEF
    # ell = GeoEllipsoid("WGS84") if ell is None else ell
    ell = ellipsoids[4] if ell is None else ell # default WGS84, jit mode
    # Calc the geodetic lat and lon of the local origin.
    org = org.copy().reshape(1,3)
    llh = xyz2llh(org, ell)
    lat,lon = llh[0][0], llh[0][1]
    # compute the rotation matrix
    R = np.array([
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [-np.sin(lon),             np.cos(lon),         0],
        [np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon),  np.sin(lat)]
    ])
    # compute topocentric coordinates (n, e, u)
    # neu = R.dot((xyz-np.tile(org,(len(xyz),1))).T).T if not xyzOC else R.dot(xyz.T).T
    # neu = R.dot(xyz.T-org.repeat(len(xyz)).reshape(-1,len(xyz))).T if not xyzOC else R.dot(xyz.T).T
    if not xyzOC:
        org = org.repeat(len(xyz)).reshape(-1,len(xyz))
        # neu = R.dot(xyz.T-org).T
        neu = np.dot(R, xyz.T-org).T
    else:
        # neu = R.dot(xyz.T).T
        neu = np.dot(R, xyz.T).T
    return neu

@jit(nopython=True)
def neu2xyz(neu, org, ell=None, xyzOC=False):
    neu = neu.copy().reshape(-1,3)
    # ell = GeoEllipsoid("WGS84") if ell is None else ell
    ell = ellipsoids[4] if ell is None else ell # default WGS84, jit mode
    org = org.copy().reshape(1,3)
    llh = xyz2llh(org, ell)
    lat,lon = llh[0][0], llh[0][1]
    R = np.array([
        [-np.sin(lat)*np.cos(lon), -np.sin(lon), np.cos(lat)*np.cos(lon)],
        [-np.sin(lat)*np.sin(lon),  np.cos(lon), np.cos(lat)*np.sin(lon)],
        [             np.cos(lat),            0,  np.sin(lat)]
    ])
    # xyz = R.dot(neu.T).T
    xyz = np.dot(R, neu.T).T
    # xyz = xyz + np.tile(org, (len(xyz),1)) if not xyzOC else xyz
    # xyz = xyz + org.repeat(len(xyz)).reshape(-1,len(xyz)).T if not xyzOC else xyz
    if not xyzOC:
        org = org.repeat(len(xyz)).reshape(-1,len(xyz))
        xyz = xyz + org
    return xyz

@jit(nopython=True)
def neu2xyz_batch(neu,org,xyzOC=True):
    # assert len(neu)==len(org)
    xyz = np.zeros((len(neu),3))
    for i in range(len(neu)):
        xyz[i] = neu2xyz(neu[i],org[i],xyzOC=xyzOC)
    return xyz

# test
# xyz = np.array([[-2148744.082831,4426641.272647,4044655.926934]])
# llr = xyz2llr(xyz)

# llh = llr
# llh[0,2] = 200

# xyz2 = llh2xyz(llh)

# llh2 = xyz2llh(xyz2)

# xyz = np.array([[-2148744.082831,4426641.272647,4044655.926934],
#                [-2148740.082831,4426640.272647,4044650.926934]
#                ])
# org = np.array([[-2148740.082831,4426640.272647,4044650.926934]])
# xyz2neu(xyz,org)
# xyz2neu(xyz-org,org,xyzOC=True)