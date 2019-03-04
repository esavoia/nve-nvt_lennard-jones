from NVE_lj import VVM_mdlj
# default values: Bz=0,q=0
# VVM_mdlj(mode=0,nstep=1000,freq=10,kt=1)
# scalar charge
VVM_mdlj(mode=0,nstep=1000,freq=10,kt=1,Bz=1,q=1)
# fcc charges
# VVM_mdlj(mode=0,nstep=1000,freq=10,kt=1,Bz=1,q=[1,-1,0,0])
