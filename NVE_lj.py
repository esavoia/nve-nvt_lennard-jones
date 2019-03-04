# nstep -> number of integration steps
# rho -> density: total mass / simulation volume, simulation volume = Lref^3
#       total mass: particle's mass * number of particles in a cell * number of cells
#                         1         * 4 = (1 + 3) [fcc]             * m^3 (see {MDLJ_lib} )
# m -> side length of the MD-box in units of lattice spacing [total number of cells: m^3 ] ( see {MDLJ_lib:fcc} )
# kt -> Kinetic Temperature
# dt -> timestep in what units?
# freq -> measure frequency, measures are execute every nfreq timesteps
# Bz -> value of External Magnetic Field (in what units?)
# q -> Charges (in what units?), a single value or one for each particle of fcc's unit cell (see {MDLJ_lib:magnetic_vel_verlet})
#      q = scalr | [ q000, qhh0, qh0h, q0hh ]
# mode -> see description
def VVM_mdlj(nstep=1000, rho=0.84, m=6, kt=0.694, dt=0.005, freq=1, Bz=0, q=0, mode=0):
    """ Modena 19/04/2015 - LJMD  3D-version
    Numerical integration of Newton's equations for N interacting LJ-particles
    in a 3 dimensional space using reduced LJ units (mass,sigma,epsilon):
    Dynamics: microcanonical - velocity Verlet algorithm;
    INPUT:
      initial configuration from file 'conf_in' (binary):
      - 1th row:  x_1 y_1 z_1 (initial coordinates of 1st particle)
      -  . . .
      - Nth row   x_N y_N z_N (initial coordinates of Nth particle)
      - N+1th row:  px_1 py_1 pz_1 (initial momenta of 1st particle)
      -  . . .
      - 2*Nth row   px_N py_N pz_N (initial momenta of Nth particle)
      if mode is 0 initial positions are sampled from FCC lattice and previous input/output files are removed
      if mode is not 0 initial positions are read from configuration file
      if mode is even initial velocities are extracted from Maxwell Distribution and previous output files are removed
      if mode is not 0 and odd the run is restarted from previous configurations
        in this case correlation functions and MSD are computed using the whole run
    OUTPUT:
    - enep          total potential energy  E_p
    - vir_p         virial coefficient
    - enek          total kinetic energy E_k
    - enet          total energy Ep+Ek (constant of motion for microcanonical dynamics)
    - vcm(x,y,z)    center of mass momentum (constant of motion)
    PLOT:
    - Energies      total, potential and kinetic energy for the whole run vs. timestep
    - Temperature   Mean temperature computed as a running average vs. timestep
    - RDF           Radial distribution function as a function of interparticle distance
    - VCF           Diagonal and off-diagonal average of velocity correlation tensor
    - VCF Diffusion Diffusion coefficient as an integral of diagonal average of VCF
    - MSD           Mean square displacement
    - MSD Diffusion Diffusion coefficient as a function of the angular coefficient of the MSD
    """
    from numpy import random, sqrt, sum, mean, abs
    from MDLJ_lib import MDLJ_lib

    a=(4/rho)**(1./3.)  # Lattice constant [ particles' mass = 1 ]
    Lref=a*m            # Lattice constant * m = side of the cubic MD-box
    N=4*m**3            # number of particles: see {MDLJ_lib:fcc}
    md=MDLJ_lib(N, Lref)    # see {MDLJ_lib}
    g=3*N-3;            # number of dof
    gdr_out='gdrmd.out'
    # check argument
    if isinstance(q,list):  # q is a list -> must contains 4 elements
        if len(q) != 4:
            print("Invalid Argument: q is not a list of 4 charges (q000, qhh0, qh0h, q0hh)")
            exit(1)
        # q is a valid list, build charges:
        # md.fcc_charges(N, q000, qhh0, qh0h, q0hh)
        q000, qhh0, qh0h, q0hh = q
        charges = md.fcc_charges(N, q000, qhh0, qh0h, q0hh)
    else: # q is a scalar
        charges = q

    print( "# Magnetic Velocity Verlet" )
    print( "# initial (kinetic) temperature kt = %8.4f" % kt )
    print( "# integration time step dt = %8.4f" % dt )
    print( "# number of time steps for this run ns = %d" % nstep )
    print( "# side of the cubic MD-box L = %8.4f " % Lref )
    print( "# number of particles  N = %d" % N )
    print( "# density rho = %8.4f"  % rho )
    print( "# External Magnetic Field = %8.4f"  % Bz )
    
    if mode :
        # mode = 1
        md.read_input(N, conf_in='conf_in.b')
        print("# initial configuration read from config file ")
        # read charges
        charges = read_q(N, conf_in='q.b')
    else :
        # mode = 0 initial positions from FCC lattice and clean_run
        md.clean_run()
        md.fcc(m)
        # initial velocities: realization of random gaussian process with zero mean
        # so that initial momentum is exactly zero at the initial time
        # this results in a stationary center of mass for the N particle system
    if mode%2 == 0:
        # mode is even (also 0) (re)samples initial velocities from gaussian
        md.clean_run(keepinput=True)
        md.t=0
        pstd=sqrt(md.mass*kt) # std dev of p
        md.px = random.normal(0., pstd, N)
        md.py = random.normal(0., pstd, N)
        md.pz = random.normal(0., pstd, N)
        vcmx  = sum(md.px)
        vcmy  = sum(md.py)
        vcmz  = sum(md.pz)
        md.px   -= vcmx/N
        md.py   -= vcmy/N
        md.pz   -= vcmz/N
        # @CAREFUL
        # coordinates are rescaled to be in [-.5,.5]: R :-> r = R/Lref 
        # Rescale coordinates => Rescale momenta: P :-> p = P*md.L  md.L is the simbox size (Lref -> see line:45)
        md.px *= md.L
        md.py *= md.L
        md.pz *= md.L
        print("# initial momenta sampled from maxwellian at temperature %8.4f " % kt)
    # ofstream eout("outmd.txt");

    # Run Integrator
    tt, ekt, ept, vir, ektsq, eptsq, etsq = md.magnetic_vel_verlet(N, nstep, dt, freq,Bz,charges)
    # save charges
    write_q(charges, N)
    # End run

    DEiE = sqrt(etsq/nstep - ((ekt + ept)/nstep)**2)/abs((ekt+ept)/nstep)
    pres = (2*ekt + vir)/(3.*md.L**3)
    print( "# END OF THE RUN: Dumping average quantities")
    print( "# average potential energy  ep   = %10.5g " % (ept/nstep) )
    print( "# average kinetic   energy  ek   = %10.5g " % (ekt/nstep) )
    print( "# average total     energy  et   = %10.5g " % ((ekt+ept)/nstep) )
    print( "# average pressure          pres = %10.5g" %  (pres/nstep) )
    print( "# average temperature       kT   = %10.5g" %(2.*ekt/(nstep*g)) )
    print( "# average total energy relative fluctuations: DE/E = %.4g" % DEiE)
    # gdr final printout - use actual value of density rho
    md.write_gdr( N, tt, rho, gdr_out )
    md.calc_vcf( N, dt, freq )
    md.calc_MSD( N, dt, freq )
#   end of md - visualization:
    from matplotlib.pyplot import plot, show
    from numpy import loadtxt,ones,size
#   Energies vs. timestep
    t,enp,enk,ent = loadtxt('run.out', usecols = (0,1,2,4), unpack=True)
    plot(t,enk,'b-',t,enp,'g-',t,ent/N,'r-')
    show()
#   Mean Temperature vs. timestep
    t,itemp,avgtemp = loadtxt('temperature.out', unpack=True)
    plot(t,itemp,'r-',t,avgtemp,'b.')
    show()
#   Radial distribution function
    r,gdr = loadtxt(gdr_out, unpack=True, skiprows=1 )
    plot(r,gdr,'b.',r,ones(size(r)),'k-')
    show()
#   Velocity correlation functions
    t,vcfxx,vcfyy,vcfzz = loadtxt('vcf.out', usecols = (0,1,5,9), unpack=True)
    vcfxy,vcfxz,vcfyx,vcfyz,vcfzx,vcfzy = loadtxt('vcf.out', usecols = (2,3,4,6,7,8), unpack=True)
    vcfdiag = mean([vcfxx,vcfyy,vcfzz], axis=0)
    vcfodiag = mean([vcfxy,vcfxz,vcfyx,vcfyz,vcfzx,vcfzy], axis=0)
    plot(t,vcfdiag,'b-',t,vcfodiag,'g-')
    show()
#   Diffusion coefficient from VCF
    t,D = loadtxt('vcf_diff.out', unpack=True)
    Davg = mean(D)
    plot(t,D,'b-',t,Davg*ones(len(t)),'r-')
    show()
#   Mean square displacement
    t,msd = loadtxt('msd.out', unpack=True)
    plot(t,msd,'b.')
    #show()
#   Diffusion coefficient from MSD
    t,Dfit,D = loadtxt('msd_diff.out', unpack=True)
    plot(t,Dfit,'r-')
    show()

# read charges file: it stores: Nr-> Number of particles, q -> charge or charges of particles
def read_q(N, conf_in='q.b'):
    import pickle
    q=0
    with open(conf_in, 'rb') as ftrj:
        Nr = pickle.load(ftrj)
        if N!=Nr :
            print('Bad Initial charges: %d charges expected, %d declared' % (N,Nr) )
            exit(1)
            q = pickle.load( ftrj)
    return q

# dump position and momenta
def write_q(q, N, conf_out='q.b'):
  import pickle
  with open(conf_out, 'wb') as ftrj:
      pickle.dump( N , ftrj, pickle.HIGHEST_PROTOCOL)
      pickle.dump( q, ftrj, pickle.HIGHEST_PROTOCOL)
      


def VV_mdlj(nstep=1000, rho=0.84, m=6, kt=0.694, dt=0.005, freq=1, mode=0):
    """ Modena 19/04/2015 - LJMD  3D-version
    Numerical integration of Newton's equations for N interacting LJ-particles
    in a 3 dimensional space using reduced LJ units (mass,sigma,epsilon):
    Dynamics: microcanonical - velocity Verlet algorithm;
    INPUT:
      initial configuration from file 'conf_in' (binary):
      - 1th row:  x_1 y_1 z_1 (initial coordinates of 1st particle)
      -  . . .
      - Nth row   x_N y_N z_N (initial coordinates of Nth particle)
      - N+1th row:  px_1 py_1 pz_1 (initial momenta of 1st particle)
      -  . . .
      - 2*Nth row   px_N py_N pz_N (initial momenta of Nth particle)
      if mode is 0 initial positions are sampled from FCC lattice and previous input/output files are removed
      if mode is not 0 initial positions are read from configuration file
      if mode is even initial velocities are extracted from Maxwell Distribution and previous output files are removed
      if mode is not 0 and odd the run is restarted from previous configurations
        in this case correlation functions and MSD are computed using the whole run
    OUTPUT:
    - enep          total potential energy  E_p
    - vir_p         virial coefficient
    - enek          total kinetic energy E_k
    - enet          total energy Ep+Ek (constant of motion for microcanonical dynamics)
    - vcm(x,y,z)    center of mass momentum (constant of motion)
    PLOT:
    - Energies      total, potential and kinetic energy for the whole run vs. timestep
    - Temperature   Mean temperature computed as a running average vs. timestep
    - RDF           Radial distribution function as a function of interparticle distance
    - VCF           Diagonal and off-diagonal average of velocity correlation tensor
    - VCF Diffusion Diffusion coefficient as an integral of diagonal average of VCF
    - MSD           Mean square displacement
    - MSD Diffusion Diffusion coefficient as a function of the angular coefficient of the MSD
    """
    from numpy import random, sqrt, sum, mean, abs
    from MDLJ_lib import MDLJ_lib

    a=(4/rho)**(1./3.)  # Lattice constant [ particles' mass = 1 ]
    Lref=a*m            # Lattice constant * m = side of the cubic MD-box
    N=4*m**3            # number of particles: see {MDLJ_lib:fcc}
    md=MDLJ_lib(N, Lref)    # see {MDLJ_lib}
    g=3*N-3;            # number of dof
    gdr_out='gdrmd.out'
    print( "# initial (kinetic) temperature kt = %8.4f" % kt )
    print( "# integration time step dt = %8.4f" % dt )
    print( "# number of time steps for this run ns = %d" % nstep )
    print( "# side of the cubic MD-box L = %8.4f " % Lref )
    print( "# number of particles  N = %d" % N )
    print( "# density rho = %8.4f"  % rho )
    if mode :
        md.read_input(N, conf_in='conf_in.b')
        print("# initial configuration read from config file ")
    else :
        # initial positions mode=0 from FCC lattice
        md.clean_run()
        md.fcc(m)
        # initial velocities: realization of random gaussian process with zero mean
        # so that initial momentum is exactly zero at the initial time
        # this results in a stationary center of mass for the N particle system
    if mode%2 == 0:
        # if mode is even (re)samples initial velocities from gaussian
        md.clean_run(keepinput=True)
        md.t=0
        pstd=sqrt(md.mass*kt) # std dev of p
        md.px = random.normal(0., pstd, N)
        md.py = random.normal(0., pstd, N)
        md.pz = random.normal(0., pstd, N)
        vcmx  = sum(md.px)
        vcmy  = sum(md.py)
        vcmz  = sum(md.pz)
        md.px   -= vcmx/N
        md.py   -= vcmy/N
        md.pz   -= vcmz/N
        # @CAREFUL
        # coordinates are rescaled to be in [-.5,.5]: R :-> r = R/Lref 
        # Rescale coordinates => Rescale momenta: P :-> p = P*md.L  md.L is the simbox size (Lref -> see line:45)
        md.px *= md.L
        md.py *= md.L
        md.pz *= md.L
        print("# initial momenta sampled from maxwellian at temperature %8.4f " % kt)
    # ofstream eout("outmd.txt");

    # Run Integrator
    tt, ekt, ept, vir, ektsq, eptsq, etsq = md.vel_verlet(N, nstep, dt, freq)
    # End run

    DEiE = sqrt(etsq/nstep - ((ekt + ept)/nstep)**2)/abs((ekt+ept)/nstep)
    pres = (2*ekt + vir)/(3.*md.L**3)
    print( "# END OF THE RUN: Dumping average quantities")
    print( "# average potential energy  ep   = %10.5g " % (ept/nstep) )
    print( "# average kinetic   energy  ek   = %10.5g " % (ekt/nstep) )
    print( "# average total     energy  et   = %10.5g " % ((ekt+ept)/nstep) )
    print( "# average pressure          pres = %10.5g" %  (pres/nstep) )
    print( "# average temperature       kT   = %10.5g" %(2.*ekt/(nstep*g)) )
    print( "# average total energy relative fluctuations: DE/E = %.4g" % DEiE)
    # gdr final printout - use actual value of density rho
    md.write_gdr( N, tt, rho, gdr_out )
    md.calc_vcf( N, dt, freq )
    md.calc_MSD( N, dt, freq )
#   end of md - visualization:
    from matplotlib.pyplot import plot, show
    from numpy import loadtxt,ones,size
#   Energies vs. timestep
    t,enp,enk,ent = loadtxt('run.out', usecols = (0,1,2,4), unpack=True)
    plot(t,enk,'b-',t,enp,'g-',t,ent/N,'r-')
    show()
#   Mean Temperature vs. timestep
    t,itemp,avgtemp = loadtxt('temperature.out', unpack=True)
    plot(t,itemp,'r-',t,avgtemp,'b.')
    show()
#   Radial distribution function
    r,gdr = loadtxt(gdr_out, unpack=True, skiprows=1 )
    plot(r,gdr,'b.',r,ones(size(r)),'k-')
    show()
#   Velocity correlation functions
    t,vcfxx,vcfyy,vcfzz = loadtxt('vcf.out', usecols = (0,1,5,9), unpack=True)
    vcfxy,vcfxz,vcfyx,vcfyz,vcfzx,vcfzy = loadtxt('vcf.out', usecols = (2,3,4,6,7,8), unpack=True)
    vcfdiag = mean([vcfxx,vcfyy,vcfzz], axis=0)
    vcfodiag = mean([vcfxy,vcfxz,vcfyx,vcfyz,vcfzx,vcfzy], axis=0)
    plot(t,vcfdiag,'b-',t,vcfodiag,'g-')
    show()
#   Diffusion coefficient from VCF
    t,D = loadtxt('vcf_diff.out', unpack=True)
    Davg = mean(D)
    plot(t,D,'b-',t,Davg*ones(len(t)),'r-')
    show()
#   Mean square displacement
    t,msd = loadtxt('msd.out', unpack=True)
    plot(t,msd,'b.')
    #show()
#   Diffusion coefficient from MSD
    t,Dfit,D = loadtxt('msd_diff.out', unpack=True)
    plot(t,Dfit,'r-')
    show()
