import math
import subprocess as sb
import time
import numpy as np
import casacore.tables as ctab
from casacore.measures import measures
from casacore.quanta import quantity
from calibration_tools import *
from astropy.io import fits
import astropy.time as atime
import casa_io

# This is a stripped down version of smart-calibration/calibration/generate_data.py

# executables
makems_binary='/home/sarod/scratch/software/bin/makems'
sagecal='/home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal_gpu'
# imager
excon='/home/sarod/work/excon/src/MS/excon'

# LOFAR core coords
X0='3826896.235129999928176m'
Y0='460979.4546659999759868m'
Z0='5064658.20299999974668m'

# find a valid target direction
# the strategy for sky model generation
# valid field: target is > 10 deg,  above horizon
# output :ra0,dec0,t0
def find_valid_target():
    # epoch coordinate UTC 
    mydm=measures()
    x=X0
    y=Y0
    z=Z0
    mypos=mydm.position('ITRF',x,y,z)
    lowest_elevation=20 # lowest elevation for a target
    valid_field=False
    # loop till we find a valid direction (above horizon) and epoch
    while not valid_field:
      # field coords (rad)
      ra0=np.random.rand(1)*math.pi*2
      dec0=np.random.rand(1)*math.pi/2
      ra0=ra0[0]
      dec0=dec0[0]

      myra=quantity(str(ra0)+'rad')
      mydec=quantity(str(dec0)+'rad')
      mydir=mydm.direction('J2000',myra,mydec)
      # always use a fixed reference time to add random offset
      # format: (Year, Month, Day, Hour, Minute, Second, Day of Week, Day of Year, DST)
      start_ = (2000, 3, 9, 10, 0, 0, -1, -1, -1)
      t0=time.mktime(start_)+np.random.rand()*24*3600.0
      mytime=mydm.epoch('UTC',str(t0)+'s')
      mydm.doframe(mytime)
      mydm.doframe(mypos)
      # check elevation and field is above horizon, lowest_elevation deg above
      azel=mydm.measure(mydir,'AZEL')
      myel=azel['m1']['value']/math.pi*180

      if myel>lowest_elevation:
          valid_field=True

    return ra0,dec0,t0


## adds an extra column to an MS (same as in smart-calibration)
def add_column(msname,colname):
  tt=ctab.table(msname,readonly=False)
  cl=tt.getcol('DATA')
  (nrows,nchans,npols)=cl.shape
  vl=np.zeros(shape=cl.shape,dtype='complex64')
  dmi=tt.getdminfo('DATA')
  dmi['NAME']=colname
  mkd=ctab.maketabdesc(ctab.makearrcoldesc(colname,shape=np.array(np.zeros([nchans,npols])).shape,valuetype='complex',value=0.))
  tt.addcols(mkd,dmi)
  tt.putcol(colname,vl)
  tt.close()

# Simulate a LOFAR HBA observation at freq f0, 
# and generate training data (for one timeslot only Ts=1)
# input: K: directions for calibration + target (so total is K, not K+1)
# n_sol: number of solutions
# n_sol_interval: time slots per solution (integration time of each time slot is 1s)
# t_integration: integration time 
# seed: if given, set random seed to this value
# do_influence: if True, calculate influence function
# frequency: central frequency for the simulation
# returns: (when n_sol=n_sol_interval=1)
# stations N (=44), baselines =N(N-1)/2
# data: B*8
# uvw: B*3 : wavelengths
# solutions: K*N*8
# sky: K x (centroid ra,dec,sI(sum)): K*3
# eigenvalues of Jacobian (8B x 8B matrix): B*8 (if do_influence=True)
# residual: B*8
# MS name: for use in evaluation 
# columns: DATA: raw data, MODEL_DATA: calibration residual, CORRECTED_DATA: influence function
def generate_training_data(K=4,n_sol=1,n_sol_interval=1,t_integration=1.0,seed=None,do_influence=True,frequency=150e6):
    do_images=False
    do_solutions=True
    
    # Full time duration (slots), multiply with -t Tdelta option for full duration
    Ts=n_sol # integer, full duration in solution slots
    Tdelta=n_sol_interval # integer, solution interval in time slots
    # integration time (s)
    Tint=t_integration

    if seed is not None:
        np.random.seed(seed)

    ra0,dec0,t0=find_valid_target()
    
    # now we have a valid ra0,dec0 and t0 tuple
    strtime=time.strftime('%Y/%m/%d/%H:%M:%S',time.gmtime(t0))
    
    hh,mm,ss=radToRA(ra0)
    dd,dmm,dss=radToDec(dec0)
    
    atable='data/ANTENNA_14'
    
    # get antennas
    tt=ctab.table(atable,readonly=True)
    N=tt.nrows()
    tt.close()

    # frequency (Hz)
    f0=frequency
    # speed of light
    c_const=299792458.0
    
    # generate makems config
    # need to have both makems.parset and makems.cfg present
    makems_parset='makems.parset'
    msout='test.MS'
    ff=open(makems_parset,'w+')
    ff.write('NParts=1\n'
      +'NBands=1\n'
      +'NFrequencies=1\n'
      +'StartFreq='+str(f0)+'\n'
      +'StepFreq=180e3\n'
      +'StartTime='+strtime+'\n'
      +'StepTime='+str(Tint)+'\n'
      +'NTimes='+str(Ts*Tdelta)+'\n'
      +'RightAscension='+str(hh)+':'+str(mm)+':'+str(int(ss))+'\n'
      +'Declination='+str(dd)+'.'+str(dmm)+'.'+str(int(dss))+'\n'
      +'WriteAutoCorr=T\n'
      +'AntennaTableName=./'+str(atable)+'\n'
      +'MSName='+str(msout)+'\n'
    )
    ff.close()
    sb.run('cp '+makems_parset+' makems.cfg',shell=True)
    sb.run(makems_binary,shell=True)
    
    # output will be msout_p0
    msoutp0=msout+'_p0'
    
    sb.run('rsync -a ./data/FIELD '+msoutp0+'/',shell=True)
    # update FIELD table
    field=ctab.table(msoutp0+'/FIELD',readonly=False)
    delay_dir=field.getcol('DELAY_DIR')
    phase_dir=field.getcol('PHASE_DIR')
    ref_dir=field.getcol('REFERENCE_DIR')
    lof_dir=field.getcol('LOFAR_TILE_BEAM_DIR')
    
    ci=0
    delay_dir[ci][0][0]=ra0
    delay_dir[ci][0][1]=dec0
    phase_dir[ci][0][0]=ra0
    phase_dir[ci][0][1]=dec0
    ref_dir[ci][0][0]=ra0
    ref_dir[ci][0][1]=dec0
    lof_dir[ci][0]=ra0
    lof_dir[ci][1]=dec0
    
    field.putcol('DELAY_DIR',delay_dir)
    field.putcol('PHASE_DIR',phase_dir)
    field.putcol('REFERENCE_DIR',ref_dir)
    field.putcol('LOFAR_TILE_BEAM_DIR',lof_dir)
    field.close()
    
    sb.run('rsync -a ./data/LOFAR_ANTENNA_FIELD '+msoutp0+'/',shell=True)
    
    # remove old files
    sb.run('rm -rf L_SB*.MS L_SB*fits',shell=True)
    
    MS='L_SB'+str(ci)+'.MS'
    sb.run('rsync -a '+msoutp0+'/ '+MS,shell=True)
    
    #########################################################################
    # sky model/error simulation
    # simulate target field and K-1 outliers 
    # Sources (directions) used in calibration: 
    # first one for center, 1,2,3,.. for outlier clusters
    # and last one for weak sources (so minimum 2), 3 will be the weak sources

    outskymodel='sky0.txt' # for simulation
    outskymodel1='sky.txt' # for calibration
    outcluster='cluster0.txt' # for simulation
    outcluster1='cluster.txt' # for calibration
    ff=open(outskymodel,'w+')
    ff1=open(outskymodel1,'w+')
    gg=open(outcluster,'w+')
    gg1=open(outcluster1,'w+')

    # sky model output: ra,dec,flux of each cluster
    sky=np.zeros((K,3),dtype=np.float32)

    lmin=0.8 # range of full simulation
    l0min=0.05 # range of each cluster
    l=(np.random.rand(K)-0.5)*lmin
    m=(np.random.rand(K)-0.5)*lmin

    # number of sources for each cluster in [4,8]
    Kc=np.random.randint(4,8)
 
    for clus in range(K):
       # generate random sources in [-lmin,lmin] at the cluster center
       a=0.1
       b=200.0#  flux in [a b]
       alpha=-2 # power law index
       sIuniform=np.random.rand(Kc)
       sI=np.power(np.power(a,(alpha+1))+sIuniform*(np.power(b,(alpha+1))-np.power(a,(alpha+1))),(1/(alpha+1)))
       # spectral indices
       sP=np.random.randn(Kc)
       sky[clus,2]=np.sum(sI)

       # output sources for cluster
       # format: P0 19 59 47.0 40 40 44.0 1.0 0 0 0 -1 0 0 0 0 0 0 1000000.0
       gg.write(str(clus+1)+' 1')
       gg1.write(str(clus+1)+' 1') 
    
       l0=(np.random.rand(Kc)-0.5)*l0min+l[clus]
       m0=(np.random.rand(Kc)-0.5)*l0min+m[clus]

       ra_c,dec_c=lmtoradec(l[clus],m[clus],ra0,dec0)
       sky[clus,0]=ra_c
       sky[clus,1]=dec_c

       for cj in range(Kc):
         ra,dec=lmtoradec(l0[cj],m0[cj],ra0,dec0)
         hh,mm,ss=radToRA(ra)
         dd,dmm,dss=radToDec(dec)
         sname='PC'+str(clus)+'_'+str(cj)
         ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj])+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
         ff1.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj])+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
         gg.write(' '+sname)
         gg1.write(' '+sname)
    
       gg.write('\n')
       gg1.write('\n')

    
    # add background sky only to simulation model
    ff.write('# weak sources\n')
    gg.write('# cluster for fixed background sources\n')
    gg.write(str(K+1)+' 1 ')
    diffuse_ra0=0
    diffuse_dec0=math.pi/2
    hh,mm,ss=radToRA(diffuse_ra0)
    dd,dmm,dss=radToDec(diffuse_dec0)
    
    shapelet_sI=10 # scale up because beta is larger
    sname='SLSI'
    #generate_random_shapelet_model(sname+'.fits.modes',hh,mm,ss,dd,mm,ss)
    ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(shapelet_sI)+' 0 0 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
    gg.write(str(sname)+' ')
    sname='SLSQ'
    #generate_random_shapelet_model(sname+'.fits.modes',hh,mm,ss,dd,mm,ss)
    ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' 0.0 '+str(shapelet_sI)+' 0 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
    gg.write(str(sname)+' ')
    sname='SLSU'
    #generate_random_shapelet_model(sname+'.fits.modes',hh,mm,ss,dd,mm,ss)
    ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' 0.0 0.0 '+str(shapelet_sI)+' 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
    gg.write(str(sname)+' ')
    # second model with slight offset
    hh,mm,ss=radToRA(diffuse_ra0+0.1)
    dd,dmm,dss=radToDec(diffuse_dec0-0.05)
    sname='SLSI1'
    ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(shapelet_sI)+' 0 0 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
    gg.write(str(sname)+' ')
    sname='SLSQ1'
    ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' 0.0 '+str(shapelet_sI)+' 0 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
    gg.write(str(sname)+' ')
    sname='SLSU1'
    ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' 0.0 0.0 '+str(shapelet_sI)+' 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
    gg.write(str(sname))

    # additional weak point sources, using a fixed random generator
    Kw=4000
    rng=np.random.default_rng(seed=33)
    # generate random sources in [-lmin,lmin] at the cluster center
    a=0.01
    b=0.05 #  flux in [a b]
    alpha=-2 # power law index
    sIuniform=rng.random(Kw)
    sI=np.power(np.power(a,(alpha+1))+sIuniform*(np.power(b,(alpha+1))-np.power(a,(alpha+1))),(1/(alpha+1)))
    # spectral indices
    sP=rng.random(Kw)

    l0min=0.5
    l0=(rng.random(Kw)-0.5)*l0min
    m0=(rng.random(Kw)-0.5)*l0min

    for cj in range(Kw):
         ra,dec=lmtoradec(l0[cj],m0[cj],diffuse_ra0,diffuse_dec0)
         hh,mm,ss=radToRA(ra)
         dd,dmm,dss=radToDec(dec)
         sname='PW'+str(clus)+'_'+str(cj)
         ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj])+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
         gg.write(' '+sname)
    
    gg.write('\n')

    ff.close()
    ff1.close()
    gg.close()
    gg1.close()

    #########################################################################
    # simulate errors for K directions, attenuate those errors
    # target = column K-1
    # outlier = columns 0..K-2
    if do_solutions:
       # storage for full solutions
       gs=np.zeros((K,8*N*Ts,1),dtype=np.float32)
    
       # normalized freqency
       norm_f=1
    
       for ck in range(K):
         # attenuate random seed
         gs[ck,0:8*N,0]=np.random.randn(8*N)*0.01
         # also add 1 to J_00 and J_22 (real part) : every 0 and 6 value
         gs[ck,0:8*N:8] +=1.
         gs[ck,6:8*N:8] +=1.
    
       # now the 1-st timeslot solutions for all freqs are generated
       # copy this to other timeslots
       for ck in range(K):
         for ct in range(1,Ts):
           gs[ck,ct*8*N:(ct+1)*8*N]=gs[ck,0:8*N]
    
       # open all files
       flist={}
       MS='L_SB'+str(ci)+'.MS'
       flist[0]=open(MS+'.S.solutions','w+')
    
       flist[0].write('#solution file created by simulate.py for SAGECal\n')
       flist[0].write('#freq(MHz) bandwidth(MHz) time_interval(min) stations clusters effective_clusters\n')
       flist[0].write(str(f0/1e6)+' 0.183105 20.027802 '+str(N)+' '+str(K+1)+' '+str(K+1)+'\n')
    
    
       for ct in range(Ts):
         for ci in range(8*N):
           stat=ci//8
           offset=ci-8*stat
           flist[0].write(str(ci)+' ')
           for ck in range(K):
               flist[0].write(str(gs[ck,ct*8*N+ci,0])+' ')
              # last column, 1 at 0 and 6, else 0
           if offset==0 or offset==6:
               flist[0].write('1\n')
           else:
               flist[0].write('0\n')
    
       flist[0].close()
    #########################################################################
    # simulation
    fi=0
    MS='L_SB'+str(fi)+'.MS'

    if do_images:
        # first only simulate background including noise
        ignorelist='ignorelist.txt' # which clusters to ignore when simulating background - only works with -p solutions are provided
        ff=open(ignorelist,'w+')
        for ck1 in range(K):
          ff.write(str(ck1+1)+'\n')
        ff.close()
        sb.run(sagecal+' -z '+str(ignorelist)+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -S 128 -O DATA -a 1 -B 2 -E 1 -p '+MS+'.S.solutions > simulation_orig.out',shell=True)
        hh,mm,ss=radToRA(diffuse_ra0)
        dd,dmm,dss=radToDec(diffuse_dec0)
        sb.run(excon+' -P '+str(hh)+','+str(mm)+','+str(ss)+','+str(dd)+','+str(dmm)+','+str(dss)+' -m '+MS+' -p 700 -x 2 -c DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 400 -Q orig_background > /dev/null',shell=True)

    if do_solutions:
        sb.run(sagecal+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -S 128 -a 1 -B 2 -E 1 -p '+MS+'.S.solutions > simulation.out',shell=True)
    else:
        sb.run(sagecal+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -S 128 -a 1 -B 2 -E 1 > simulation.out',shell=True)

    # noise power ||noise||^2
    NP=0.01
    sb.run('python add_unscaled_noise.py '+MS+' '+str(NP),shell=True)

    if do_images:
        sb.run(excon+' -P '+str(hh)+','+str(mm)+','+str(ss)+','+str(dd)+','+str(dmm)+','+str(dss)+' -m '+MS+' -p 700 -x 2 -c DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 400 > /dev/null',shell=True)
    
    # calibration without consensus (with increased iterations)
    sb.run(sagecal+' -d '+str(MS)+' -s sky.txt -c cluster.txt -I DATA -O MODEL_DATA -p zsol -e 6 -g 10 -n 6 -t '+str(Tdelta)+' -S 128 -B 2 -E 1 > calibration.out',shell=True)

    # read solutions (for all time slots)
    freq0,J=readsolutions('zsol')
    assert(J.shape[0]==K)
    assert(J.shape[1]==2*N*Ts)
    assert(J.shape[2]==2)
    # convert to real tensor: 2*K x 2*N x 2
    Jsol=np.concatenate((J.real, J.imag))
    
    # influence function calculation
    if do_influence:
       sb.run(sagecal+' -d '+MS+' -s sky.txt -c cluster.txt -I DATA -O CORRECTED_DATA -n 6 -e 6 -g 10 -B 2 -S 128 -E 1 -i 1 -t '+str(Tdelta)+' > influence.out',shell=True)

    if do_images:
        sb.run(excon+' -P '+str(hh)+','+str(mm)+','+str(ss)+','+str(dd)+','+str(dmm)+','+str(dss)+' -m '+MS+' -p 700 -x 2 -c MODEL_DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 400 -Q residual > /dev/null',shell=True)
        hh,mm,ss=radToRA(diffuse_ra0)
        dd,dmm,dss=radToDec(diffuse_dec0)
        sb.run(excon+' -P '+str(hh)+','+str(mm)+','+str(ss)+','+str(dd)+','+str(dmm)+','+str(dss)+' -m '+MS+' -p 700 -x 2 -c MODEL_DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 400 -Q background > /dev/null',shell=True)

    # extract information from MS (eigenvalues, before shifting)
    # (corr_data=eigenvalues of influence func, model_data=residual)
    n_stat,uvw,obs_data,corr_data,model_data=casa_io.read_corr_full(MS)

    # do not shift phase center
    hh,mm,ss=radToRA(diffuse_ra0)
    dd,dmm,dss=radToDec(diffuse_dec0)
    #sb.run(excon+' -P '+str(hh)+','+str(mm)+','+str(ss)+','+str(dd)+','+str(dmm)+','+str(dss)+' -m '+MS+' -J 1 > /dev/null',shell=True)
    # extract information from MS (after phase shifting)
    # (corr_data=eigenvalues of influence func, model_data=residual)
    #n_stat,uvw,obs_data,_,model_data=casa_io.read_corr_full(MS)
    assert(N==n_stat)
    # when inversion is not possible, eigenvalues 1-lambda becomes negative,
    # set them to very small value
    if do_influence:
       eigs=1-corr_data
       eigs[eigs<1e-6]=1e-6
       # sorted from low to high 
       logeigs=np.log(np.sort(eigs))
       # if eigs > 1 (due to noise), log(eigs) > 0, set them to 0
       logeigs[logeigs>0]=0
    else:
        logeigs=corr_data
    uvw *=(f0/c_const) # wavelengths
    # metadata: {n_stat(=N,B=N(N-1)/2), uvw(B*3), data(B*8), solutions(K*N*8), sky(K*3), logeig(B*8)}, residual(B*8), MS name
    # Note: flatten() vectorizes in row major order, so first row -> first 8 values etc.
    return n_stat, uvw.flatten(), obs_data.flatten(), Jsol.flatten(), sky.flatten(), logeigs.flatten(), model_data.flatten(), MS


# make image by writing the given array to given column of the given MS
# read the FITS files, return them as np arrays
def make_image(msname,ra0,dec0,data):
   # first write to MS
   casa_io.write_corr_full(msname,'MODEL_DATA',data)
   hh,mm,ss=radToRA(ra0)
   dd,dmm,dss=radToDec(dec0)
   sb.run(excon+' -P '+str(hh)+','+str(mm)+','+str(ss)+','+str(dd)+','+str(dmm)+','+str(dss)+' -m '+msname+' -p 50 -x 2 -u 6000 -c MODEL_DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 1200 -Q predict > /dev/null',shell=True)
   hdu=fits.open(msname+'_predict_I.fits')
   sI=np.squeeze(hdu[0].data[0])
   hdu.close()
   hdu=fits.open(msname+'_predict_Q.fits')
   sQ=np.squeeze(hdu[0].data[0])
   hdu.close()
   hdu=fits.open(msname+'_predict_U.fits')
   sU=np.squeeze(hdu[0].data[0])
   hdu.close()
   hdu=fits.open(msname+'_predict_V.fits')
   sV=np.squeeze(hdu[0].data[0])
   hdu.close()

   return sI,sQ,sU,sV

# make given columns of MS, without reading, attach qualifier to images
# if RA,Dec given, phase shift to this ra,dec
# if fullpol=True, make IQUV, elese I only
def make_image_col(msname,colname,qual,ra=None,dec=None,fullpol=False):
    if not fullpol:
        imgmode='-x 0'
    else:
        imgmode='-x 2'
    if ra is not None and dec is not None:
       hh,mm,ss=radToRA(ra)
       dd,dmm,dss=radToDec(dec)
       sb.run(excon+' -P '+str(hh)+','+str(mm)+','+str(ss)+','+str(dd)+','+str(dmm)+','+str(dss)+' -m '+msname+' -p 50 -c '+str(colname)+' -d 3000 -Q '+str(qual)+' '+imgmode+' > /dev/null',shell=True)
    else:
       sb.run(excon+' -m '+msname+' -p 50 -c '+str(colname)+' -d 3000 -Q '+str(qual)++' '+imgmode+' > /dev/null',shell=True)


# make model image by predicting the model to MS
# read the FITS files, return them as np arrays
def make_model_image(msname,ra0,dec0,K,tslots=1):
   Tdelta=tslots # make sure this agrees with MS
   # first predict model to MS
   ignorelist='ignorelist.txt' # which clusters to ignore when simulating background - only works with -p solutions are provided
   ff=open(ignorelist,'w+')
   for ck1 in range(K):
          ff.write(str(ck1+1)+'\n')
   ff.close()
   # run with full beam 
   sb.run(sagecal+' -z '+str(ignorelist)+' -d '+msname+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -S 128 -E 1 -p '+msname+'.S.solutions > simulation_orig.out',shell=True)

   hh,mm,ss=radToRA(ra0)
   dd,dmm,dss=radToDec(dec0)
   sb.run(excon+' -P '+str(hh)+','+str(mm)+','+str(ss)+','+str(dd)+','+str(dmm)+','+str(dss)+' -m '+msname+' -p 50 -x 2 -u 6000 -c DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 1200 -Q predict > /dev/null',shell=True)
   hdu=fits.open(msname+'_predict_I.fits')
   sI=np.squeeze(hdu[0].data[0])
   hdu.close()
   hdu=fits.open(msname+'_predict_Q.fits')
   sQ=np.squeeze(hdu[0].data[0])
   hdu.close()
   hdu=fits.open(msname+'_predict_U.fits')
   sU=np.squeeze(hdu[0].data[0])
   hdu.close()
   hdu=fits.open(msname+'_predict_V.fits')
   sV=np.squeeze(hdu[0].data[0])
   hdu.close()

   return sI,sQ,sU,sV


# copy fitsname to new_fitsname, write data
def write_to_fits(fitsname,new_fitsname,data):
   sb.run('/usr/bin/cp '+fitsname+' '+new_fitsname,shell=True)
   with fits.open(new_fitsname,mode='update') as hdu:
      hdu[0].data[0]=data.reshape(hdu[0].data[0].shape)
      hdu.flush()


# check quality of data to see if acceptable
def quality_ok(data,residual):
    MAX_RESIDUAL=100
    n_data=data.size
    if np.linalg.norm(residual) > np.linalg.norm(data):
        return False
    if np.linalg.norm(residual)/n_data > MAX_RESIDUAL:
        return False
    return True
 
if 0:
   K=6
   n_stat, uvw, data, sol, sky, logeigs, residual, msname=generate_training_data(K,n_sol=1,n_sol_interval=1,t_integration=10,seed=1,do_influence=True)
   B=n_stat*(n_stat-1)//2
   print(np.max(uvw)) # 1/1000
   assert(uvw.size==B*3)
   print(np.max(data)) # 100
   assert(data.size==B*8)
   print(np.max(sol)) # 1
   assert(sol.size==K*n_stat*8)
   print(np.max(sky)) # 1
   assert(sky.size==K*3)
   assert(logeigs.size==B*8)
   print(np.sum(logeigs))
   print(np.max(residual)) # 100
   assert(residual.size==B*8)
   import matplotlib.pyplot as plt
   fig=plt.figure()
   fig,axs=plt.subplots()
   axs.plot(logeigs,'bo',linestyle='')
   plt.savefig('logeigs.png',dpi=300)
