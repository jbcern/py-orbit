import math
import sys
import time
import orbit_mpi
import timeit
import numpy as np
import scipy.io as sio
import os

# utils
from orbit.utils.orbit_mpi_utils import bunch_orbit_to_pyorbit, bunch_pyorbit_to_orbit
from orbit.utils.consts import mass_proton, speed_of_light, pi

# bunch
from bunch import Bunch
from bunch import BunchTwissAnalysis, BunchTuneAnalysis
from orbit.bunch_utils import ParticleIdNumber

# diagnostics
from orbit.diagnostics import TeapotStatLatsNode, TeapotMomentsNode, TeapotTuneAnalysisNode
from orbit.diagnostics import addTeapotDiagnosticsNodeAsChild
from orbit.diagnostics import addTeapotMomentsNodeSet, addTeapotStatLatsNodeSet

# PTC lattice
from libptc_orbit import *
from ext.ptc_orbit import PTC_Lattice
from ext.ptc_orbit import PTC_Node
from ext.ptc_orbit.ptc_orbit import setBunchParamsPTC, readAccelTablePTC, readScriptPTC
from ext.ptc_orbit.ptc_orbit import updateParamsPTC, synchronousSetPTC, synchronousAfterPTC
from ext.ptc_orbit.ptc_orbit import trackBunchThroughLatticePTC, trackBunchInRangePTC
from orbit.aperture import TeapotApertureNode

# longitudinal space charge
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode, addLongitudinalSpaceChargeNodeAsChild, SC1D_AccNode
from spacecharge import LSpaceChargeCalc

# transverse space charge
from orbit.space_charge.sc2p5d import scAccNodes, scLatticeModifications
from spacecharge import SpaceChargeCalc2p5D, Boundary2D

# apertures 
from orbit.utils import orbitFinalize, NamedObject, ParamsDictObject
from orbit.aperture import addTeapotApertureNode
from orbit.aperture import TeapotApertureNode, CircleApertureNode, EllipseApertureNode, RectangleApertureNode
from orbit.aperture import addCircleApertureSet, addEllipseApertureSet, addRectangleApertureSet

# collimator
from orbit.collimation import TeapotCollimatorNode
from orbit.collimation import addTeapotCollimatorNode
from collimator import Collimator

# dictionary
from output_dictionary import *

print "Start ..."

#----------------------------------------------
# Create folder structure
#----------------------------------------------
from mpi_helpers import mpi_mkdir_p
#mpi_mkdir_p('Input')
mpi_mkdir_p('Output')

#----------------------------------------------
# Simulation Parameters
#----------------------------------------------

turn = 0
turns_max = 100
turns_print = xrange(-1, turns_max, 10)

#----------------------------------------------
# Initialize a Teapot-Style PTC lattice
#----------------------------------------------
PTC_File='Input/PSB_FLAT_NoPert.TXT'
Lattice = PTC_Lattice("PSB")
Lattice.readPTC(PTC_File)
readScriptPTC('ptc/fringe.txt')
readScriptPTC('ptc/time.txt')
readScriptPTC('ptc/chrom.txt')
readScriptPTC('ptc/ramp_magnet.ptc')
readScriptPTC('ptc/ramp_cavities.ptc')
readScriptPTC('ptc/energize_lattice.ptc')
readScriptPTC('ptc/twiss_script.ptc')
readScriptPTC('ptc/write_FINAL_SETTINGS.ptc')

paramsDict = {}
paramsDict["length"]=Lattice.getLength()/Lattice.nHarm
print '\nLattice parameters ...'
print '  circumference: \t', Lattice.getLength(), 'm'
print '  alphax0: \t\t', Lattice.alphax0
print '  betax0: \t\t', Lattice.betax0, 'm'
print '  alphay0: \t\t', Lattice.alphay0
print '  betay0: \t\t', Lattice.betay0, 'm'
print '  Dx0: \t\t\t', Lattice.etax0, 'm'
print '  Dpx0: \t\t', Lattice.etapx0, 'm'
print '  harm. number: \t', Lattice.nHarm
print '  nodes: \t\t', Lattice.nNodes

#----------------------------------------------
# Add apertures
#----------------------------------------------
position = 0
pos_start = 0
pos_stop  = Lattice.getLength()
n=0
for node in Lattice.getNodes():
   (node_pos_start,node_pos_stop) = Lattice.getNodePositionsDict()[node]
   print "node number=",n, "node=",node.getName()," type=",node.getType(),"  pos=",node_pos_start," L=",node.getLength()," end=",node_pos_start+node.getLength()
   myaperturenode = TeapotApertureNode(1, 10, 18, position)
   node.addChildNode(myaperturenode, node.EXIT)
   position += node.getLength()
   n += 1

#----------------------------------------------
# Add the main bunch and lost particles bunch
#----------------------------------------------
print '\nAdding main bunch ...'
Intensity = 170e+10
m0 = mass_proton 	# protons ...
N_mp = 100000
Particle_distribution_file = 'Input/INITIAL-6D.dat'

macrosize = Intensity/N_mp
bunch = Bunch()
setBunchParamsPTC(bunch)
kin_Energy = bunch.getSyncParticle().kinEnergy()
print '  Momentum: ', bunch.getSyncParticle().momentum(), 'GeV'
print '  Ekin:     ', bunch.getSyncParticle().kinEnergy(), 'GeV'
print '  Gamma:    ',bunch.getSyncParticle().gamma() 
print '  Beta:     ', bunch.getSyncParticle().beta()
print '  Charge:   ', bunch.charge(), 'e'
print '  Mass:     ', bunch.mass(), 'GeV'

bunch_orbit_to_pyorbit(paramsDict["length"], kin_Energy, Particle_distribution_file, bunch, N_mp + 1) #read in only first N_mp particles. 
print '  Number of particles in the bunch: ',bunch.getSizeGlobal()
ParticleIdNumber().addParticleIdNumbers(bunch) # Give particles unique number ids
bunch.dumpBunch("Output/mainbunch_start.dat")

bunch.addPartAttr("macrosize")
for i in range(bunch.getSize()):
	bunch.partAttrValue("macrosize", i, 0, macrosize)  #main bunch has finite macrosize for space charge

lostbunch = Bunch()
lostbunch.addPartAttr('ParticlePhaseAttributes')
lostbunch.addPartAttr("LostParticleAttributes")

paramsDict["lostbunch"] = lostbunch
paramsDict["bunch"]= bunch

#----------------------------------------------------
# Add transverse potential space charge node with
# rectangular boundary
#----------------------------------------------------
print '\nAdding longitudinal space charge ...'
b_a = 1.5
nMacrosMin = 32
useSpaceCharge = 1
nBins= 64     # Number of longitudinal slices in the 1D space charge solver
position = 1   # The location in the lattice. Can be any empty place
length = Lattice.getLength() 
sc1Dnode = SC1D_AccNode(b_a,length, nMacrosMin, useSpaceCharge, nBins)
nodes = Lattice.getNodes()
AccNode = nodes[1]
addLongitudinalSpaceChargeNodeAsChild(Lattice, AccNode, sc1Dnode)


print '\nAdding transverse space charge nodes ...'
sizeX = 128
sizeY = 128
sizeZ = 64  # Number of longitudinal slies in the 2.5D solver
calc2p5d = SpaceChargeCalc2p5D(sizeX,sizeY,sizeZ)
calc2p5d.setSmoothBinning(True) #smooth longitudinal binning flag
smooth_flag = calc2p5d.getSmoothBinning() #check the smooth binning option
print 'smooth_flag', smooth_flag
sc_path_length_min = 0.00000001

# Add the space charge solver to the lattice as child nodes
sc_nodes = scLatticeModifications.setSC2p5DAccNodes(Lattice, sc_path_length_min, calc2p5d)
print '  Installed', len(sc_nodes), 'space charge nodes ...'


#----------------------------------------------------
# Define twiss analysis and output dictionary
#----------------------------------------------------
bunchtwissanalysis = BunchTwissAnalysis() #Prepare the analysis class that will look at emittances, etc.
get_dpp = lambda b, bta: np.sqrt(bta.getCorrelation(5,5)) / (b.getSyncParticle().gamma()*b.mass()*b.getSyncParticle().beta()**2)
get_bunch_length = lambda b, bta: 4 * np.sqrt(bta.getCorrelation(4,4)) / (speed_of_light*b.getSyncParticle().beta())
get_eps_z = lambda b, bta: 1e9 * 4 * pi * bta.getEmittance(2) / (speed_of_light*b.getSyncParticle().beta())

output_file = 'Output/output.dat'
output = Output_dictionary()
output.addParameter('turn', lambda: turn)
output.addParameter('intensity', lambda: bunchtwissanalysis.getGlobalMacrosize())
output.addParameter('n_mp', lambda: bunchtwissanalysis.getGlobalCount())
output.addParameter('gamma', lambda: bunch.getSyncParticle().gamma())
output.addParameter('mean_x', lambda: bunchtwissanalysis.getAverage(0))
output.addParameter('mean_xp', lambda: bunchtwissanalysis.getAverage(1))
output.addParameter('mean_y', lambda: bunchtwissanalysis.getAverage(2))
output.addParameter('mean_yp', lambda: bunchtwissanalysis.getAverage(3))
output.addParameter('mean_z', lambda: bunchtwissanalysis.getAverage(4))
output.addParameter('mean_dE', lambda: bunchtwissanalysis.getAverage(5))
output.addParameter('epsn_x', lambda: bunchtwissanalysis.getEmittanceNormalized(0))
output.addParameter('epsn_y', lambda: bunchtwissanalysis.getEmittanceNormalized(1))
output.addParameter('eps_z', lambda: get_eps_z(bunch, bunchtwissanalysis))
output.addParameter('bunchlength', lambda: get_bunch_length(bunch, bunchtwissanalysis))
output.addParameter('dpp_rms', lambda: get_dpp(bunch, bunchtwissanalysis))
if os.path.exists(output_file):
	output.import_from_matfile(output_file)
	
#----------------------------------------------------
# Do some turns and dump particle information
#----------------------------------------------------
print '\n now start tracking...'

for turn in range(turn, turns_max):
	Lattice.trackBunch(bunch, paramsDict)
	bunchtwissanalysis.analyzeBunch(bunch)  # analyze twiss and emittance	
	output.update()
	if turn in turns_print:
		bunch.dumpBunch("Output/mainbunch_%s"%(str(turn).zfill(6)))
		lostbunch.dumpBunch("Output/lostbunch_%s"%(str(turn).zfill(6)))
		#saveBunchAsMatfile(bunch, "Output/mainbunch_%s"%(str(turn).zfill(6)))
		#saveBunchAsMatfile(lostbunch, "Output/lostbunch_%s"%(str(turn).zfill(6)))
		output.save_to_matfile(output_file)
		readScriptPTC('ptc/write_FINAL_SETTINGS.ptc')

