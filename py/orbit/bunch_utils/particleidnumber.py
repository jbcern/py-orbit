#!/usr/bin/env python

"""
"""
import math
import random
import sys
from bunch import Bunch
import orbit_mpi
from orbit_mpi import mpi_comm
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op

class ParticleIdNumber:
	""" 
	This routine adds id numbers to particle in a bunch.
	"""
	
	@staticmethod
	def addParticleIdNumbers(b, fixedidnumber = -1, startidnumber=1):

		rank = 0
		numprocs = 1
		
		mpi_init = orbit_mpi.MPI_Initialized()
		comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
		
		if(mpi_init):
			rank = orbit_mpi.MPI_Comm_rank(comm)
			numprocs = orbit_mpi.MPI_Comm_size(comm)
		
		nparts_arr_local = []
		for i in range(numprocs):
			nparts_arr_local.append(0)
				
		nparts_arr_local[rank] = b.getSize()
		data_type = mpi_datatype.MPI_INT
		op = mpi_op.MPI_SUM
	
		nparts_arr = orbit_mpi.MPI_Allreduce(nparts_arr_local,data_type,op,comm)

		if(b.hasPartAttr("ParticleIdNumber")==0):
			b.addPartAttr("ParticleIdNumber")

		if(fixedidnumber >= 0):
			for i in range(b.getSize()):
				b.partAttrValue("ParticleIdNumber", i, 0, fixedidnumber)
				
		else:
			istart = startidnumber
			if(rank!=0):
				for i in range(rank):
					istart = istart + nparts_arr[i]
			n=0
			for i in range(b.getSize()):
				if(b.partAttrValue("ParticleIdNumber", i, 0)==0):
					idnumber = istart + n
					b.partAttrValue("ParticleIdNumber", i, 0, idnumber)
					n += 1
				
