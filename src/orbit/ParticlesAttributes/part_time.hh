/////////////////////////////// -*- C++ -*- //////////////////////////////
//
// FILE NAME
//   MyAttr.hh
//
// AUTHOR
//    T. Gorlov
//
// CREATED
//    07/14/2005
//
// DESCRIPTION
//    A subclass of the particle attributes class. This is container
//    for a macrosize of macro-particles in the bunch.
//
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
//
// INCLUDE FILES
//
///////////////////////////////////////////////////////////////////////////
#ifndef PART_TIME_HH_
#define PART_TIME_HH_

///////////////////////////////////////////////////////////////////////////
//
// INCLUDE FILES
//
///////////////////////////////////////////////////////////////////////////
#include <string>

///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//     WaveFunctionAmplitudes
//
///////////////////////////////////////////////////////////////////////////
#include "ParticleAttributes.hh"

class part_time : public ParticleAttributes
{
public:
  //--------------------------------------
  //the public methods of the ParticleMacroSize class
  //--------------------------------------
	
	/** Constructor. This Attribute describe complex coefficients of Wave functions.
	  * The defailt size is 400. 
		*/
	part_time(Bunch* bunch);
	
	/** This Attribute describe complex coefficients of Wave functions.
	  * User can specify the number of variables that he wants to reserve.
		*/
	part_time(Bunch* bunch, int size_in);
	
  ~part_time();
  
	
  int getAttSize();
	
private:
	int size;
	
	
};

///////////////////////////////////////////////////////////////////////////
//
// END OF FILE
//
///////////////////////////////////////////////////////////////////////////


#endif /*PRF_TIME_HH_*/