"""Support code for running unit tests"""

import unittest
import lsst.daf.base as dafBase
import os
import sys

try:
    type(memId0)
except NameError:
    memId0 = 0                          # ignore leaked blocks with IDs before memId0
    nleakPrintMax = 20                  # maximum number of leaked blocks to print

def init():
    global memId0
    memId0 = dafBase.Citizen_getNextMemId()  # used by MemoryTestCase

def run(suite, exit=True):
    """Exit with the status code resulting from running the provided test suite"""
    status = 0 if unittest.TextTestRunner().run(suite).wasSuccessful() else 1
    if exit:
        sys.exit(status)
    else:
        return status

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        
class MemoryTestCase(unittest.TestCase):
    """Check for memory leaks since memId0 was allocated"""
    def setUp(self):
        pass

    def testLeaks(self):
        """Check for memory leaks in the preceding tests"""

        global memId0, nleakPrintMax
        nleak = dafBase.Citizen_census(0, memId0)
        if nleak != 0:
            print "\n%d Objects leaked:" % dafBase.Citizen_census(0, memId0)
            
            if nleak <= nleakPrintMax:
                print dafBase.Citizen_census(dafBase.cout, memId0)
            else:
                census = dafBase.Citizen_census_for_swig()
                print "..."
                for i in range(nleakPrintMax - 1, -1, -1):
                    print census[i].repr()
                
            self.fail("Leaked %d blocks" % dafBase.Citizen_census(0, memId0))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def findFileFromRoot(ifile):
    """Find file which is specified as a path relative to the toplevel directory;
    we start in $cwd and walk up until we find the file (or throw IOError if it doesn't exist)

    This is useful for running tests that may be run from mwi/tests or mwi"""
    
    if os.path.isfile(ifile):
        return ifile

    ofile = None
    file = ifile
    while file != "":
        dirname, basename = os.path.split(file)
        if ofile:
            ofile = os.path.join(basename, ofile)
        else:
            ofile = basename

        if os.path.isfile(ofile):
            return ofile

        file = dirname

    raise IOError, "Can't find %s" % ifile