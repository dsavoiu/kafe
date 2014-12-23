'''
.. module:: __version_info
   :platform: Unix
   :synopsis: Version 0.5.3 of kafe, release Dec. 2014

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
                  Guenter Quast <g.quast@kit.edu>

'''

major = 0
minor = 5
revision = 3

def get_version():
  '''
  kafe version 0.5.3
  '''
  return str(major)+'.'+str(minor)+'.'+str(revision)
