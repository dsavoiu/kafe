'''
.. module:: _version_info
   :platform: Unix
   :synopsis: Version 1.2.0 of kafe, release Jun. 2016

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
                  Guenter Quast <g.quast@kit.edu>
'''

major = 1
minor = 2
revision = 0

def _get_version_tuple():
  '''
  kafe version as a tuple
  '''
  return (major, minor, revision)

def _get_version_string():
  '''
  kafe version as a string
  '''
  return "%d.%d.%d" % _get_version_tuple()

