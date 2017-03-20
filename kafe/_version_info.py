'''
.. module:: _version_info
   :platform: Unix
   :synopsis: Version 1.3.0 of kafe, release Feb. 2017

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
                  Guenter Quast <g.quast@kit.edu>
'''

major = 1
minor = 3
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

