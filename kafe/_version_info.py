'''
.. module:: _version_info
   :platform: Unix
   :synopsis: Version 1.0.0 of kafe, release Jul. 2015

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
                  Guenter Quast <g.quast@kit.edu>

'''

major = 1
minor = 0
revision = 1

def _get_version_tuple():
  '''
  kafe version 1.0.1
  '''
  return (major, minor, revision)

def _get_version_string():
  '''
  kafe version 1.0.1
  '''
  return "%d.%d.%d" % _get_version_tuple()

