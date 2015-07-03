'''
.. module:: __version_info
   :platform: Unix
   :synopsis: Version 0.5.4 of kafe, release Jan. 2015

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
                  Guenter Quast <g.quast@kit.edu>

'''

major = 1
minor = 0
revision = 0

def _get_version_tuple():
  '''
  kafe version 1.0.0
  '''
  return (major, minor, revision)

def _get_version_string():
  '''
  kafe version 1.0.0
  '''
  return "%d.%d.%d" % _get_version_tuple()

