'''
.. module:: _version_info
   :platform: Unix

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
                  Guenter Quast <g.quast@kit.edu>
'''

_version_info = dict(major=1, minor=3, revision=0,
                     is_development=False)

def _get_version_tuple():
  '''
  kafe version as a tuple
  '''
  return (_version_info['major'],
          _version_info['minor'],
          _version_info['revision'])

def _get_version_string():
  '''
  kafe version as a string
  '''
  if _version_info['is_development']:
      return "{major}.{minor}-devel".format(**_version_info)
  return "{major}.{minor}.{revision}".format(**_version_info)

