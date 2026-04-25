# MCAT DTL DRAGON Autofocus File

This sy.f/data.dat/fort.1xx bundle is used by MCAT to compute DTL autofocus from MEBT end to DRAGON start. Note, RF cavities are not represented, instead there are aphysical steps in E/A at cavity midpoints. The DTL E/A for each cavity is specified in data.dat. To turn off a cavity, set it's E/A to whatever the E/A downstream was set to. Make sure to follow the design energy configuration of the machine, or obtain nonsense results!~

Olivier
