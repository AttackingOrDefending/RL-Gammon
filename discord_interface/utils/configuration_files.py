import sys
from pathlib import Path
def load_configurations():

    d = {}

    with open('parameters.conf','r') as f:
        lines = f.readlines()

    for l in lines:

        if '#' in l:
            l = l.split('#')[0]

        if '=' in l:
            try:
                variable, value = l.split('=')

                variable = variable.strip()
                value = value.strip()

                if value.lower() in ['true', 'yes']:
                    value = True
                elif value.lower() in ['false', 'no']:
                    value = False
                elif value.isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                elif value.replace(' ','') == '':
                    value = ''

                d[variable] = value
            except Exception:
                import traceback
                traceback.print_exc()

    return d