'''
Autor:      Ole Uphaus
Datum:     16.06.2025
Beschreibung:
Dies ist ein ganz kurzes skript, mit dem ich sites definieren werde, die dann in Mujoco als Trajektorien dienen sollen.
'''

import os

# Pfad erzeugen
script_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_path, 'traj_sites.xml')

# Datei erzeugen mit 100 Sites
with open(xml_path, "w") as f:
    for i in range(500):
        f.write(f'''  <body name="sitebody_des_{i}" pos="0 0 0">
    <site name="des_traj_{i}" type="sphere" size="0.005" rgba="0 0 1 1"/>
  </body>\n''')