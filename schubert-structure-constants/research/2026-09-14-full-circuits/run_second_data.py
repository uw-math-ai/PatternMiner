"""Generate the reserved replacement assessment and disjoint training neighbors."""
import subprocess,sys
import session as s


for name in ['generate_second_populations','generate_second_neighbors','prepare_second_neighbors']:
    with (s.HERE/(name+'.log')).open('x') as stream:
        result=subprocess.run([sys.executable,str(s.HERE/(name+'.py'))],stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,name
print('Second assessment and neighboring training data completed.',flush=True)
