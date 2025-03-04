cd barbell_clique
# RWNN-transformer
python3 -m synthexp.run_multiple --yaml_name barbell-rwnn-1000
python3 -m synthexp.run_multiple --yaml_name barbell-rwnn-500
python3 -m synthexp.run_multiple --yaml_name barbell-rwnn-200
python3 -m synthexp.run_multiple --yaml_name barbell-rwnn-100
python3 -m synthexp.run_multiple --yaml_name barbell-rwnn-50
python3 -m synthexp.run_multiple --yaml_name clique-rwnn-1000
python3 -m synthexp.run_multiple --yaml_name clique-rwnn-500
python3 -m synthexp.run_multiple --yaml_name clique-rwnn-200
python3 -m synthexp.run_multiple --yaml_name clique-rwnn-100
python3 -m synthexp.run_multiple --yaml_name clique-rwnn-50
# RWNN-MLP
python3 -m synthexp.run_multiple --yaml_name barbell-rwnn-mlp-1000
python3 -m synthexp.run_multiple --yaml_name barbell-rwnn-mlp-100
python3 -m synthexp.run_multiple --yaml_name clique-rwnn-mlp-1000
python3 -m synthexp.run_multiple --yaml_name clique-rwnn-mlp-100
# WalkLM-transformer
python3 -m synthexp.run_multiple --yaml_name barbell-walklm-1000
python3 -m synthexp.run_multiple --yaml_name barbell-walklm-100
python3 -m synthexp.run_multiple --yaml_name clique-walklm-1000
python3 -m synthexp.run_multiple --yaml_name clique-walklm-100
# CRaWl and CRaWl*
python3 -m synthexp.run_multiple --yaml_name barbell-crawl-1000
python3 -m synthexp.run_multiple --yaml_name barbell-crawl*-1000
python3 -m synthexp.run_multiple --yaml_name clique-crawl-1000
python3 -m synthexp.run_multiple --yaml_name clique-crawl*-1000
cd ..
