p="eval_harness/cli/config.py"
s=open(p,encoding="utf-8").read()
s=s.replace("    return system","    from typing import cast\n    return cast(SystemUnderEvaluation, system)")
open(p,"w",encoding="utf-8").write(s)
print("done")
