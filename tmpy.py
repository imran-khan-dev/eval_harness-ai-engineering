p="eval_harness/cli/config.py"
s=open(p,encoding="utf-8").read()
s=s.replace("    if not isinstance(system, SystemUnderEvaluation):\n        raise TypeError(\"factory did not return SystemUnderEvaluation\")",
            "    if not (hasattr(system, \"run\") and callable(getattr(system, \"run\"))):\n        raise TypeError(\"factory did not return SystemUnderEvaluation\")")
open(p,"w",encoding="utf-8").write(s)
print("done")
