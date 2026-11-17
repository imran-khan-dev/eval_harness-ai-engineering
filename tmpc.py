p="eval_harness/cli/config.py"
s=open(p,encoding="utf-8").read()
s=s.replace("def run_experiment(cfg: \"CLIConfig\"):", "def run_experiment(cfg: \"CLIConfig\") -> tuple[\"CLIConfig\", Any, Any]:")
open(p,"w",encoding="utf-8").write(s)
print("done")
