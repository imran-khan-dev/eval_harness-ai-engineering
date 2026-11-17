p="eval_harness/cli/main.py"
s=open(p,encoding="utf-8").read()
s=s.replace("def main(argv=None) -> int:","def main(argv: list[str] | None = None) -> int:")
open(p,"w",encoding="utf-8").write(s)
print("done")
