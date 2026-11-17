p="eval_harness/reporting/markdown_report.py"
s=open(p,encoding="utf-8").read()
s=s.replace("    def generate_quality_gate(self, result) -> str:","    def generate_quality_gate(self, result: Any) -> str:")
open(p,"w",encoding="utf-8").write(s)
print("done")
