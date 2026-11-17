p="eval_harness/reporting/markdown_report.py"
s=open(p,encoding="utf-8").read()
old="    def write_markdown("
new="    def generate_quality_gate(self, result) -> str:\n        lines = [\"# Quality Gate Report\", \"\"]\n        lines.append(f\"Overall: {result.status} (passed={result.passed})\")\n        lines.append(\"\")\n        lines.append(\"| Rule | Status | Actual | Threshold | Explanation |\")\n        lines.append(\"|---|---|---|---|---|\")\n        for c in result.checks:\n            lines.append(f\"| {c.rule_id} | {c.status} | {c.actual_value} | {c.expected_threshold} | {c.explanation} |\")\n        return chr(10).join(lines)\n\n    def write_markdown("
s=s.replace(old,new,1)
open(p,"w",encoding="utf-8").write(s)
print("done")
