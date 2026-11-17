p="eval_harness/cli/main.py"
s=open(p,encoding="utf-8").read()
s=s.replace(
"""            cfg, dataset, result = run_experiment(CLIConfig.from_file(args.config))""",
"""            cfg, dataset, result, json_path, md_path = run_experiment(CLIConfig.from_file(args.config))""")
s=s.replace(
"""        print(f"Status: {result.status}")
        return EXIT_SUCCESS""",
"""        print(f"Status: {result.status}")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")
        return EXIT_SUCCESS""")
open(p,"w",encoding="utf-8").write(s)
print("done")
