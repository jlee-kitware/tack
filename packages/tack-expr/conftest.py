# The test_*.py files here are scripts, run as `python test_more.py cuda`: they run their checks
# when imported, on the GPU by default, so pytest leaves them alone.
collect_ignore_glob = ["test_*.py"]

